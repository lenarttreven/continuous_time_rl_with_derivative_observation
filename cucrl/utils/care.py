import time

import control
import jax.numpy as jnp
import numpy as np
from jax import random, jit, pure_callback
from jax.lax import while_loop
from jax.scipy.linalg import cho_solve, cho_factor
from scipy.linalg import solve_continuous_are, schur

from cucrl.utils.helper_functions import enable_print, block_print


@jit
def continuous_controller_pinv(a, b, q, r):
    return pure_callback(_continuous_controller_pinv, b.T, a, b, q, r)


def _continuous_controller_pinv(a, b, q, r):
    dim = a.shape[0]
    z = np.block([[a, -b @ r @ b.T], [-q, -a.T]])
    t, vecs, s_dim = schur(z, sort=lambda x: x.real <= 0.0)
    spanning_vecs = vecs[:, :dim]
    u_11 = spanning_vecs[:dim, :]
    u_21 = spanning_vecs[dim:, :]
    p = u_21 @ np.linalg.pinv(u_11)
    chol = cho_factor(r)
    return -cho_solve(chol, b.T @ p)


@jit
def continuous_controller(a, b, q, r):
    chol = cho_factor(r)
    b_in = b @ cho_solve(chol, b.T)
    p = sda_care(a, b_in, q, -1)
    return -cho_solve(chol, b.T @ p)


@jit
def continuous_controller_minimal_form(a, b, q, r):
    return pure_callback(_continuous_controller_minimal_form, b.T, a, b, q, r)


def _continuous_controller_minimal_form(a, b, q, r):
    block_print()
    state_dim, control_dim = b.shape
    system = control.ss(a, b, np.eye(state_dim), np.zeros(shape=(state_dim, control_dim)))
    min_system = control.minreal(system, tol=1e-8)
    k_bar, s, e = control.lqr(min_system.A, min_system.B, min_system.C.T @ q @ min_system.C, r)
    k = k_bar @ min_system.C.T
    enable_print()
    return -k


@jit
def continuous_controller_scipy(a, b, q, r):
    chol = cho_factor(r)
    p = pure_callback(solve_continuous_are, a, a, b, q, r)
    return -cho_solve(chol, b.T @ p)


@jit
def sda_care(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, g_c: float):
    n = a.shape[0]
    tol = 1e-15
    kmax = 500
    a_i = jnp.linalg.inv(a + g_c * jnp.eye(n))
    r = b @ a_i.T @ c
    s_1 = jnp.linalg.inv(a + g_c * jnp.eye(n) + r)
    e = s_1 @ (a - g_c * jnp.eye(n) + r)
    r = jnp.eye(n) - a_i @ (a - g_c * jnp.eye(n))
    g = s_1 @ b @ r.T
    p = - s_1.T @ c @ r
    err = 1
    k = 0

    init_val = (k, err, e, p, g)

    def body_fun(init_val):
        k, err, e, p, g = init_val
        igp = jnp.eye(n) - g @ p
        z = jnp.linalg.solve(igp.T, jnp.concatenate([e, p.T]).T).T
        e_1 = z[:n, :]
        p_1 = z[n:, :]
        g = g + e_1 @ g @ e.T
        p = p + e.T @ p_1.T @ e
        e = e_1 @ e
        err = jnp.linalg.norm(e, ord=1)
        return k + 1, err, e, p, g

    def cond_fun(init_val):
        k, err = init_val[0], init_val[1]
        return (err > tol) * (k < kmax)

    init_val = while_loop(cond_fun, body_fun, init_val)
    return init_val[3]


if __name__ == '__main__':
    from jax.config import config

    config.update("jax_enable_x64", True)
    config.update("jax_debug_infs", True)
    config.update("jax_debug_nans", True)
    np.set_printoptions(precision=2, suppress=True)

    dim_x = 4
    dim_u = 4
    key = random.PRNGKey(0)
    times = []

    for i in range(10):
        key, subkey = random.split(key)
        a = random.normal(key=subkey, shape=(dim_x, dim_x), dtype=jnp.float64)
        key, subkey = random.split(key)
        b = random.normal(key=subkey, shape=(dim_x, dim_u), dtype=jnp.float64)
        q, r = jnp.eye(dim_x), jnp.eye(dim_u)

        chol = cho_factor(r)
        b_in = b @ cho_solve(chol, b.T)

        x_s = continuous_controller_scipy(a, b, q, r)
        print(x_s.shape)
        start_time = time.time()
        x = _continuous_controller_pinv(a, b, q, r)
        print(x.shape)
        times.append(time.time() - start_time)
        print(jnp.sum(jnp.abs(x_s - x)))

    print('Median time: ', jnp.median(jnp.array(times)))
