from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from jax.config import config
from jax.lax import cond
from trajax.optimizers import CEMHyperparams, ILQRHyperparams, ILQR_with_CEM_warmstart

from cucrl.utils.greedy_point_selection import greedy_distance_maximization_1d_jit

config.update('jax_enable_x64', True)

x_dim = 2
u_dim = 1
T = 10

g = 9.81
l = 5.0

num_proposed_actions = 1000
num_action_points = 10

initial_state = jnp.array([jnp.pi / 2, 0.0])
initial_action_proposed = jnp.zeros(shape=(num_proposed_actions, u_dim))
initial_actions = jnp.zeros(shape=(num_action_points, u_dim))
control_low = jnp.array([-5.0]).reshape(1, )
control_high = jnp.array([5.0]).reshape(1, )


class TimeDiscretization(NamedTuple):
    """
    ts = jnp.linspace(0, T, num_steps)
    ts_delta = jnp.diff(ts)
    """
    ts: jax.Array
    ts_delta: jax.Array


def cost_fn(x, u, t, params: TimeDiscretization):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)

    def running_cost(x, u, t):
        return params.ts_delta[t] * (jnp.sum(x ** 2) + jnp.sum(u ** 2))

    def terminal_cost(x, u, t):
        return jnp.sum(x ** 2)

    return cond(t == num_proposed_actions, terminal_cost, running_cost, x, u, t)


def dynamics_fn(x, u, t, params: TimeDiscretization):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)
    x0 = x[1]
    x1 = u[0] + g / l * jnp.sin(x[0])
    return x + jnp.array([x0, x1]) * params.ts_delta[t]


# ts = jnp.arange(0, T, dt)

cem_params = CEMHyperparams(max_iter=10, sampling_smoothing=0.0, num_samples=200, evolution_smoothing=0.0,
                            elite_portion=0.1)
ilqr_params = ILQRHyperparams(maxiter=100, )

key = random.PRNGKey(0)
optimizer = ILQR_with_CEM_warmstart(cost_fn, dynamics_fn)

ts_proposed = jnp.linspace(0, T, num_proposed_actions + 1, dtype=jnp.float64)
time_discretization_proposed = TimeDiscretization(ts=ts_proposed, ts_delta=jnp.diff(ts_proposed))

# First we solve problem with very high resolution to get trajectory
out = optimizer.solve(time_discretization_proposed, time_discretization_proposed, initial_state,
                      initial_action_proposed,
                      control_low=control_low, control_high=control_high, ilqr_hyperparams=ilqr_params,
                      cem_hyperparams=cem_params, random_key=key)
print('Cost: ', out[2])

plt.plot(ts_proposed[:-1], out.xs[:-1, :], label='xs')
plt.title('iLQR warmup')
plt.step(ts_proposed[:-1], out.us, label='us', where='post')
plt.legend()
plt.show()


# We compute the best possible allocation of the control on the trajectory

def kernel(x, y):
    return jnp.exp(-jnp.sum((x - y) ** 2) / 1.0)


kernel_v = jax.vmap(kernel, in_axes=(0, None), out_axes=0)
kernel_m = jax.vmap(kernel_v, in_axes=(None, 0), out_axes=1)

zs = jnp.concatenate([out.xs[:-1], out.us], axis=1)
K = kernel_m(zs, zs)
k = jnp.arange(num_action_points + 1)

proposed_indices, all_indices = greedy_distance_maximization_1d_jit(K, k)

ts = jnp.linspace(0, T, num_proposed_actions + 1)[:-1][jnp.sort(proposed_indices)]
time_discretization = TimeDiscretization(ts=ts, ts_delta=jnp.diff(ts))


def cost_fn_one(x, u, t, params: TimeDiscretization):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)

    def running_cost(x, u, t):
        return params.ts_delta[t] * (jnp.sum(x ** 2) + jnp.sum(u ** 2))

    def terminal_cost(x, u, t):
        return jnp.sum(x ** 2)

    return cond(t == num_action_points, terminal_cost, running_cost, x, u, t)


optimizer = ILQR_with_CEM_warmstart(cost_fn_one, dynamics_fn)

# First we solve problem with very high resolution to get trajectory
out = optimizer.solve(time_discretization, time_discretization, initial_state, initial_actions,
                      control_low=control_low, control_high=control_high, ilqr_hyperparams=ilqr_params,
                      cem_hyperparams=cem_params, random_key=key)
print('Cost: ', out[2])

plt.plot(ts, out.xs, label='xs')
plt.title('iLQR warmup')
plt.step(ts[:-1], out.us, label='us', where='post')
plt.legend()
plt.show()
