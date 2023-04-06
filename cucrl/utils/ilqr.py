from abc import ABC, abstractmethod
from typing import NamedTuple, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit
from jax.tree_util import register_pytree_node_class

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.simulator.simulator_costs import Pendulum as PendulumCost
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.simulator.simulator_dynamics import Pendulum


class LinearDynamics(NamedTuple):
    f_x: jnp.array  # A
    f_u: jnp.array  # B

    def __call__(self, x, u, k=None):
        f_x, f_u = self
        return f_x @ x + f_u @ u if k is None else self[k](x, u)

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)


class QuadraticCost(NamedTuple):
    c: jnp.array  # c
    c_x: jnp.array  # q
    c_u: jnp.array  # r
    c_xx: jnp.array  # Q
    c_uu: jnp.array  # R
    c_ux: jnp.array  # H.T

    @classmethod
    def from_pure_quadratic(cls, c_xx, c_uu, c_ux):
        return cls(
            jnp.zeros((c_xx.shape[:-2])),
            jnp.zeros(c_xx.shape[:-1]),
            jnp.zeros(c_uu.shape[:-1]),
            c_xx,
            c_uu,
            c_ux,
        )

    def __call__(self, x, u, k=None):
        c, c_x, c_u, c_xx, c_uu, c_ux = self
        return c + c_x @ x + c_u @ u + x @ c_xx @ x / 2 + u @ c_uu @ u / 2 + u @ c_ux @ x if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)


class QuadraticStateCost(NamedTuple):
    v: jnp.array  # p (scalar)
    v_x: jnp.array  # p (vector)
    v_xx: jnp.array  # P

    @classmethod
    def from_pure_quadratic(cls, v_xx):
        return cls(
            jnp.zeros(v_xx.shape[:-2]),
            jnp.zeros(v_xx.shape[:-1]),
            v_xx,
        )

    def __call__(self, x, k=None):
        v, v_x, v_xx = self
        return v + v_x @ x + x @ v_xx @ x / 2 if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)


class AffinePolicy(NamedTuple):
    l: jnp.array  # l
    l_x: jnp.array  # L

    def __call__(self, x, k=None):
        l, l_x = self
        return l + l_x @ x if k is None else self[k](x)

    def __getitem__(self, key):
        return jax.tree_map(lambda x: x[key], self)


def rollout_state_feedback_policy(dynamics, policy, x0, step_range, x_nom=None, u_nom=None):
    def scan_fn(x, k):
        u = policy(x, k) if x_nom is None else u_nom[k] + policy(x - x_nom[k], k)
        x1 = dynamics(x, u, k)
        return (x1, (x1, u))

    xs, us = jax.lax.scan(scan_fn, x0, step_range)[1]
    return jnp.concatenate([x0[None], xs]), us


def riccati_step(
        current_step_dynamics: LinearDynamics,
        current_step_cost: QuadraticCost,
        next_state_value: QuadraticStateCost,
):
    f_x, f_u = current_step_dynamics
    c, c_x, c_u, c_xx, c_uu, c_ux = current_step_cost
    v, v_x, v_xx = next_state_value

    q = c + v
    q_x = c_x + f_x.T @ v_x
    q_u = c_u + f_u.T @ v_x
    q_xx = c_xx + f_x.T @ v_xx @ f_x
    q_uu = c_uu + f_u.T @ v_xx @ f_u
    q_ux = c_ux + f_u.T @ v_xx @ f_x

    l = -jnp.linalg.solve(q_uu, q_u)
    l_x = -jnp.linalg.solve(q_uu, q_ux)

    current_state_value = QuadraticStateCost(
        q - l.T @ q_uu @ l / 2,
        q_x - l_x.T @ q_uu @ l,
        q_xx - l_x.T @ q_uu @ l_x,
    )
    current_step_optimal_policy = AffinePolicy(l, l_x)
    return current_state_value, current_step_optimal_policy


@jax.jit
def linear_quadratic_regulator(Qs, Rs, Hs, As, Bs):
    final_state_value = QuadraticStateCost.from_pure_quadratic(Qs[-1])
    dynamics = LinearDynamics(As, Bs)
    cost = QuadraticCost.from_pure_quadratic(Qs[:-1], Rs, Hs)

    def scan_fn(next_state_value, current_step_dynamics_cost):
        current_step_dynamics, current_step_cost = current_step_dynamics_cost
        current_state_value, current_step_optimal_policy = riccati_step(
            current_step_dynamics,
            current_step_cost,
            next_state_value,
        )
        return current_state_value, (current_state_value, current_step_optimal_policy)

    value_functions, optimal_policy = jax.lax.scan(scan_fn, final_state_value, (dynamics, cost), reverse=True)[1]
    return jax.tree_map(lambda x, y: jnp.concatenate([x, y]), value_functions, final_state_value[None]), optimal_policy


def ensure_positive_definite(a, eps=1e-3):
    w, v = jnp.linalg.eigh(a)
    return (v * jnp.maximum(w, eps)) @ v.T


class TotalCost(NamedTuple):
    running_cost: Callable
    terminal_cost: Callable

    def __call__(self, xs, us):
        step_range = jnp.arange(us.shape[0])
        return jnp.sum(jax.vmap(self.running_cost)(xs[:-1], us, step_range)) + self.terminal_cost(xs[-1])


@jax.jit
def iterative_linear_quadratic_regulator(dynamics, total_cost, x0, u_guess, maxiter=100, atol=1e-3):
    running_cost, terminal_cost = total_cost
    n, (N, m) = x0.shape[-1], u_guess.shape
    step_range = jnp.arange(N)

    xs_iterates, us_iterates = jnp.zeros((maxiter, N + 1, n)), jnp.zeros((maxiter, N, m))
    xs, us = rollout_state_feedback_policy(dynamics, lambda x, k: u_guess[k], x0, step_range)
    xs_iterates, us_iterates = xs_iterates.at[0].set(xs), us_iterates.at[0].set(us)
    j_curr = total_cost(xs, us)
    value_functions_iterates = QuadraticStateCost.from_pure_quadratic(jnp.zeros((maxiter, N + 1, n, n)))

    def continuation_criterion(loop_vars):
        i, _, _, j_curr, j_prev, _ = loop_vars
        return (j_curr < j_prev - atol) & (i < maxiter)

    def ilqr_iteration(loop_vars):
        i, xs_iterates, us_iterates, j_curr, j_prev, value_functions_iterates = loop_vars
        xs, us = xs_iterates[i], us_iterates[i]

        f_x, f_u = jax.vmap(jax.jacobian(dynamics, (0, 1)))(xs[:-1], us, step_range)
        c = jax.vmap(running_cost)(xs[:-1], us, step_range)
        c_x, c_u = jax.vmap(jax.grad(running_cost, (0, 1)))(xs[:-1], us, step_range)
        (c_xx, c_xu), (c_ux, c_uu) = jax.vmap(jax.hessian(running_cost, (0, 1)))(xs[:-1], us, step_range)
        v, v_x, v_xx = terminal_cost(xs[-1]), jax.grad(terminal_cost)(xs[-1]), jax.hessian(terminal_cost)(xs[-1])

        # Ensure quadratic cost terms are positive definite.
        c_zz = jnp.block([[c_xx, c_xu], [c_ux, c_uu]])
        c_zz = jax.vmap(ensure_positive_definite)(c_zz)
        c_xx, c_uu, c_ux = c_zz[:, :n, :n], c_zz[:, -m:, -m:], c_zz[:, -m:, :n]
        v_xx = ensure_positive_definite(v_xx)

        linearized_dynamics = LinearDynamics(f_x, f_u)
        quadratized_running_cost = QuadraticCost(c, c_x, c_u, c_xx, c_uu, c_ux)
        quadratized_terminal_cost = QuadraticStateCost(v, v_x, v_xx)

        def scan_fn(next_state_value, current_step_dynamics_cost):
            current_step_dynamics, current_step_cost = current_step_dynamics_cost
            current_state_value, current_step_policy = riccati_step(
                current_step_dynamics,
                current_step_cost,
                next_state_value,
            )
            return current_state_value, (current_state_value, current_step_policy)

        value_functions, policy = jax.lax.scan(scan_fn,
                                               quadratized_terminal_cost,
                                               (linearized_dynamics, quadratized_running_cost),
                                               reverse=True)[1]
        value_functions_iterates = jax.tree_map(lambda x, xi, xiN: x.at[i].set(jnp.concatenate([xi, xiN[None]])),
                                                value_functions_iterates, value_functions, quadratized_terminal_cost)

        def rollout_linesearch_policy(alpha):
            # Note that we roll out the true `dynamics`, not the `linearized_dynamics`!
            l, l_x = policy
            return rollout_state_feedback_policy(dynamics, AffinePolicy(alpha * l, l_x), x0, step_range, xs, us)

        # Backtracking line search (step sizes evaluated in parallel).
        all_xs, all_us = jax.vmap(rollout_linesearch_policy)(0.5 ** jnp.arange(16))
        js = jax.vmap(total_cost)(all_xs, all_us)
        a = jnp.argmin(js)
        j = js[a]
        xs_iterates = xs_iterates.at[i + 1].set(jnp.where(j < j_curr, all_xs[a], xs))
        us_iterates = us_iterates.at[i + 1].set(jnp.where(j < j_curr, all_us[a], us))
        return i + 1, xs_iterates, us_iterates, jnp.minimum(j, j_curr), j_curr, value_functions_iterates

    i, xs_iterates, us_iterates, j_curr, j_prev, value_functions_iterates = jax.lax.while_loop(
        continuation_criterion, ilqr_iteration,
        (0, xs_iterates, us_iterates, j_curr, jnp.inf, value_functions_iterates))

    return {
        "optimal_trajectory": (xs_iterates[i], us_iterates[i]),
        "optimal_cost": j_curr,
        "num_iterations": i,
        "trajectory_iterates": (xs_iterates, us_iterates),
        "value_functions_iterates": value_functions_iterates
    }


class ApplyDynamics(ABC):

    @abstractmethod
    def apply(self, params, stats, x, u, data_stats) -> jnp.array:
        pass


@register_pytree_node_class
class NonlinearDynamics:
    def __init__(self, dynamics: ApplyDynamics):
        self.dynamics = dynamics

        self.params = None
        self.data_stats = None
        self.stats = None

    def tree_flatten(self):
        children = (self.params, self.stats, self.data_stats)  # arrays / dynamic values
        aux_data = {'dynamics': self.dynamics}  # static values
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        params, stats, data_stats = children
        new_class = cls(dynamics=aux_data['dynamics'])
        new_class.params = params
        new_class.stats = stats
        new_class.data_stats = data_stats
        return new_class

    def update(self, params, stats, data_stats):
        self.params = params
        self.stats = stats
        self.data_stats = data_stats

    @jit
    def __call__(self, x, u, k):
        assert x.shape == (state_dim,) and u.shape == (action_dim,)
        return x + h * self.dynamics.apply(self.params, self.stats, x, u, self.data_stats)


class ILOR:
    def __init__(self, dynamics: AbstractDynamics, system_cost: SimulatorCostsAndConstraints):
        self.dynamics = dynamics
        self.system_cost = system_cost


if __name__ == '__main__':
    from jax.config import config

    config.update("jax_enable_x64", True)
    system = Pendulum()
    system_cost = PendulumCost()

    state_dim = 2
    action_dim = 1

    x0 = jnp.array([jnp.pi / 2, 0.0], dtype=jnp.float64)

    h = 0.01
    T = 10
    N = int(T / h)


    @register_pytree_node_class
    class Pendulum:
        def __init__(self, system_params=jnp.array([5.0, 9.81], jnp.float64)):
            self.system_params = system_params

        def tree_flatten(self):
            children = self.system_params  # arrays / dynamic values
            aux_data = {}  # static values
            return children, aux_data

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            params = children
            new_class = cls(system_params=params)
            return new_class

        def __call__(self, x, u, k):
            return x + h * jnp.array(
                [x[1], self.system_params[1] / self.system_params[0] * jnp.sin(x[0]) + u.reshape()])


    class SimpleRunningCost(NamedTuple):
        gain: float = 1.0

        def __call__(self, x, u, k):
            assert x.shape == (state_dim,) and u.shape == (action_dim,)
            return h * system_cost.running_cost(x, u)


    class SimpleTerminalCost(NamedTuple):
        def __call__(self, x):
            assert x.shape == (state_dim,)
            return system_cost.terminal_cost(x, jnp.zeros(shape=(action_dim,)))


    u_guess = jnp.zeros((N, action_dim))
    import time

    start_time = time.time()
    solution = iterative_linear_quadratic_regulator(
        Pendulum(system_params=jnp.array([5.0, 9.81])),
        TotalCost(SimpleRunningCost(), SimpleTerminalCost()),
        x0,
        u_guess,
    )
    print(time.time() - start_time)
    start_time = time.time()
    solution = iterative_linear_quadratic_regulator(
        Pendulum(system_params=jnp.array([5.0, 9.81])),
        TotalCost(SimpleRunningCost(), SimpleTerminalCost()),
        x0,
        u_guess,
    )
    print(time.time() - start_time)
    i = solution["num_iterations"]

    xs, us = solution['optimal_trajectory']
    ts = jnp.linspace(0, T, N + 1)

    plt.plot(ts, xs, label='State')
    plt.plot(ts[:-1], us, label='Control')
    plt.show()
