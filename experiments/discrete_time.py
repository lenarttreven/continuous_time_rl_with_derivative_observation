import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap
from jax.lax import cond, scan
from trajax.optimizers import ILQRHyperparams, ILQR

from cucrl.simulator.simulator_costs import Pendulum as PendulumCost
from cucrl.simulator.simulator_dynamics import Pendulum
from jax.config import config

config.update("jax_enable_x64", True)

simulator = Pendulum()
simulator_cost = PendulumCost()

x_dim = 2
u_dim = 1

num_action_points = 10
T = 10
dt = T / num_action_points

initial_state = jnp.array([jnp.pi / 2, 0.0])
initial_actions = jnp.zeros(shape=(num_action_points, u_dim))
num_steps = initial_actions.shape[0]

# Here we implement low level integration of the system at much higher frequency
num_low_steps = 300
time_span = dt


@partial(jit, static_argnums=(3, 4))
def integrate(x: jax.Array, u: jax.Array, length, time_span: float, num_low_steps: int):
    _dt = time_span / num_low_steps

    def f(_x, _):
        x_dot = simulator.dynamics(
            _x,
            u,
            jnp.array(1.0).reshape(
                1,
            ),
        )
        return _x + x_dot * _dt, None

    x_next, _ = scan(f, x, None, length=num_low_steps)
    return x_next


def cost_fn(x, u, t, length):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)

    def running_cost(x, u, t):
        return dt * simulator_cost.running_cost(x, u)

    def terminal_cost(x, u, t):
        return simulator_cost.terminal_cost(x, u)

    return cond(t == num_steps, terminal_cost, running_cost, x, u, t)


def dynamics_fn(x, u, t, length):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)
    return integrate(
        x, u, length=length, time_span=time_span, num_low_steps=num_low_steps
    )


ilqr = ILQR(cost_fn, dynamics_fn)

ts = jnp.arange(0, T, dt)

ilqr_params = ILQRHyperparams(maxiter=100)

dynamics_params = jnp.array(5.0)
cost_params = dynamics_params

start_time = time.time()
out = ilqr.solve(
    dynamics_params, cost_params, initial_state, initial_actions, ilqr_params
)
print("Cost: ", out[2])
print("Time taken: ", time.time() - start_time)


# Evaluation:


@partial(jit, static_argnums=(3, 4))
def integrate_eval(
    x: jax.Array, u: jax.Array, length, time_span: float, num_low_steps: int
):
    _dt = time_span / num_low_steps

    def f(_x, _):
        x_dot = simulator.dynamics(
            _x,
            u,
            jnp.array(1.0).reshape(
                1,
            ),
        )
        _x_next = _x + x_dot * _dt
        return _x_next, _x_next

    x_next, xs_next = scan(f, x, None, length=num_low_steps)
    return x_next, xs_next


def dynamics_fn_eval(x, u, length):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)
    return integrate_eval(
        x, u, length=length, time_span=time_span, num_low_steps=num_low_steps
    )


def rollout_eval(U, x0, length):
    def dynamics_for_scan(x, u):
        x_next, xs_next = dynamics_fn_eval(x, u, length)
        return x_next, xs_next

    return scan(dynamics_for_scan, x0, U)


x_last, xs_all = rollout_eval(out.us, initial_state, dynamics_params)
us_all = jnp.repeat(out.us[:, None, :], repeats=num_low_steps, axis=1)

xs_all = xs_all.reshape(-1, x_dim)
us_all = us_all.reshape(-1, u_dim)
xs_all = jnp.concatenate([initial_state[None, :], xs_all])

eval_dt = dt / num_low_steps


def running_cost(x, u):
    return eval_dt * simulator_cost.running_cost(x, u)


def terminal_cost(x, u):
    return simulator_cost.terminal_cost(x, u)


def cost_fn_eval(xs, us):
    _running_cost = jnp.sum(vmap(running_cost)(xs[:-1], us))
    _terminal_cost = terminal_cost(xs[-1], us[-1])
    return _running_cost + _terminal_cost


true_cost = cost_fn_eval(xs_all, us_all)
print("True cost: ", true_cost)
print(
    us_all.shape,
    xs_all.shape,
)

# ts_eval = jnp.linspace(0, T, num_low_steps * num_action_points + 1)
# plt.plot(ts_eval, xs_all, label='xs')
# plt.plot(ts_eval[:-1], us_all, label='us')
# plt.legend()
# plt.show()
