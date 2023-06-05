import jax.numpy as jnp
from cyipopt import minimize_ipopt
from jax import jit, grad
from jax.config import config

from cucrl.simulator.simulator_dynamics import QuadrotorEuler

config.update("jax_enable_x64", True)
config.update("jax_debug_infs", True)
config.update("jax_debug_nans", True)

state_scaling = jnp.diag(
    jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float64)
)
control_scaling = jnp.diag(jnp.array([1, 1e2, 1e2, 1e2], dtype=jnp.float64))
system = QuadrotorEuler(control_scaling=control_scaling)

state_target = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
action_target = jnp.array([0.018, 0.0, 0.0, 0.0], dtype=jnp.float64)


def equation(u):
    assert u.shape == (4,)
    return jnp.mean(system.dynamics(state_target, u, jnp.zeros(shape=(1,))) ** 2)


objective_grad = jit(grad(equation))

u0 = jnp.zeros(shape=(4,))
out = minimize_ipopt(equation, x0=u0, jac=objective_grad, options={"disp": 4})

print(system.dynamics(state_target, action_target, jnp.zeros(shape=(1,))))
