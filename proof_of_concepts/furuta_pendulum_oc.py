import jax.numpy as jnp
from jax import random

from cucrl.optimal_cost.optimal_cost import OptimalCost
from cucrl.simulator.simulator_costs import FurutaPendulum as FurutaPendulumCosts
from cucrl.simulator.simulator_dynamics import FurutaPendulum

from jax.config import config
config.update("jax_enable_x64", True)


def run_furuta_pendulum():
    state_dim = 4
    action_dim = 1
    time_horizon = (0, 10)
    system = FurutaPendulum()
    cost = FurutaPendulumCosts()
    initial_state = jnp.array([0.0, 0.0, jnp.pi, 0.0], dtype=jnp.float64)
    num_nodes = 50
    optimizer = OptimalCost(simulator_dynamics=system, simulator_costs=cost, num_nodes=num_nodes,
                            time_horizon=time_horizon)

    out = optimizer.solve(initial_state)
    print('We completed the test successfully')
    return out


if __name__ == '__main__':
    print(run_furuta_pendulum())