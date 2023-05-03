import jax.numpy as jnp
from jax.config import config
import matplotlib.pyplot as plt

from cucrl.optimal_cost.optimal_cost_ilqr import OptimalCost
from cucrl.simulator.simulator_costs import QuadrotorEuler as QuadrotorEulerCosts
from cucrl.simulator.simulator_dynamics import QuadrotorEuler

config.update("jax_enable_x64", True)


def run():
    time_horizon = (0, 15)
    state_scaling = jnp.diag(jnp.array([1, 1, 1, 1, 1, 1, 10, 10, 1, 10, 10, 1], dtype=jnp.float64))

    system = QuadrotorEuler(state_scaling=state_scaling)
    cost = QuadrotorEulerCosts(state_scaling=state_scaling)
    initial_state = jnp.array([1.0, 1.0, 1.0,
                               0., 0., 0.,
                               0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0], dtype=jnp.float64)
    num_nodes = 100
    optimizer = OptimalCost(simulator_dynamics=system, simulator_costs=cost, num_nodes=num_nodes,
                            time_horizon=time_horizon)

    out = optimizer.solve(initial_state)
    print('We completed the test successfully')
    ts = jnp.linspace(time_horizon[0], time_horizon[1], num_nodes + 1)
    xs = out.xs
    us = out.us
    plt.plot(ts, xs, label='xs')
    plt.plot(ts[:-1], us, label='us')
    plt.legend()
    plt.show()
    return out


if __name__ == '__main__':
    out = run()
