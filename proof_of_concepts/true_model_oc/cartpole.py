import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.config import config

from cucrl.optimal_cost.optimal_cost_ilqr import OptimalCost
from cucrl.simulator.simulator_costs import CartPole as CartPoleCosts
from cucrl.simulator.simulator_dynamics import CartPole

config.update("jax_enable_x64", True)


def run_cartpole():
    time_horizon = (0, 10)
    system = CartPole()
    cost = CartPoleCosts()
    initial_state = jnp.array([jnp.pi, 0.0, 0.0, 0.0], dtype=jnp.float64)
    num_nodes = 200
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
    print('Optimal cost: ', out.obj)


if __name__ == '__main__':
    run_cartpole()
