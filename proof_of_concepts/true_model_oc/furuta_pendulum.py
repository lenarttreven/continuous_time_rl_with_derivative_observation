import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.config import config

from cucrl.optimal_cost.optimal_cost_ilqr import OptimalCost
from cucrl.simulator.simulator_costs import FurutaPendulum as FurutaPendulumCosts
from cucrl.simulator.simulator_dynamics import FurutaPendulum

config.update("jax_enable_x64", True)


def run_furuta_pendulum():
    time_horizon = (0, 5)
    system = FurutaPendulum(control_scaling=0.05 * jnp.eye(1))
    cost = FurutaPendulumCosts(control_scaling=0.05 * jnp.eye(1))
    initial_state = jnp.array([0.0, 0.0, jnp.pi, 0.0], dtype=jnp.float64)
    num_nodes = 1000
    optimizer = OptimalCost(
        simulator_dynamics=system,
        simulator_costs=cost,
        num_nodes=num_nodes,
        time_horizon=time_horizon,
    )

    out = optimizer.solve(initial_state)
    print("We completed the test successfully")
    ts = jnp.linspace(time_horizon[0], time_horizon[1], num_nodes + 1)
    plt.title("Xs")
    plt.plot(ts, out.xs, label="x")
    plt.legend()
    plt.show()
    plt.title("Us")
    plt.plot(ts[:-1], out.us)
    plt.show()
    return out


if __name__ == "__main__":
    out = run_furuta_pendulum()
    print(out.obj)
