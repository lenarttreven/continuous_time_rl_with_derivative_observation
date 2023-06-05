import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import cond
from trajax.optimizers import ILQRHyperparams, ILQR

from cucrl.simulator.simulator_costs import RaceCar as RaceCarCost
from cucrl.simulator.simulator_dynamics import RaceCar
from cucrl.main.config import Scaling


def run_race_car():
    state_dim = 6
    action_dim = 2
    scaling = Scaling(
        state_scaling=jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 10.0, 1.0])),
        control_scaling=jnp.eye(action_dim),
        time_scaling=jnp.ones(shape=(1,)),
    )

    dynamics_model = RaceCar(state_scaling=scaling.state_scaling)
    costs = RaceCarCost(state_scaling=scaling.state_scaling)

    x_dim = dynamics_model.state_dim
    u_dim = dynamics_model.control_dim
    num_nodes = 1000
    T = 10
    dt = T / num_nodes
    ts = jnp.linspace(0, T, num_nodes + 1)

    def running_cost(x, u, t):
        return dt * costs.running_cost(x, u)

    def terminal_cost(x, u, t):
        return costs.terminal_cost(x, u)

    def cost(x, u, t, params=None):
        return cond(
            t == num_nodes,
            terminal_cost,
            running_cost,
            x,
            u,
            t.reshape(
                1,
            ),
        )

    def dynamics(x, u, t, params=None):
        assert x.shape == (x_dim,) and u.shape == (u_dim,)
        return x + dt * dynamics_model.dynamics(
            x,
            u,
            t.reshape(
                1,
            ),
        )

    ilqr_params = ILQRHyperparams(maxiter=100)
    initial_state = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64)
    initial_actions = 0.01 * jnp.ones(shape=(num_nodes, u_dim))

    ilqr = ILQR(cost, dynamics)
    out = ilqr.solve(None, None, initial_state, initial_actions, ilqr_params)
    for i in range(x_dim):
        plt.plot(ts, out.xs[:, i], label=r"x_{}".format(i))
    plt.legend()
    plt.show()

    for i in range(u_dim):
        plt.plot(ts[:-1], jnp.tanh(out.us[:, i]), label=r"u_{}".format(i))
    plt.legend()
    plt.show()

    print("Final cost: ", out.obj)
    print("Last state", out.xs[-1])
    print("Last action", out.us[-1])
    print(
        "Distance to target at the end: ",
        jnp.linalg.norm(out.xs[-1] - costs.state_target),
    )


if __name__ == "__main__":
    from jax.config import config

    config.update("jax_enable_x64", True)
    run_race_car()
