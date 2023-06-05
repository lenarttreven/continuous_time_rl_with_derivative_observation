import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import cond
from trajax.optimizers import ILQRHyperparams, ILQR

from cucrl.main.config import Scaling
from cucrl.simulator.simulator_dynamics import RaceCar
from simulation import prepare_tracking_data

if __name__ == "__main__":
    spline = prepare_tracking_data()

    state_dim = 6
    action_dim = 2
    scaling = Scaling(
        state_scaling=jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 10.0, 1.0])),
        control_scaling=jnp.eye(action_dim),
        time_scaling=jnp.ones(shape=(1,)),
    )

    dynamics_model = RaceCar(state_scaling=scaling.state_scaling)
    # costs = RaceCarCost(state_scaling=scaling.state_scaling)

    x_dim = dynamics_model.state_dim
    u_dim = dynamics_model.control_dim
    num_nodes = 1000
    T = 10
    dt = T / num_nodes
    ts = jnp.linspace(0, T, num_nodes + 1)

    def running_cost(x, u, t):
        # Need to transform t to time
        t = t * T / num_nodes
        return dt * (jnp.sum((spline(t.reshape()) - x) ** 2) + jnp.sum(u**2))

    def terminal_cost(x, u, t):
        return jnp.zeros(shape=())

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
        plt.plot(ts[:-1], out.us[:, i], label=r"u_{}".format(i))
    plt.legend()
    plt.show()

    print("Final cost: ", out.obj)
    print("Last state", out.xs[-1])
    print("Last action", out.us[-1])
