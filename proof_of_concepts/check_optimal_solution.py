import time
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import cond
from trajax.optimizers import ILQR, ILQRHyperparams

from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.simulator.simulator_dynamics import SimulatorDynamics


def plot_optimal_trajectory(sim: SimulatorDynamics, sim_cc: SimulatorCostsAndConstraints, time_horizon: Tuple[int, int],
                            num_nodes: int, initial_state: jax.Array, title: str):
    dt = (time_horizon[1] - time_horizon[0]) / num_nodes
    initial_actions = jnp.zeros(shape=(num_nodes, sim.control_dim))

    def cost_fn(x, u, t, params):
        assert x.shape == (sim.state_dim,) and u.shape == (sim.control_dim,)

        def running_cost(x, u, t):
            return dt * sim_cc.running_cost(x, u)

        def terminal_cost(x, u, t):
            return sim_cc.terminal_cost(x, u)

        return cond(t == num_nodes, terminal_cost, running_cost, x, u, t)

    def dynamics_fn(x, u, t, params):
        assert x.shape == (sim.state_dim,) and u.shape == (sim.control_dim,)
        return x + sim.dynamics(x, u, t.reshape(1, )) * dt

    ilqr_params = ILQRHyperparams(maxiter=10000, make_psd=False, psd_delta=1e-2)
    ts = jnp.linspace(time_horizon[0], time_horizon[1], num_nodes + 1)

    optimizer = ILQR(cost_fn, dynamics_fn)

    start_time = time.time()
    out = optimizer.solve(None, None, initial_state, initial_actions, ilqr_params)
    print('Cost: ', out[2])
    print("Time taken: ", time.time() - start_time)
    plt.plot(ts, out.xs, label="xs")
    plt.plot(ts, jnp.concatenate([out.us, out.us[-1].reshape(1, -1)]), label="us")
    plt.title(title)
    plt.legend()
    plt.show()
    return out


def pendulum_oc():
    from cucrl.simulator.simulator_costs import Pendulum as PendulumCosts
    from cucrl.simulator.simulator_dynamics import Pendulum
    sim = Pendulum()
    sim_cc = PendulumCosts()

    plot_optimal_trajectory(sim, sim_cc, time_horizon=(0, 10),
                            num_nodes=1000, initial_state=jnp.array([jnp.pi / 2, 0.0]), title='Pendulum')


def furuta_pendulum_oc():
    from cucrl.simulator.simulator_costs import FurutaPendulum as FurutaPendulumCosts
    from cucrl.simulator.simulator_dynamics import FurutaPendulum
    sim = FurutaPendulum()
    sim_cc = FurutaPendulumCosts()

    plot_optimal_trajectory(sim, sim_cc, time_horizon=(0, 10),
                            num_nodes=1000, initial_state=jnp.array([0.0, 0.0, jnp.pi, 0.0], dtype=jnp.float64),
                            title='Furuta Pendulum')


def mountain_car_oc():
    from cucrl.simulator.simulator_costs import MountainCar as MountainCarCosts
    from cucrl.simulator.simulator_dynamics import MountainCar
    sim = MountainCar()
    sim_cc = MountainCarCosts()

    plot_optimal_trajectory(sim, sim_cc, time_horizon=(0, 1),
                            num_nodes=1000, initial_state=jnp.array([-jnp.pi / 6, 0.0]), title='Mountain Car')


def cartpole_oc():
    from cucrl.simulator.simulator_costs import CartPole as CartPoleCosts
    from cucrl.simulator.simulator_dynamics import CartPole
    sim = CartPole()
    sim_cc = CartPoleCosts()

    plot_optimal_trajectory(sim, sim_cc, time_horizon=(0, 10), num_nodes=10000,
                            initial_state=jnp.array([jnp.pi, 0.0, 0.0, 0.0], dtype=jnp.float64), title='Cart Pole')

def acrobot_oc():
    from cucrl.simulator.simulator_costs import Acrobot as DoublePendulumCosts
    from cucrl.simulator.simulator_dynamics import Acrobot
    sim = Acrobot()
    sim_cc = DoublePendulumCosts()

    out = plot_optimal_trajectory(sim, sim_cc, time_horizon=(0, 10), num_nodes=100,
                            initial_state=jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64), title='Double Pendulum')
    return out


if __name__ == '__main__':
    from jax.config import config

    config.update("jax_enable_x64", True)

    # cartpole_oc()
    # mountain_car_oc()
    out = acrobot_oc()
    # furuta_pendulum_oc()
    # pendulum_oc()
