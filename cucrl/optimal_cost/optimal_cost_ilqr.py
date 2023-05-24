from typing import Tuple

import jax
import jax.numpy as jnp
from jax.lax import cond
from trajax.optimizers import ILQRHyperparams, ILQR

from cucrl.simulator.simulator_costs import Bicycle as BicycleCost
from cucrl.simulator.simulator_costs import FurutaPendulum as FurutaPendulumCost
from cucrl.simulator.simulator_costs import MountainCar as MountainCarCost
from cucrl.simulator.simulator_costs import Pendulum as PendulumCost
from cucrl.simulator.simulator_costs import RaceCar as RaceCarCost
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.simulator.simulator_dynamics import Pendulum, Bicycle, RaceCar, FurutaPendulum, MountainCar
from cucrl.simulator.simulator_dynamics import SimulatorDynamics


class OptimalCost:
    def __init__(self, simulator_dynamics: SimulatorDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 time_horizon: Tuple[float, float], num_nodes: int = 50):
        self.simulator_dynamics = simulator_dynamics
        self.simulator_costs = simulator_costs
        self.num_nodes = num_nodes
        self.time_horizon = time_horizon

        self.dt = (self.time_horizon[1] - self.time_horizon[0]) / num_nodes
        self.ts = jnp.linspace(self.time_horizon[0], self.time_horizon[1], num_nodes + 1)
        self.ilqr_params = ILQRHyperparams(maxiter=1000, make_psd=False, psd_delta=1e0)
        self.ilqr = ILQR(self.cost, self.dynamics)
        self.results = None

    def running_cost(self, x, u, t):
        return self.dt * self.simulator_costs.running_cost(x, u)

    def terminal_cost(self, x, u, t):
        return self.simulator_costs.terminal_cost(x, u)

    def cost(self, x, u, t, params=None):
        return cond(t == self.num_nodes, self.terminal_cost, self.running_cost, x, u, t.reshape(1, ))

    def dynamics(self, x, u, t, params=None):
        assert x.shape == (self.simulator_dynamics.state_dim,) and u.shape == (self.simulator_dynamics.control_dim,)
        return x + self.dt * self.simulator_dynamics.dynamics(x, u, t.reshape(1, ))

    def solve(self, initial_conditions: jax.Array):
        initial_actions = 0.01 * jnp.ones(shape=(self.num_nodes, self.simulator_dynamics.control_dim,))
        out = self.ilqr.solve(None, None, initial_conditions, initial_actions, self.ilqr_params)
        self.results = out
        return out


if __name__ == "__main__":
    from jax.config import config

    config.update("jax_enable_x64", True)
    # Pendulum
    system = Pendulum()
    costs = PendulumCost()
    optimal_cost = OptimalCost(simulator_dynamics=system, simulator_costs=costs, time_horizon=(0.0, 10.0),
                               num_nodes=10000)
    out = optimal_cost.solve(jnp.array([0.5 * jnp.pi, 0]))
    print("Pendulum optimal objective:", out)

    # Bicycle
    system = Bicycle()
    costs = BicycleCost()
    optimal_cost = OptimalCost(simulator_dynamics=system, simulator_costs=costs, time_horizon=(0.0, 10.0),
                               num_nodes=10000)
    out = optimal_cost.solve(jnp.array([0.0, 0.0, 0.0, 0.0]))
    print("Bicycle optimal objective:", out)

    # RaceCar
    system = RaceCar()
    costs = RaceCarCost()
    optimal_cost = OptimalCost(simulator_dynamics=system, simulator_costs=costs, time_horizon=(0.0, 10.0),
                               num_nodes=10000)
    out = optimal_cost.solve(jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64))
    print("RaceCar optimal objective:", out)

    # FurutaPendulum
    system = FurutaPendulum()
    costs = FurutaPendulumCost()
    optimal_cost = OptimalCost(simulator_dynamics=system, simulator_costs=costs, time_horizon=(0.0, 10.0),
                               num_nodes=10000)
    out = optimal_cost.solve(jnp.array([0.0, 0.0, jnp.pi, 0.0], dtype=jnp.float64))
    print("FurutaPendulum optimal objective:", out)

    # MountainCar
    system = MountainCar()
    costs = MountainCarCost()
    optimal_cost = OptimalCost(simulator_dynamics=system, simulator_costs=costs, time_horizon=(0.0, 10.0),
                               num_nodes=10000)
    out = optimal_cost.solve(jnp.array([-jnp.pi / 6, 0.0]))
    print("MountainCar optimal objective:", out)
