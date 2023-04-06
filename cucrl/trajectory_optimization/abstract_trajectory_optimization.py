from abc import ABC
from typing import Tuple

import jax.numpy as jnp

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.simulator.simulator_dynamics import SimulatorDynamics
from cucrl.trajectory_optimization.numerical_computations.numerical_computation import get_numerical_computation
from cucrl.utils.classes import OCSolution, DynamicsIdentifier
from cucrl.utils.representatives import ExplorationStrategy, NumericalComputation


class AbstractTrajectoryOptimization(ABC):
    def __init__(self, state_dim: int, control_dim: int, num_nodes: int, time_horizon: Tuple[float, float],
                 dynamics: AbstractDynamics | SimulatorDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 numerical_method: NumericalComputation = NumericalComputation.LGL,
                 minimize_method='IPOPT', exploration_strategy=ExplorationStrategy.OPTIMISTIC_ETA):
        self.minimize_method = minimize_method
        self.exploration_strategy = exploration_strategy
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.num_nodes = num_nodes
        self.time_horizon = time_horizon
        self.dynamics = dynamics
        self.method = numerical_method
        self.simulator_costs = simulator_costs
        self.numerical_computation = get_numerical_computation(numerical_computation=self.method, num_nodes=num_nodes,
                                                               time_horizon=time_horizon)
        self.time = self.numerical_computation.time
        self.numerical_derivative = self.numerical_computation.numerical_derivative
        self.numerical_integral = self.numerical_computation.numerical_integral

        self.num_traj_params = (self.state_dim + self.control_dim) * self.num_nodes
        if self.exploration_strategy == ExplorationStrategy.OPTIMISTIC_ETA_TIME:
            self.example_dynamics_id = DynamicsIdentifier(eta=jnp.ones(shape=(self.num_nodes, self.state_dim)),
                                                          idx=jnp.ones(shape=(), dtype=jnp.uint32),
                                                          key=jnp.ones(shape=(2,), dtype=jnp.uint32))
        else:
            self.example_dynamics_id = DynamicsIdentifier(eta=jnp.zeros(shape=(self.state_dim,)),
                                                          idx=jnp.ones(shape=(), dtype=jnp.uint32),
                                                          key=jnp.ones(shape=(2,), dtype=jnp.uint32))
        self.example_OCSolution = OCSolution(ts=self.time,
                                             xs=jnp.ones(shape=(self.num_nodes, self.state_dim)),
                                             us=jnp.ones(shape=(self.num_nodes, self.control_dim)),
                                             opt_value=jnp.ones(shape=()),
                                             dynamics_id=self.example_dynamics_id)
