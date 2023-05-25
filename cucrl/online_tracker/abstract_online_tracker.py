from abc import abstractmethod
from typing import Tuple

import jax

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.trajectory_optimization.abstract_trajectory_optimization import AbstractTrajectoryOptimization
from cucrl.utils.classes import OCSolution, TrackingData, MPCParameters, DynamicsModel
from cucrl.utils.representatives import ExplorationStrategy, NumericalComputation


class AbstractOnlineTracker(AbstractTrajectoryOptimization):
    def __init__(self, x_dim: int, u_dim: int, num_nodes: int, time_horizon: Tuple[float, float],
                 dynamics: AbstractDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 numerical_method=NumericalComputation.LGL, minimize_method='IPOPT',
                 exploration_strategy=ExplorationStrategy.OPTIMISTIC_ETA):
        super().__init__(x_dim=x_dim, u_dim=u_dim, num_nodes=num_nodes, time_horizon=time_horizon,
                         dynamics=dynamics, simulator_costs=simulator_costs, numerical_method=numerical_method,
                         minimize_method=minimize_method, exploration_strategy=exploration_strategy)
        pass

    @abstractmethod
    def track_online(self, dynamics_model: DynamicsModel, initial_conditions: jax.Array, mpc_params: MPCParameters,
                     tracking_data: TrackingData, t_start) -> OCSolution:
        pass
