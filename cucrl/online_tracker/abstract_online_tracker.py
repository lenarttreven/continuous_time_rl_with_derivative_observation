from abc import abstractmethod

import jax

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.main.config import TimeHorizon
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.trajectory_optimization.abstract_trajectory_optimization import AbstractTrajectoryOptimization
from cucrl.utils.classes import OCSolution, TrackingData, MPCParameters, DynamicsModel


class AbstractOnlineTracker(AbstractTrajectoryOptimization):
    def __init__(self, x_dim: int, u_dim: int, time_horizon: TimeHorizon, dynamics: AbstractDynamics,
                 simulator_costs: SimulatorCostsAndConstraints):
        super().__init__(x_dim=x_dim, u_dim=u_dim, time_horizon=time_horizon, dynamics=dynamics,
                         simulator_costs=simulator_costs)
        pass

    @abstractmethod
    def track_online(self, dynamics_model: DynamicsModel, initial_conditions: jax.Array, mpc_params: MPCParameters,
                     tracking_data: TrackingData, t_start) -> OCSolution:
        pass
