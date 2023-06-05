from abc import abstractmethod
from typing import Tuple

import jax

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.trajectory_optimization.abstract_trajectory_optimization import (
    AbstractTrajectoryOptimization,
)
from cucrl.utils.classes import OfflinePlanningParams, DynamicsModel, OCSolution


class AbstractOfflinePlanner(AbstractTrajectoryOptimization):
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        time_horizon: Tuple[float, float],
        dynamics: AbstractDynamics,
        simulator_costs: SimulatorCostsAndConstraints,
    ):
        super().__init__(
            x_dim=x_dim,
            u_dim=u_dim,
            time_horizon=time_horizon,
            dynamics=dynamics,
            simulator_costs=simulator_costs,
        )
        pass

    @abstractmethod
    def plan_offline(
        self,
        dynamics_model: DynamicsModel,
        initial_parameters: OfflinePlanningParams,
        initial_conditions: jax.Array,
    ) -> OCSolution:
        pass
