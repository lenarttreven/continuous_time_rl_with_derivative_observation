from abc import ABC, abstractmethod
from typing import Tuple

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.simulator.simulator_dynamics import SimulatorDynamics
from cucrl.utils.classes import OCSolution, DynamicsIdentifier


class AbstractTrajectoryOptimization(ABC):
    def __init__(self, x_dim: int, u_dim: int, time_horizon: Tuple[float, float],
                 dynamics: AbstractDynamics | SimulatorDynamics, simulator_costs: SimulatorCostsAndConstraints):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.time_horizon = time_horizon
        self.dynamics = dynamics
        self.simulator_costs = simulator_costs

    @abstractmethod
    def example_dynamics_id(self) -> DynamicsIdentifier:
        pass

    @abstractmethod
    def example_OCSolution(self) -> OCSolution:
        pass
