from abc import ABC, abstractmethod
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import random

from cucrl.main.config import InteractionConfig, Scaling
from cucrl.main.data_stats import Normalizer
from cucrl.offline_planner.abstract_offline_planner import AbstractOfflinePlanner
from cucrl.utils.classes import IntegrationCarry, TrackingData, MPCCarry, DynamicsModel
from cucrl.utils.classes import OfflinePlanningData
from cucrl.utils.helper_functions import AngleLayerDynamics

# PolicyOit is the output of the policy and represents: [us, t_next, events]
PolicyOut = Tuple[jax.Array, jax.Array, IntegrationCarry]


class Policy(ABC):
    def __init__(self, x_dim: int, u_dim: int, initial_condition: List[jnp.ndarray], normalizer: Normalizer,
                 offline_planner: AbstractOfflinePlanner, interaction_config: InteractionConfig,
                 angle_layer: AngleLayerDynamics, scaling: Scaling):
        self.scaling = scaling
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.initial_conditions = jnp.stack(initial_condition)
        self.num_traj = len(self.initial_conditions)
        self.offline_planner = offline_planner
        self.interaction_config = interaction_config
        self.initial_control = self.prepare_initial_control()
        self.normalizer = normalizer
        self.angle_layer = angle_layer

    def prepare_initial_control(self):
        if type(self.interaction_config.policy.initial_control) == float:
            def initial_control(x, t):
                return jnp.array([self.interaction_config.policy.initial_control] * self.u_dim, dtype=jnp.float64)

            return initial_control
        elif callable(self.interaction_config.policy.initial_control):
            return self.interaction_config.policy.initial_control
        elif isinstance(self.interaction_config.policy.initial_control, jnp.ndarray):
            assert self.interaction_config.policy.initial_control.shape == (self.u_dim,)

            def initial_control(x, t):
                return self.interaction_config.policy.initial_control

            return initial_control
        else:
            raise NotImplementedError('This type of initial control has not been implemented yet')

    @abstractmethod
    def apply(self, x: jnp.ndarray, t: jnp.ndarray, tracking_data, dynamics_model, traj_idx, events) -> PolicyOut:
        pass

    @abstractmethod
    def update(self, dynamics_model: DynamicsModel,
               key: random.PRNGKey) -> Tuple[MPCCarry, OfflinePlanningData, TrackingData]:
        pass
