import chex
from typing import Callable
from jaxtyping import PyTree

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.environment_interactor.interactor import Interactor, Interaction
from cucrl.main.config import InteractionConfig, Scaling
from cucrl.main.data_stats import Normalizer
from cucrl.offline_planner.abstract_offline_planner import AbstractOfflinePlanner
from cucrl.utils.classes import DynamicsModel, IntegrationCarry
from cucrl.utils.helper_functions import AngleLayerDynamics




class SACInteractor(Interactor):
    def __init__(self, x_dim: int, u_dim: int, dynamics: AbstractDynamics, x0s: chex.Array, normalizer: Normalizer,
                 angle_layer: AngleLayerDynamics, interaction_config: InteractionConfig,
                 offline_planner: AbstractOfflinePlanner, scaling: Scaling):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.dynamics = dynamics
        self.x0s = x0s
        self.normalizer = normalizer
        self.angle_layer = angle_layer
        self.interaction_config = interaction_config
        self.scaling = scaling


    def _create_policy_callable(self) -> Callable[[chex.Array, PyTree], chex.Array]:
        # Todo: create callable policy
        policy: Callable[[chex.Array, PyTree], chex.Array] = None
        return policy

    def update(self, dynamics_model: DynamicsModel, key: chex.PRNGKey):
        # Todo: Create SAC model
        # Todo: Train SAC model
        # Todo update self.params_model
        self.params_model = None

    def interact(self, x: chex.Array, t: chex.Array, traj_idx: chex.Array, events: IntegrationCarry) -> Interaction:
        pass
