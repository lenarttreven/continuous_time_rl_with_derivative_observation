from typing import Any, Tuple, Callable

import chex
from jax.lax import cond
from jaxtyping import PyTree

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.environment_interactor.policy.abstract_policy import Policy
from cucrl.main.config import InteractionConfig, Scaling
from cucrl.utils.classes import IntegrationCarry
from cucrl.utils.helper_functions import AngleLayerDynamics

pytree = Any

# PolicyOut is the output of the policy and represents: [us, events]
PolicyOut = Tuple[chex.Array, IntegrationCarry]


class SACPolicy(Policy):
    def __init__(self, x_dim, u_dim, dynamics: AbstractDynamics, initial_conditions, normalizer,
                 angle_layer: AngleLayerDynamics, interaction_config: InteractionConfig, scaling: Scaling,
                 apply_policy: Callable):
        super().__init__(x_dim, u_dim, initial_conditions, normalizer, interaction_config, angle_layer, scaling)

        self.dynamics = dynamics
        self.apply_policy = apply_policy

    def episode_zero(self, x_k: chex.Array, t_k: chex.Array, params: PyTree) -> chex.Array:
        return self.initial_control(x_k, t_k)

    def other_episode(self, x_k: chex.Array, t_k: chex.Array, params: PyTree) -> chex.Array:
        assert x_k.shape == (self.dynamics.x_dim,) and t_k.shape == ()
        chex.assert_type(t_k, int)
        u = self.apply_policy(x_k, params)
        return u[:self.u_dim]

    def apply(self, x_k: chex.Array, t_k: chex.Array, params, dynamics_model) -> chex.Array:
        return cond(dynamics_model.episode == 0, self.episode_zero, self.other_episode, x_k, t_k, params)

    def update(self, dynamics_model, key):
        pass
