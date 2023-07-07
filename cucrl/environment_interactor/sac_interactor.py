from typing import Callable, Tuple

import chex
import jax.random as jr
from jaxtyping import PyTree

from cucrl.brax_sac_implementation.sac import SAC
from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.environment_interactor.brax_learned_dynamics_model import LearnedModel
from cucrl.environment_interactor.interactor import Interactor, Interaction
from cucrl.main.config import InteractionConfig, Scaling
from cucrl.main.data_stats import Normalizer
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.utils.classes import DynamicsModel, IntegrationCarry
from cucrl.utils.helper_functions import AngleLayerDynamics, AngleNormalizer


class SACInteractor(Interactor):
    def __init__(self, x_dim: int, u_dim: int, dynamics: AbstractDynamics, x0s: chex.Array, normalizer: Normalizer,
                 angle_layer: AngleLayerDynamics, interaction_config: InteractionConfig,
                 simulator_costs: SimulatorCostsAndConstraints, scaling: Scaling):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.dynamics = dynamics
        self.x0s = x0s
        self.normalizer = normalizer
        self.angle_layer = angle_layer
        self.interaction_config = interaction_config
        self.scaling = scaling
        self.angle_normalizer = AngleNormalizer(state_dim=x_dim, control_dim=u_dim,
                                                angles_dim=interaction_config.angles_dim,
                                                state_scaling=scaling.state_scaling)
        self.simulator_costs = simulator_costs
        self.sac_config = dict(num_timesteps=30_000, num_evals=20, reward_scaling=10, episode_length=100,
                               normalize_observations=True, action_repeat=1, discounting=0.999, lr_policy=3e-4,
                               lr_alpha=3e-4, lr_q=3e-4, num_envs=16, batch_size=64, grad_updates_per_step=32,
                               max_replay_size=2 ** 14, min_replay_size=2 ** 8, num_eval_envs=1,
                               deterministic_eval=True, tau=0.005, wd_policy=1e-2, wd_q=1e-2, wd_alpha=1e-2,
                               wandb_logging=True)
        self.brax_env_config = dict(dt=0.1, dynamics_model=self.dynamics, sim_cost=self.simulator_costs,
                                    angle_normalizer=self.angle_normalizer)
        self.apply_policy, self.params_model = self._create_policy_callable()

    def _create_policy_callable(self) -> Tuple[Callable[[chex.Array, PyTree], chex.Array], PyTree]:
        _env = LearnedModel(dynamics_params=DynamicsModel(), **self.brax_env_config)
        _sac_trainer = SAC(environment=_env, **self.sac_config)
        make_policy = _sac_trainer.make_policy

        def policy(x: chex.Array, params: PyTree) -> chex.Array:
            return make_policy(params, deterministic=True)(x, jr.PRNGKey(0))[0]

        _training_state = _sac_trainer.init_training_state(jr.PRNGKey(0))
        _params = (_training_state.normalizer_params, _training_state.policy_params)
        return policy, _params

    def update(self, dynamics_model: DynamicsModel, key: chex.PRNGKey):
        brax_env = LearnedModel(dynamics_params=dynamics_model, **self.brax_env_config)
        sac_trainer = SAC(environment=brax_env, **self.sac_config)
        _, params, _ = sac_trainer.run_training(key=jr.PRNGKey(0))
        self.params_model = params

    def interact(self, x: chex.Array, t: chex.Array, traj_idx: chex.Array, events: IntegrationCarry) -> Interaction:
        pass
