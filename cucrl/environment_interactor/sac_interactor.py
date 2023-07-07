from typing import Callable, Tuple

import chex
import jax.random as jr
from jaxtyping import PyTree

from cucrl.brax_sac_implementation.sac import SAC
from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.environment_interactor.brax_learned_dynamics_model import LearnedModel
from cucrl.environment_interactor.interactor import Interactor, Interaction
from cucrl.environment_interactor.sac_mss import SACMSS
from cucrl.environment_interactor.sac_policy import SACPolicy
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

        # Need to compute dt and number of steps
        total_time = interaction_config.time_horizon.length()
        policy_config = interaction_config.policy
        total_int_steps = policy_config.num_control_steps * policy_config.num_int_step_between_nodes
        self.inner_dt = total_time / total_int_steps
        self.sac_config = dict(num_timesteps=30_000, num_evals=20, reward_scaling=10,
                               episode_length=policy_config.num_control_steps,
                               normalize_observations=True, action_repeat=1, discounting=0.999, lr_policy=3e-4,
                               lr_alpha=3e-4, lr_q=3e-4, num_envs=16, batch_size=64, grad_updates_per_step=32,
                               max_replay_size=2 ** 14, min_replay_size=2 ** 8, num_eval_envs=1,
                               deterministic_eval=True, tau=0.005, wd_policy=1e-2, wd_q=1e-2, wd_alpha=1e-2,
                               wandb_logging=True)

        self.brax_env_config = dict(integration_dt=self.inner_dt,
                                    num_int_steps=policy_config.num_int_step_between_nodes,
                                    dynamics_model=self.dynamics, sim_cost=self.simulator_costs,
                                    angle_normalizer=self.angle_normalizer)
        self.apply_policy, self.params_model = self._create_policy_callable()
        self.sac_policy = SACPolicy(x_dim=self.x_dim, u_dim=self.u_dim, dynamics=self.dynamics,
                                    initial_conditions=self.x0s, normalizer=self.normalizer,
                                    angle_layer=self.angle_layer, interaction_config=self.interaction_config,
                                    scaling=self.scaling, apply_policy=self.apply_policy)
        self.sac_mss = SACMSS(x_dim=self.x_dim, u_dim=self.u_dim, dynamics=self.dynamics,
                              interaction_config=self.interaction_config, num_traj=len(x0s),
                              apply_policy=self.apply_policy)

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
        self.dynamics_model = dynamics_model
        self.params_model = params

        subkey, key = jr.split(key)
        collector_carry = self.sac_mss.update(dynamics_model, subkey)
        self.integration_carry = IntegrationCarry(mpc_carry=None, collector_carry=collector_carry)

    def interact(self, x: chex.Array, t: chex.Array, traj_idx: chex.Array, events: IntegrationCarry) -> Interaction:
        u = self.sac_policy.apply(x, t, params=self.params_model, dynamics_model=self.dynamics_model)
        # Add measurement collector
        measurement_selection, new_events = self.sac_mss.apply(x, t, self.params_model, self.dynamics_model, events)
        return u, measurement_selection, new_events

    def tree_flatten(self):
        children = (self.dynamics_model, self.params_model, self.integration_carry)
        aux_data = {'x_dim': self.x_dim, 'u_dim': self.u_dim, 'dynamics': self.dynamics,
                    'x0s': self.x0s, 'normalizer': self.normalizer,
                    'angle_layer': self.angle_layer, 'interaction_config': self.interaction_config,
                    'scaling': self.scaling, 'simulator_costs': self.simulator_costs}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        dynamics_model, params_model, integration_carry = children
        new_class = cls(x_dim=aux_data['v'], u_dim=aux_data['u_dim'],
                        dynamics=aux_data['dynamics'], x0s=aux_data['x0s'],
                        normalizer=aux_data['normalizer'], angle_layer=aux_data['angle_layer'],
                        interaction_config=aux_data['interaction_config'],
                        scaling=aux_data['scaling'], simulator_costs=aux_data['simulator_costs'])
        new_class.dynamics_model = dynamics_model
        new_class.params_model = params_model
        new_class.integration_carry = integration_carry
        return new_class
