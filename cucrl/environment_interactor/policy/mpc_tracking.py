from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax import random
from jax.lax import cond
from jax.tree_util import tree_map

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.environment_interactor.mpc_tracker import MPCTracker
from cucrl.environment_interactor.policy.abstract_policy import Policy
from cucrl.main.config import InteractionConfig, Scaling
from cucrl.offline_planner.abstract_offline_planner import AbstractOfflinePlanner
from cucrl.offline_planner.planner import get_planner
from cucrl.utils.classes import IntegrationCarry, TrackingData, MPCCarry, MPCParameters, TruePolicy, DynamicsModel
from cucrl.utils.helper_functions import AngleLayerDynamics

pytree = Any

# PolicyOut is the output of the policy and represents: [us, events]
PolicyOut = Tuple[jax.Array, IntegrationCarry]


class MPCTracking(Policy):
    def __init__(self, state_dim, control_dim, dynamics: AbstractDynamics, initial_conditions, normalizer,
                 angle_layer: AngleLayerDynamics, interaction_config: InteractionConfig,
                 offline_planner: AbstractOfflinePlanner, scaling: Scaling):
        super(MPCTracking, self).__init__(state_dim, control_dim, initial_conditions, normalizer, offline_planner,
                                          interaction_config, angle_layer, scaling)
        self.mpc_tracker = MPCTracker(state_dim, control_dim, interaction_config.angles_dim, scaling,
                                      dynamics, offline_planner.simulator_costs,
                                      interaction_config.policy.online_tracking)

        self.dynamics = dynamics
        self.planner = get_planner(offline_planner, control_config=interaction_config)
        self.dt = self.interaction_config.policy.online_tracking.mpc_dt

    def update_mpc_for_cond(self, x, t, events: IntegrationCarry, tracking_data, dynamics_model: DynamicsModel):
        new_next_update_time = t.reshape() + self.dt
        new_key, subkey = random.split(events.mpc_carry.key)
        new_true_policy = self.mpc_tracker.update_mpc(x, t, tracking_data, events.mpc_carry.mpc_params, dynamics_model)
        new_mpc_carry = MPCCarry(next_update_time=new_next_update_time, key=subkey,
                                 mpc_params=events.mpc_carry.mpc_params, true_policy=new_true_policy)
        new_events = IntegrationCarry(mpc_carry=new_mpc_carry, collector_carry=events.collector_carry)
        return new_events

    @staticmethod
    def no_update_mpc_for_cond(x, t, events: IntegrationCarry, tracking_data, dynamics_model: DynamicsModel):
        return events

    def episode_zero(self, x, t, events, tracking_data, dynamics_model: DynamicsModel):
        return self.initial_control(x, t), events

    def other_episode(self, x, t, events: IntegrationCarry, tracking_data, dynamics_model: DynamicsModel):
        new_events = cond(t.reshape() >= events.mpc_carry.next_update_time, self.update_mpc_for_cond,
                          self.no_update_mpc_for_cond, x, t, events, tracking_data, dynamics_model)
        return new_events.mpc_carry.true_policy(t.reshape()), new_events

    def apply(self, x: jnp.ndarray, t: jnp.ndarray, tracking_data, dynamics_model, traj_idx, events) -> PolicyOut:
        u, new_events = cond(dynamics_model.episode == 0, self.episode_zero, self.other_episode, x, t, events,
                             tree_map(lambda z: z[traj_idx], tracking_data), dynamics_model)
        return u, new_events

    def update(self, dynamics_model: DynamicsModel, key: random.PRNGKey):
        key, subkey = random.split(key)
        if dynamics_model.episode == 0:
            offline_planning_data = self.planner.initialize(
                dynamics_model=dynamics_model, initial_conditions=self.initial_conditions, key=subkey)
        else:
            offline_planning_data = self.planner.plan(
                dynamics_model=dynamics_model, initial_conditions=self.initial_conditions, key=subkey)

        tracking_data = TrackingData(ts=offline_planning_data.ts, xs=offline_planning_data.xs,
                                     us=offline_planning_data.us, final_t=offline_planning_data.final_t,
                                     target_x=offline_planning_data.target_x, target_u=offline_planning_data.target_u)

        next_update_time = jnp.zeros(shape=(self.num_traj,))
        mpc_parameters = MPCParameters(dynamics_id=offline_planning_data.dynamics_ids)

        ts_init = jnp.linspace(0, 1, self.mpc_tracker.mpc_tracker.num_nodes).reshape(-1, 1)
        true_policy = TruePolicy(ts=jnp.repeat(ts_init[jnp.newaxis, ...], repeats=self.num_traj, axis=0),
                                 us=jnp.zeros(
                                     shape=(self.num_traj, self.mpc_tracker.mpc_tracker.num_nodes, self.control_dim)))

        keys = random.split(key, self.num_traj)
        mpc_carry = MPCCarry(next_update_time=next_update_time, key=keys, mpc_params=mpc_parameters,
                             true_policy=true_policy)

        return mpc_carry, offline_planning_data, tracking_data
