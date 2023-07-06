from typing import Any, Tuple

import chex
import jax.numpy as jnp
from jax import random
from jax.lax import cond
from jax.tree_util import tree_map

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.environment_interactor.discretization.runge_kutta import RungeKutta
from cucrl.environment_interactor.mpc_tracker import MPCTracker
from cucrl.environment_interactor.policy.abstract_policy import Policy
from cucrl.main.config import InteractionConfig, Scaling
from cucrl.offline_planner.abstract_offline_planner import AbstractOfflinePlanner
from cucrl.offline_planner.planner import get_planner
from cucrl.utils.classes import IntegrationCarry, MPCCarry, MPCParameters, TruePolicy, DynamicsModel
from cucrl.utils.classes import TrackingData
from cucrl.utils.helper_functions import AngleLayerDynamics

pytree = Any

# PolicyOut is the output of the policy and represents: [us, events]
PolicyOut = Tuple[chex.Array, IntegrationCarry]


class MPCTracking(Policy):
    def __init__(self, x_dim, u_dim, dynamics: AbstractDynamics, initial_conditions, normalizer,
                 angle_layer: AngleLayerDynamics, interaction_config: InteractionConfig,
                 offline_planner: AbstractOfflinePlanner, scaling: Scaling):
        super().__init__(x_dim, u_dim, initial_conditions, normalizer, offline_planner,
                         interaction_config, angle_layer, scaling)

        control_discretization = RungeKutta(time_horizon=interaction_config.time_horizon,
                                            num_control_steps=interaction_config.policy.num_control_steps,
                                            buffer_control_steps=interaction_config.policy.online_tracking.control_steps + 1)
        self.control_discretization = control_discretization
        self.mpc_tracker = MPCTracker(x_dim=x_dim, u_dim=u_dim, scaling=scaling, dynamics=dynamics,
                                      simulator_costs=offline_planner.simulator_costs,
                                      interaction_config=interaction_config,
                                      control_discretization=control_discretization)

        self.dynamics = dynamics
        self.planner = get_planner(offline_planner, control_config=interaction_config)
        self.period = self.interaction_config.policy.online_tracking.mpc_update_period

    def update_mpc_for_cond(self, x_k: chex.Array, t_k: chex.Array, events: IntegrationCarry,
                            tracking_data: TrackingData, dynamics_model: DynamicsModel) -> IntegrationCarry:
        chex.assert_type(t_k, int)
        next_update_idx = t_k + self.period
        new_key, subkey = random.split(events.mpc_carry.key)
        new_true_policy = self.mpc_tracker.update_mpc(x_k, t_k, tracking_data, events.mpc_carry.mpc_params,
                                                      dynamics_model)
        # jax.debug.breakpoint()
        # jax.debug.print('new_true_policy: {x}', x=jax.lax.dynamic_slice_in_dim(tracking_data.us, t_k,
        #                                                                        self.control_discretization.buffer_steps - 1,
        #                                                                        axis=0))
        # jax.debug.print('tracking: {x}', x=jax.lax.dynamic_slice_in_dim(new_true_policy.us, t_k,
        #                                                                 self.control_discretization.buffer_steps - 1,
        #                                                                 axis=0))
        new_mpc_params = MPCParameters(true_policy=new_true_policy, dynamics_id=events.mpc_carry.mpc_params.dynamics_id)
        new_mpc_carry = MPCCarry(next_update_time=next_update_idx, key=subkey,
                                 mpc_params=new_mpc_params)

        new_events = IntegrationCarry(mpc_carry=new_mpc_carry, collector_carry=events.collector_carry)
        return new_events

    @staticmethod
    def no_update_mpc_for_cond(x_k: chex.Array, t_k: chex.Array, events: IntegrationCarry, tracking_data: TrackingData,
                               dynamics_model: DynamicsModel) -> IntegrationCarry:
        return events

    def episode_zero(self, x_k: chex.Array, t_k: chex.Array, events, tracking_data: TrackingData,
                     dynamics_model: DynamicsModel) -> PolicyOut:
        return self.initial_control(x_k, t_k), events

    def other_episode(self, x_k: chex.Array, t_k: chex.Array, events: IntegrationCarry, tracking_data: TrackingData,
                      dynamics_model: DynamicsModel) -> PolicyOut:
        assert x_k.shape == (self.dynamics.x_dim,) and t_k.shape == ()
        chex.assert_type(t_k, int)
        new_events = cond(t_k == events.mpc_carry.next_update_time, self.update_mpc_for_cond,
                          self.no_update_mpc_for_cond, x_k, t_k, events, tracking_data, dynamics_model)
        return new_events.mpc_carry.mpc_params.true_policy(t_k), new_events

    def apply(self, x_k: chex.Array, t_k: chex.Array, tracking_data: TrackingData, dynamics_model, traj_idx,
              events) -> PolicyOut:
        u, new_events = cond(dynamics_model.episode == 0, self.episode_zero, self.other_episode, x_k, t_k, events,
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

        mpc_control_steps = jnp.repeat(
            jnp.array(self.interaction_config.policy.online_tracking.control_steps)[None, ...], repeats=self.num_traj,
            axis=0)

        tracking_data = TrackingData(ts=offline_planning_data.ts, xs=offline_planning_data.xs,
                                     us=offline_planning_data.us,
                                     mpc_control_steps=mpc_control_steps,
                                     target_x=offline_planning_data.target_x, target_u=offline_planning_data.target_u)

        next_update_time = jnp.zeros(shape=(self.num_traj,), dtype=jnp.int64)
        mpc_parameters = MPCParameters(dynamics_id=offline_planning_data.dynamics_ids,
                                       true_policy=TruePolicy(us=jnp.zeros(shape=(
                                           self.num_traj, self.control_discretization.continuous_times.size,
                                           self.u_dim))))

        keys = random.split(key, self.num_traj)
        mpc_carry = MPCCarry(next_update_time=next_update_time, key=keys, mpc_params=mpc_parameters)

        return mpc_carry, offline_planning_data, tracking_data
