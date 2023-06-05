from typing import Any, Tuple, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
from jax.lax import cond

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.environment_interactor.measurements_collector.abstract_measurement_collector import (
    AbstractMeasurementsCollector,
)
from cucrl.environment_interactor.mpc_tracker import MPCTracker
from cucrl.main.config import InteractionConfig, Scaling
from cucrl.offline_planner.abstract_offline_planner import AbstractOfflinePlanner
from cucrl.time_sampler.time_sampler import TimeSampler
from cucrl.utils.classes import (
    IntegrationCarry,
    MPCParameters,
    DynamicsModel,
    CollectorCarry,
)
from cucrl.utils.classes import MeasurementSelection, TrackingData, TruePolicy

pytree = Any


@chex.dataclass
class HallucinationCarry:
    x_k: chex.Array
    t_k: chex.Array
    next_mpc_update_idx: chex.Array
    tracking_data: Any
    true_policy: Any
    key: chex.PRNGKey
    dynamics_model: DynamicsModel
    mpc_parameters: MPCParameters


class HallucinatedTrajectory(NamedTuple):
    xs: chex.Array
    us: chex.Array
    ts: chex.Array


# HallucinationOut is the hallucination and represents: [t_next, events]
HallucinationOut = Tuple[MeasurementSelection, IntegrationCarry]


class MeasurementsCollector(AbstractMeasurementsCollector):
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        dynamics: AbstractDynamics,
        interaction_config: InteractionConfig,
        offline_planner: AbstractOfflinePlanner,
        scaling: Scaling,
        num_traj: int,
    ):
        self.mpc_tracker = MPCTracker(
            x_dim=x_dim,
            u_dim=u_dim,
            scaling=scaling,
            dynamics=dynamics,
            simulator_costs=offline_planner.simulator_costs,
            interaction_config=interaction_config,
        )

        self.all_ts = jnp.linspace(
            *interaction_config.time_horizon, interaction_config.policy.num_nodes + 1
        )
        self.all_ts_idx = jnp.arange(interaction_config.policy.num_nodes + 1)

        self.interaction_config = interaction_config
        self.num_traj = num_traj
        self.dynamics = dynamics
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.offline_planner = offline_planner
        self.time_sampler = TimeSampler(interaction_config=interaction_config)

        # Prepare inner dt
        policy_config = interaction_config.policy
        total_time = (
            interaction_config.time_horizon[1] - interaction_config.time_horizon[0]
        )
        total_int_steps = (
            policy_config.num_nodes * policy_config.num_int_step_between_nodes
        )
        self.inner_dt = total_time / total_int_steps
        self.ts_inner_nodes = jnp.linspace(
            self.all_ts[0], self.all_ts[1], policy_config.num_int_step_between_nodes + 1
        )

    def apply(
        self,
        x_k: chex.Array,
        t_k: chex.Array,
        tracking_data,
        dynamics_model,
        traj_idx,
        events: IntegrationCarry,
    ) -> HallucinationOut:
        measurement_selection_next, new_events = self.next_measurement_times(
            x_k, t_k, tracking_data, dynamics_model, traj_idx, events
        )
        return measurement_selection_next, new_events

    def other_episode_hallucination(
        self,
        x_k: chex.Array,
        t_k: chex.Array,
        events: IntegrationCarry,
        dynamics_model: DynamicsModel,
        tracking_data: TrackingData,
    ) -> HallucinationOut:
        assert x_k.shape == (self.x_dim,) and t_k.shape == ()
        chex.assert_type(t_k, int)
        # Events are also filtered to the trajectory that we are hallucinating (filtering is done with the integrator)
        # Hallucinate trajectory
        jax.debug.print("Hallucinate other episode time step: {t}", t=t_k)
        hallucination_steps_arr = events.collector_carry.hallucination_steps_arr

        new_key, subkey = random.split(events.collector_carry.key)
        hal_carry = HallucinationCarry(
            x_k=x_k,
            t_k=t_k,
            tracking_data=tracking_data,
            dynamics_model=dynamics_model,
            key=subkey,
            mpc_parameters=events.mpc_carry.mpc_params,
            next_mpc_update_idx=t_k,
            true_policy=self.mpc_tracker.true_policy_place_holder(),
        )

        _, traj = jax.lax.scan(
            self.hallucinate_step,
            hal_carry,
            xs=None,
            length=hallucination_steps_arr.shape[0],
        )

        # Find batch of times on the hallucinated trajectory where we should measure
        measurement_selection = self.dynamics.propose_measurement_times(
            dynamics_model,
            traj.xs,
            traj.us,
            traj.ts.reshape(-1, 1),
            self.interaction_config.measurement_collector.noise_std,
            jnp.arange(
                self.interaction_config.measurement_collector.batch_size_per_time_horizon
            ),
        )

        # new_events are the same as the old ones, except for events.mpc_carry.next_update_time
        next_measurement_t_k = t_k + hallucination_steps_arr.shape[0]
        new_collector_carry = CollectorCarry(
            hallucination_steps_arr=hallucination_steps_arr,
            next_measurement_time=next_measurement_t_k,
            key=new_key,
        )

        new_events = IntegrationCarry(
            mpc_carry=events.mpc_carry, collector_carry=new_collector_carry
        )
        return measurement_selection, new_events

    def hallucinate_step(self, hal_carry: HallucinationCarry, _):
        # Update true policy
        def update_true_policy(
            _hal_carry: HallucinationCarry,
        ) -> Tuple[TruePolicy, HallucinationCarry]:
            new_true_policy = self.mpc_tracker.update_mpc(
                _hal_carry.x_k,
                _hal_carry.t_k,
                _hal_carry.tracking_data,
                _hal_carry.mpc_parameters,
                _hal_carry.dynamics_model,
            )

            period = self.interaction_config.policy.online_tracking.mpc_update_period
            _hal_carry = hal_carry.replace(
                true_policy=new_true_policy, next_mpc_update_idx=_hal_carry.t_k + period
            )
            return new_true_policy, _hal_carry

        def dont_update_true_policy(_hal_carry: HallucinationCarry):
            return hal_carry.true_policy, hal_carry

        true_policy, hal_carry = cond(
            hal_carry.t_k == hal_carry.next_mpc_update_idx,
            update_true_policy,
            dont_update_true_policy,
            hal_carry,
        )

        key, new_key = random.split(hal_carry.key)
        u = true_policy(hal_carry.t_k)

        def _next_step(x: chex.Array, t: chex.Array) -> Tuple[chex.Array, chex.Array]:
            x_dot = self.dynamics.mean_eval_one(hal_carry.dynamics_model, x, u)
            x_next = x + self.inner_dt * x_dot
            return x_next, x_next

        jax.debug.print("Hallucinate other episode time step: {t}", t=hal_carry.t_k)

        # Prepare times for scan
        ts_scan = self.all_ts[hal_carry.t_k] + self.ts_inner_nodes
        x_next, _ = jax.lax.scan(_next_step, hal_carry.x_k, ts_scan)

        # Return next hallucination carry and trajectory for jax.lax.scan
        new_hal_carry = hal_carry.replace(
            x_k=x_next, t_k=hal_carry.t_k + 1, key=new_key
        )
        return new_hal_carry, HallucinatedTrajectory(
            xs=hal_carry.x_k, us=u, ts=self.all_ts[hal_carry.t_k]
        )

    def episode_zero_hallucination(
        self,
        x_k: chex.Array,
        t_k: chex.Array,
        events: IntegrationCarry,
        dynamics_model: DynamicsModel,
        tracking_data: TrackingData,
    ) -> HallucinationOut:
        # new_events are the same as the old ones, except for events.mpc_carry.next_update_time
        jax.debug.print("Hallucinate episode 0 time step: {t}", t=t_k)
        hallucination_steps_arr = events.collector_carry.hallucination_steps_arr
        next_measurement_time = t_k + hallucination_steps_arr.shape[0]
        new_collector_carry = CollectorCarry(
            hallucination_steps_arr=hallucination_steps_arr,
            next_measurement_time=next_measurement_time,
            key=events.collector_carry.key,
        )
        new_events = IntegrationCarry(
            mpc_carry=events.mpc_carry, collector_carry=new_collector_carry
        )
        config = self.interaction_config.measurement_collector

        proposed_times = jnp.linspace(
            self.all_ts[t_k],
            self.all_ts[t_k + hallucination_steps_arr.shape[0]],
            config.batch_size_per_time_horizon + 2,
        )[1:-1]

        measurement_selection = MeasurementSelection(
            potential_xs=jnp.zeros(shape=(config.num_interpolated_values, self.x_dim)),
            potential_us=jnp.zeros(shape=(config.num_interpolated_values, self.u_dim)),
            potential_ts=jnp.zeros(shape=(config.num_interpolated_values, 1)),
            proposed_ts=proposed_times.reshape(-1, 1),
            vars_before_collection=jnp.ones(
                shape=(config.num_interpolated_values, self.x_dim)
            ),
            proposed_indices=jnp.arange(
                config.batch_size_per_time_horizon, dtype=jnp.int32
            ),
        )
        return measurement_selection, new_events

    def apply_hallucination(
        self,
        x_k: chex.Array,
        t_k: chex.Array,
        events: IntegrationCarry,
        tracking_data: TrackingData,
        dynamics_model: DynamicsModel,
    ) -> HallucinationOut:
        # return self.default_measurement_selection(), events
        # We can hallucinate on the mean dynamics, the optimistic dynamics with epsilon, dynamics with particle or
        # with Gaussian Process, we implement it here for the mean dynamics first, we return the next time for sampling
        jax.debug.print("Hallucinate time step: {t}", t=t_k)
        jax.debug.print(
            "Episode for hallucination: {episode}", episode=dynamics_model.episode
        )
        return cond(
            dynamics_model.episode == 0,
            self.episode_zero_hallucination,
            self.other_episode_hallucination,
            x_k,
            t_k,
            events,
            dynamics_model,
            tracking_data,
        )

    def not_apply_hallucination(
        self,
        x_k: chex.Array,
        t_k: chex.Array,
        events: IntegrationCarry,
        tracking_data: TrackingData,
        dynamics_model: DynamicsModel,
    ) -> HallucinationOut:
        return self.default_measurement_selection(), events

    def default_measurement_selection(self) -> MeasurementSelection:
        config = self.interaction_config.measurement_collector
        proposed_times = -1 * jnp.ones(
            shape=(config.batch_size_per_time_horizon, 1), dtype=jnp.float64
        )
        measurement_selection = MeasurementSelection(
            potential_xs=jnp.zeros(shape=(config.num_interpolated_values, self.x_dim)),
            potential_us=jnp.zeros(shape=(config.num_interpolated_values, self.u_dim)),
            potential_ts=jnp.zeros(shape=(config.num_interpolated_values, 1)),
            proposed_ts=proposed_times.reshape(-1, 1),
            vars_before_collection=jnp.ones(
                shape=(config.num_interpolated_values, self.x_dim)
            ),
            proposed_indices=jnp.arange(
                config.batch_size_per_time_horizon, dtype=jnp.int32
            ),
        )
        return measurement_selection

    def next_measurement_times(
        self,
        x_k: chex.Array,
        t_k: chex.Array,
        tracking_data,
        dynamics_model,
        traj_idx,
        events: IntegrationCarry,
    ) -> HallucinationOut:
        return cond(
            t_k == events.collector_carry.next_measurement_time,
            self.apply_hallucination,
            self.not_apply_hallucination,
            x_k,
            t_k,
            events,
            jtu.tree_map(lambda z: z[traj_idx], tracking_data),
            dynamics_model,
        )

    def update(
        self, dynamics_model: DynamicsModel, key: random.PRNGKey
    ) -> CollectorCarry:
        hallucination_steps = self.time_sampler.time_steps(dynamics_model.beta)
        hallucination_steps_arr = jnp.zeros(hallucination_steps, dtype=jnp.int64)

        next_measurement_time = jnp.zeros(shape=(self.num_traj,), dtype=jnp.int64)
        keys = random.split(key, self.num_traj)
        collector_carry = CollectorCarry(
            next_measurement_time=next_measurement_time,
            key=keys,
            hallucination_steps_arr=jnp.repeat(
                hallucination_steps_arr[None, ...], repeats=self.num_traj, axis=0
            ),
        )
        return collector_carry
