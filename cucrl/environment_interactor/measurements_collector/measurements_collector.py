from typing import Any, Tuple, NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random
from jax.lax import cond

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.environment_interactor.measurements_collector.abstract_measurement_collector import \
    AbstractMeasurementsCollector
from cucrl.environment_interactor.mpc_tracker import MPCTracker
from cucrl.main.config import InteractionConfig, Scaling
from cucrl.offline_planner.abstract_offline_planner import AbstractOfflinePlanner
from cucrl.time_sampler.time_sampler import TimeSampler
from cucrl.utils.classes import HallucinationSetup, MeasurementSelection
from cucrl.utils.classes import IntegrationCarry, MPCParameters, DynamicsModel, CollectorCarry
from cucrl.utils.representatives import DynamicsTracking

pytree = Any


class HallucinationCarry(NamedTuple):
    x: jax.Array
    t: jax.Array
    dt: jax.Array
    tracking_data: Any
    key: Any
    dynamics_model: DynamicsModel
    mpc_parameters: MPCParameters


class HallucinatedTrajectory(NamedTuple):
    xs: jax.Array
    us: jax.Array
    ts: jax.Array


# HallucinationOut is the hallucination and represents: [t_next, events]
HallucinationOut = Tuple[jax.Array | None, IntegrationCarry]


class MeasurementsCollector(AbstractMeasurementsCollector):
    def __init__(self, state_dim, control_dim, dynamics: AbstractDynamics, interaction_config: InteractionConfig,
                 offline_planner: AbstractOfflinePlanner, scaling: Scaling, num_traj: int):

        self.mpc_tracker = MPCTracker(state_dim, control_dim, interaction_config.angles_dim, scaling, dynamics,
                                      offline_planner.simulator_costs, interaction_config.policy.online_tracking)
        self.interaction_config = interaction_config

        self.num_traj = num_traj
        self.dynamics = dynamics
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.offline_planner = offline_planner
        self.time_sampler = TimeSampler(interaction_config.system_assumptions,
                                        interaction_config.measurement_collector.time_horizon)

    def apply(self, x: jnp.ndarray, t: jnp.ndarray, tracking_data, dynamics_model, traj_idx, events):
        measurement_selection_next, new_events = self.next_measurement_times(x, t, tracking_data, dynamics_model,
                                                                             traj_idx, events)
        return measurement_selection_next, new_events

    def other_episode_hallucination(self, x, t, events: IntegrationCarry, dynamics_model: DynamicsModel, tracking_data):
        # Events are also filtered to the trajectory that we are hallucinating (filtering is done with the integrator)
        # Hallucinate trajectory
        jax.debug.print("Hallucinate other episode: {x}", x=t)
        hallucination_setup = events.collector_carry.hallucination_setup
        dt = hallucination_setup.time_horizon / self.interaction_config.measurement_collector.num_hallucination_nodes
        new_key, subkey = random.split(events.collector_carry.key)
        hal_carry = HallucinationCarry(x=x, t=t, dt=dt, tracking_data=tracking_data,
                                       dynamics_model=dynamics_model, key=subkey,
                                       mpc_parameters=events.mpc_carry.mpc_params)
        _, traj = jax.lax.scan(self.hallucinate_step, hal_carry, xs=None,
                               length=self.interaction_config.measurement_collector.num_hallucination_nodes)

        # Find batch of times on the hallucinated trajectory where we should measure
        measurement_selection = self.dynamics.propose_measurement_times(
            dynamics_model, traj.xs, traj.us, traj.ts, self.interaction_config.measurement_collector.noise_std,
            jnp.arange(self.interaction_config.measurement_collector.batch_size_per_time_horizon))

        # new_events are the same as the old ones, except for events.mpc_carry.next_update_time
        next_measurement_planning_time = t + hallucination_setup.time_horizon
        new_collector_carry = CollectorCarry(hallucination_setup=hallucination_setup,
                                             next_measurement_time=next_measurement_planning_time.reshape(),
                                             key=new_key)

        new_events = IntegrationCarry(mpc_carry=events.mpc_carry, collector_carry=new_collector_carry)
        return measurement_selection, new_events

    def hallucinate_step(self, hal_carry: HallucinationCarry, _):
        # Update true policy
        new_true_policy = self.mpc_tracker.update_mpc(hal_carry.x, hal_carry.t, hal_carry.tracking_data,
                                                      hal_carry.mpc_parameters, hal_carry.dynamics_model)
        key, new_key = random.split(hal_carry.key)
        u = new_true_policy(hal_carry.t.reshape())
        # Execute true policy on mean dynamics
        if self.interaction_config.policy.online_tracking.dynamics_tracking == DynamicsTracking.MEAN:
            x_next = hal_carry.x + hal_carry.dt * self.dynamics.mean_eval_one(hal_carry.dynamics_model, hal_carry.x, u)
        else:
            mean, std = self.dynamics.mean_and_std_eval_one(hal_carry.dynamics_model, hal_carry.x, u)
            x_dot = mean + std * hal_carry.mpc_parameters.dynamics_id.eta
            x_next = hal_carry.x + hal_carry.dt * x_dot

        # Return next hallucination carry and trajectory for jax.lax.scan
        new_hal_carry = HallucinationCarry(x=x_next, t=hal_carry.t + hal_carry.dt, dt=hal_carry.dt,
                                           tracking_data=hal_carry.tracking_data,
                                           mpc_parameters=hal_carry.mpc_parameters,
                                           dynamics_model=hal_carry.dynamics_model, key=new_key)

        return new_hal_carry, HallucinatedTrajectory(xs=hal_carry.x, us=u, ts=hal_carry.t)

    def episode_zero_hallucination(self, x, t, events: IntegrationCarry, dynamics_model: DynamicsModel, tracking_data):
        # new_events are the same as the old ones, except for events.mpc_carry.next_update_time
        jax.debug.print("Hallucinate episode 0: {x}", x=t)
        hallucination_setup = events.collector_carry.hallucination_setup
        next_measurement_time = t + hallucination_setup.time_horizon
        new_collector_carry = CollectorCarry(hallucination_setup=hallucination_setup,
                                             next_measurement_time=next_measurement_time.reshape(),
                                             key=events.collector_carry.key)
        new_events = IntegrationCarry(mpc_carry=events.mpc_carry, collector_carry=new_collector_carry)
        config = self.interaction_config.measurement_collector
        proposed_times = jnp.linspace(t.reshape(), t.reshape() + hallucination_setup.time_horizon,
                                      config.batch_size_per_time_horizon + 2)[1:-1]
        measurement_selection = MeasurementSelection(
            potential_xs=jnp.zeros(shape=(config.num_interpolated_values, self.state_dim)),
            potential_us=jnp.zeros(shape=(config.num_interpolated_values, self.control_dim)),
            potential_ts=jnp.zeros(shape=(config.num_interpolated_values, 1)),
            proposed_ts=proposed_times.reshape(-1, 1),
            vars_before_collection=jnp.ones(shape=(config.num_interpolated_values, self.state_dim)),
            proposed_indices=jnp.arange(config.batch_size_per_time_horizon, dtype=jnp.int32),
        )
        return measurement_selection, new_events

    def apply_hallucination(self, x: jnp.ndarray, t: jnp.ndarray, events, tracking_data,
                            dynamics_model) -> HallucinationOut:
        # We can hallucinate on the mean dynamics, the optimistic dynamics with epsilon, dynamics with particle or
        # with Gaussian Process, we implement it here for the mean dynamics first, we return the next time for sampling
        jax.debug.print("Hallucinate: {x}", x=t)
        jax.debug.print("Episode for hallucination: {x}", x=dynamics_model.episode)
        return cond(dynamics_model.episode == 0, self.episode_zero_hallucination, self.other_episode_hallucination, x,
                    t, events, dynamics_model, tracking_data)

    def not_apply_hallucination(self, x: jnp.ndarray, t: jnp.ndarray, events, tracking_data,
                                dynamics_model) -> HallucinationOut:
        return self.default_measurement_selection(), events

    def default_measurement_selection(self):
        config = self.interaction_config.measurement_collector
        proposed_times = -1 * jnp.ones(
            shape=(config.batch_size_per_time_horizon, 1),
            dtype=jnp.float64)
        measurement_selection = MeasurementSelection(
            potential_xs=jnp.zeros(shape=(config.num_interpolated_values, self.state_dim)),
            potential_us=jnp.zeros(shape=(config.num_interpolated_values, self.control_dim)),
            potential_ts=jnp.zeros(shape=(config.num_interpolated_values, 1)),
            proposed_ts=proposed_times.reshape(-1, 1),
            vars_before_collection=jnp.ones(shape=(config.num_interpolated_values, self.state_dim)),
            proposed_indices=jnp.arange(config.batch_size_per_time_horizon, dtype=jnp.int32)
        )
        return measurement_selection

    def next_measurement_times(self, x: jnp.ndarray, t: jnp.ndarray, tracking_data, dynamics_model, traj_idx, events):
        return cond(t.reshape() >= events.collector_carry.next_measurement_time, self.apply_hallucination,
                    self.not_apply_hallucination, x, t, events, jtu.tree_map(lambda z: z[traj_idx], tracking_data),
                    dynamics_model)

    def update(self, dynamics_model: DynamicsModel, key: random.PRNGKey) -> CollectorCarry:
        hallucination_setup = HallucinationSetup(
            time_horizon=self.time_sampler.time_horizon(dynamics_model.beta),
            num_steps=self.interaction_config.measurement_collector.num_hallucination_nodes)
        next_measurement_time = jnp.zeros(shape=(self.num_traj,))
        keys = random.split(key, self.num_traj)
        collector_carry = CollectorCarry(
            next_measurement_time=next_measurement_time, key=keys,
            hallucination_setup=jtu.tree_map(lambda x: jnp.repeat(x, repeats=self.num_traj), hallucination_setup))
        return collector_carry
