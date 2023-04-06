import copy
from typing import List, Tuple, Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from cucrl.environment_interactor.interactor import Interactor
from cucrl.main.config import DataGenerationConfig
from cucrl.simulator.ode_integrator import ForwardEuler
from cucrl.utils.classes import MeasurementSelection
from cucrl.utils.splines import MultivariateSpline

Schedule = Callable[[int], float]
pytree = Any


class Keys(NamedTuple):
    episode_key: jnp.ndarray
    step_key: jnp.ndarray


class ObservationData(NamedTuple):
    ts: List[jax.Array]
    ys: List[jax.Array]
    x0s: List[jax.Array]
    xs: List[jax.Array]
    xs_dot: List[jax.Array]
    us: List[jax.Array]


class VisualisationData(NamedTuple):
    ts: jax.Array
    x0s: jax.Array
    xs: jax.Array
    xs_dot: jax.Array
    us: jax.Array
    ys: jax.Array


class MatchingData(NamedTuple):
    ts: jax.Array
    x0s: jax.Array
    us: jax.Array


class DataRepr(NamedTuple):
    observation_data: ObservationData
    visualization_data: VisualisationData
    matching_data: MatchingData


class DataGenerator:
    def __init__(self, data_generation: DataGenerationConfig, interactor: Interactor):
        self.initial_conditions = data_generation.initial_conditions
        self.num_matching_points = data_generation.num_matching_points
        self.num_visualization_points = data_generation.num_visualization_points
        self.state_dim = self.initial_conditions[0].size
        self.control_dim = data_generation.control_dim
        self.stacked_initial_conditions = jnp.stack(self.initial_conditions)

        self.num_trajectories = len(self.initial_conditions)
        self.time_horizon = data_generation.time_horizon
        self.visualization_times_whole_horizon = jnp.linspace(*self.time_horizon,
                                                              self.num_visualization_points).reshape(-1, 1)
        self.simulator_type = data_generation.simulator_type
        self.simulator_parameters = data_generation.simulator_params
        observation_noise = data_generation.noise
        self.simulation_noise = [None] if observation_noise is None else observation_noise
        self.simulator = ForwardEuler(interactor=interactor, simulator_type=self.simulator_type,
                                      scaling=data_generation.scaling, step_size=data_generation.simulator_step_size,
                                      termination_config=data_generation.termination_config)

    def prepare_matching_times(self, times: List[jnp.array]) -> List[jnp.array]:
        matching_times = []
        for traj_id in range(self.num_trajectories):
            min_time, max_time = jnp.min(times[traj_id]), jnp.max(times[traj_id])
            matching_times.append(jnp.linspace(min_time, max_time, self.num_matching_points))
        return matching_times

    def prepare_initial_conditions(self, observation_times: List[jnp.array], matching_times: List[jnp.array],
                                   visualization_times: List[jnp.array], ic: List[jnp.array]) -> Tuple[jnp.array, ...]:
        # We repeat initial conditions so that every time point has its own initial condition
        repeated_ic_obs, repeated_ic_match, repeated_ic_vis = [], [], []
        for traj_id in range(self.num_trajectories):
            repeated_ic_obs.append(jnp.repeat(ic[traj_id].reshape(1, -1), observation_times[traj_id].size, axis=0))
            repeated_ic_match.append(jnp.repeat(ic[traj_id].reshape(1, -1), matching_times[traj_id].size, axis=0))
            repeated_ic_vis.append(jnp.repeat(ic[traj_id].reshape(1, -1), visualization_times[traj_id].size, axis=0))
        return repeated_ic_obs, repeated_ic_match, repeated_ic_vis

    def prepare_matching_controls(self, controls: List[jnp.array], obs_times: List[jnp.array],
                                  matching_times: List[jnp.array]) -> List[jnp.array]:
        matching_controls = []
        for traj_id in range(self.num_trajectories):
            spline = MultivariateSpline(obs_times[traj_id], controls[traj_id])
            matching_controls.append(spline(matching_times[traj_id]))
        return matching_controls

    def generate_trajectories(self, rng: jnp.array) -> Tuple[DataRepr, MeasurementSelection]:
        key, subkey = jax.random.split(rng)
        # Simulate trajectory with the new policy
        trajectories, vis_trajectories, measurement_selection = self.simulator.simulate_trajectories(
            ics=self.initial_conditions, time_horizon=self.time_horizon, num_vis_ts=self.num_visualization_points,
            sigmas=self.simulation_noise, key=subkey, events=self.simulator.interactor.integration_carry)

        obs_times = [trajectory.ts for trajectory in trajectories]
        vis_times = [trajectory.ts for trajectory in vis_trajectories]

        matching_times = self.prepare_matching_times(obs_times)
        repeated_ic_obs, repeated_ic_match, repeated_ic_vis = self.prepare_initial_conditions(obs_times, matching_times,
                                                                                              vis_times,
                                                                                              self.initial_conditions)
        controls = [trajectory.us for trajectory in trajectories]
        control_matching = self.prepare_matching_controls(controls, obs_times, matching_times)
        observation_data = ObservationData(
            ys=[trajectory.ys for trajectory in trajectories],
            ts=[trajectory.ts.reshape(-1, 1) for trajectory in trajectories],
            xs=[trajectory.xs for trajectory in trajectories],
            xs_dot=[trajectory.d_xs for trajectory in trajectories],
            us=[trajectory.us for trajectory in trajectories],
            x0s=repeated_ic_obs
        )

        vis_trajectories = jtu.tree_map(lambda *x: jnp.stack(x), *vis_trajectories)
        visualization_data = VisualisationData(
            ts=jnp.expand_dims(vis_trajectories.ts, -1),
            ys=vis_trajectories.ys,
            xs=vis_trajectories.xs,
            us=vis_trajectories.us,
            xs_dot=vis_trajectories.d_xs,
            x0s=jnp.stack(repeated_ic_vis)
        )

        matching_data = MatchingData(
            ts=jnp.expand_dims(jnp.stack(matching_times), -1),
            us=jnp.stack(control_matching),
            x0s=jnp.stack(repeated_ic_match)
        )

        measurement_selection = jtu.tree_map(lambda *x: jnp.stack(x), *measurement_selection)

        return DataRepr(observation_data, visualization_data, matching_data), measurement_selection


class DynamicsDataManager:
    def __init__(self, state_dim: int, control_dim: int):
        self.state_dim = state_dim
        self.control_dim = control_dim
        # self.pool = self.initialize_pool()
        self.permanent_pool = self.initialize_pool()
        self.training_pool = self.initialize_pool()
        self.test_pool = self.initialize_pool()

    def initialize_pool(self):
        x = jnp.ones(shape=(0, self.state_dim))
        u = jnp.ones(shape=(0, self.control_dim))
        x_dot = jnp.ones(shape=(0, self.state_dim))
        std_x_dot = jnp.ones(shape=(0, self.state_dim))
        return {'xs': x, 'us': u, 'xs_dot': x_dot, 'xs_dot_std': std_x_dot}

    def add_data_to_training_pool(self, x: jnp.array, u: jnp.array, x_dot: jnp.array, std_x_dot: jnp.ndarray):
        assert x.shape == x_dot.shape == std_x_dot.shape and x.shape[0] == u.shape[0]
        assert x.shape[1] == self.state_dim and u.shape[1] == self.control_dim
        self.training_pool['xs'] = jnp.concatenate((self.training_pool['xs'], x))
        self.training_pool['us'] = jnp.concatenate((self.training_pool['us'], u))
        self.training_pool['xs_dot'] = jnp.concatenate((self.training_pool['xs_dot'], x_dot))
        self.training_pool['xs_dot_std'] = jnp.concatenate((self.training_pool['xs_dot_std'], std_x_dot))

    def add_data_to_permanent_pool(self, x: jnp.array, u: jnp.array, x_dot: jnp.array, std_x_dot: jnp.ndarray):
        assert x.shape == x_dot.shape == std_x_dot.shape and x.shape[0] == u.shape[0]
        assert x.shape[1] == self.state_dim and u.shape[1] == self.control_dim
        self.permanent_pool['xs'] = jnp.concatenate((self.permanent_pool['xs'], x))
        self.permanent_pool['us'] = jnp.concatenate((self.permanent_pool['us'], u))
        self.permanent_pool['xs_dot'] = jnp.concatenate((self.permanent_pool['xs_dot'], x_dot))
        self.permanent_pool['xs_dot_std'] = jnp.concatenate((self.permanent_pool['xs_dot_std'], std_x_dot))

    def add_data_to_test_pool(self, x: jnp.array, u: jnp.array, x_dot: jnp.array, std_x_dot: jnp.ndarray):
        assert x.shape == x_dot.shape == std_x_dot.shape and x.shape[0] == u.shape[0]
        assert x.shape[1] == self.state_dim and u.shape[1] == self.control_dim
        self.test_pool['xs'] = jnp.concatenate((self.test_pool['xs'], x))
        self.test_pool['us'] = jnp.concatenate((self.test_pool['us'], u))
        self.test_pool['xs_dot'] = jnp.concatenate((self.test_pool['xs_dot'], x_dot))
        self.test_pool['xs_dot_std'] = jnp.concatenate((self.test_pool['xs_dot_std'], std_x_dot))

    def set_training_pool_to_permanent_pool(self):
        self.training_pool = copy.deepcopy(self.permanent_pool)
