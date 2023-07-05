from typing import List, Tuple, Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from cucrl.environment_interactor.interactor import Interactor
from cucrl.main.config import DataGeneratorConfig
from cucrl.simulator.ode_integrator import ForwardEuler
from cucrl.utils.classes import MeasurementSelection, DynamicsData

Schedule = Callable[[int], float]
pytree = Any


class Keys(NamedTuple):
    episode_key: jnp.ndarray
    step_key: jnp.ndarray


class ObservationData(NamedTuple):
    ts: List[jax.Array]
    us: List[jax.Array]
    xs: List[jax.Array]
    xs_dot_true: List[jax.Array]
    xs_dot_noise: List[jax.Array]


class VisualisationData(NamedTuple):
    ts: jax.Array
    us: jax.Array
    xs: jax.Array
    xs_dot_true: jax.Array
    xs_dot_noise: jax.Array


class DataRepr(NamedTuple):
    observation_data: ObservationData
    visualization_data: VisualisationData


class DataGenerator:
    def __init__(self, data_generation: DataGeneratorConfig, interactor: Interactor):
        self.initial_conditions = data_generation.data_collection.initial_conditions
        self.num_visualization_points = data_generation.data_collection.num_visualization_points
        self.state_dim = self.initial_conditions[0].size
        self.control_dim = data_generation.control_dim
        self.stacked_initial_conditions = jnp.stack(self.initial_conditions)

        self.num_trajectories = len(self.initial_conditions)
        self.time_horizon = data_generation.simulator.time_horizon
        self.visualization_times_whole_horizon = jnp.linspace(self.time_horizon.t_min, self.time_horizon.t_max,
                                                              self.num_visualization_points).reshape(-1, 1)
        self.simulator_type = data_generation.simulator.simulator_type
        observation_noise = data_generation.data_collection.noise
        self.simulation_noise = [None] if observation_noise is None else observation_noise
        self.simulator = ForwardEuler(interactor=interactor, simulator_config=data_generation.simulator)

    def generate_trajectories(self, rng: jnp.array) -> Tuple[DataRepr, MeasurementSelection]:
        key, subkey = jax.random.split(rng)
        # Simulate trajectory with the new policy
        trajectories, vis_trajectories, measurement_selection = self.simulator.simulate_trajectories(
            ics=self.initial_conditions, time_horizon=self.time_horizon, num_vis_ts=self.num_visualization_points,
            sigmas=self.simulation_noise, key=subkey, events=self.simulator.interactor.integration_carry)

        observation_data = ObservationData(
            ts=[trajectory.ts.reshape(-1, 1) for trajectory in trajectories],
            us=[trajectory.us for trajectory in trajectories],
            xs=[trajectory.xs for trajectory in trajectories],
            xs_dot_true=[trajectory.xs_dot_true for trajectory in trajectories],
            xs_dot_noise=[trajectory.xs_dot_noise for trajectory in trajectories]
        )

        vis_trajectories = jtu.tree_map(lambda *x: jnp.stack(x), *vis_trajectories)
        visualization_data = VisualisationData(
            ts=jnp.expand_dims(vis_trajectories.ts, -1),
            us=vis_trajectories.us,
            xs=vis_trajectories.xs,
            xs_dot_true=vis_trajectories.xs_dot_true,
            xs_dot_noise=vis_trajectories.xs_dot_noise)

        measurement_selection = jtu.tree_map(lambda *x: jnp.stack(x), *measurement_selection)
        return DataRepr(observation_data, visualization_data), measurement_selection


class DynamicsDataManager:
    def __init__(self, state_dim: int, control_dim: int):
        self.state_dim = state_dim
        self.control_dim = control_dim

        self.permanent_pool = self.initialize_pool()
        self.test_pool = self.initialize_pool()

    def initialize_pool(self) -> DynamicsData:
        ts = jnp.ones(shape=(0, 1))
        xs = jnp.ones(shape=(0, self.state_dim))
        us = jnp.ones(shape=(0, self.control_dim))
        xs_dot = jnp.ones(shape=(0, self.state_dim))
        std_xs_dot = jnp.ones(shape=(0, self.state_dim))
        return DynamicsData(ts=ts, xs=xs, us=us, xs_dot=xs_dot, xs_dot_std=std_xs_dot)

    def add_data_to_permanent_pool(self, dynamics_data: DynamicsData):
        assert dynamics_data.xs.shape == dynamics_data.xs_dot.shape == dynamics_data.xs_dot_std.shape
        assert dynamics_data.xs.shape[0] == dynamics_data.us.shape[0]
        assert dynamics_data.xs.shape[1] == self.state_dim and dynamics_data.us.shape[1] == self.control_dim
        self.permanent_pool = jtu.tree_map(lambda x, y: jnp.concatenate((x, y)), self.permanent_pool, dynamics_data)

    def add_data_to_test_pool(self, dynamics_data: DynamicsData):
        assert dynamics_data.xs.shape == dynamics_data.xs_dot.shape == dynamics_data.xs_dot_std.shape
        assert dynamics_data.xs.shape[0] == dynamics_data.us.shape[0]
        assert dynamics_data.xs.shape[1] == self.state_dim and dynamics_data.us.shape[1] == self.control_dim
        self.test_pool = jtu.tree_map(lambda x, y: jnp.concatenate((x, y)), self.test_pool, dynamics_data)
