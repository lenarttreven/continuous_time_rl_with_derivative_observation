import math
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.main.config import MeasurementCollectionConfig
from cucrl.main.data_stats import DataStats
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.simulator.simulator_dynamics import SimulatorDynamics
from cucrl.utils.classes import DynamicsModel, MeasurementSelection
from cucrl.utils.splines import MultivariateSpline


class TrueDynamicsWrapper(AbstractDynamics):
    def __init__(self, simulator_dynamics: SimulatorDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 measurement_collection_config: MeasurementCollectionConfig = MeasurementCollectionConfig()):
        self.measurement_collection_config = measurement_collection_config
        self.simulator_dynamics = simulator_dynamics
        self.simulator_costs = simulator_costs
        super().__init__(self.simulator_dynamics.state_dim, self.simulator_dynamics.control_dim)

    def mean_and_std_eval_one(self, dynamics_model: DynamicsModel, x: jax.Array,
                              u: jax.Array) -> Tuple[jax.Array, jax.Array]:
        mean = self.simulator_dynamics.dynamics(x, u, jnp.zeros((1,)))
        std = jnp.zeros_like(mean)
        return mean, std

    def mean_eval_one(self, dynamics_model: DynamicsModel, x: jax.Array, u: jax.Array) -> jax.Array:
        return self.simulator_dynamics.dynamics(x, u, jnp.zeros((1,)))

    def initialize_parameters(self, key: jax.random.PRNGKey) -> Tuple[FrozenDict, FrozenDict]:
        return None, None

    def loss(self, params, stats, state, control, state_der, std_state_der, data_stats: DataStats,
             num_train_points: int, key):
        return jnp.array(0.0), None

    def calculate_calibration_alpha(self, dynamics_model: DynamicsModel, xs: jax.Array, us: jax.Array,
                                    xs_dot: jax.Array, xs_dot_std) -> jax.Array:
        return jnp.ones(shape=(self.x_dim,))

    def propose_measurement_times(self, dynamics_model: DynamicsModel, xs_potential: jax.Array, us_potential: jax.Array,
                                  ts_potential: jax.Array, noise_std: float,
                                  num_meas_array: jax.Array) -> MeasurementSelection:
        ts_potential = ts_potential.reshape(-1)

        xs_spline = MultivariateSpline(ts_potential, xs_potential)
        us_spline = MultivariateSpline(ts_potential, us_potential)

        ts_potential = jnp.linspace(jnp.min(ts_potential), jnp.max(ts_potential),
                                    self.measurement_collection_config.num_interpolated_values)

        xs_potential = xs_spline(ts_potential)
        us_potential = us_spline(ts_potential)

        num_step = math.ceil(self.measurement_collection_config.num_interpolated_values // len(num_meas_array))
        greedy_indices = jnp.arange(0, self.measurement_collection_config.num_interpolated_values, num_step).astype(
            jnp.int32)

        ts_potential = ts_potential.reshape(-1, 1)
        proposed_ts = ts_potential[greedy_indices]

        initial_variances = jnp.zeros((xs_potential.shape[0], self.x_dim,))
        assert initial_variances.shape == (xs_potential.shape[0], self.x_dim,)

        return MeasurementSelection(proposed_ts=proposed_ts, potential_ts=ts_potential, potential_us=us_potential,
                                    potential_xs=xs_potential, vars_before_collection=initial_variances,
                                    proposed_indices=greedy_indices)
