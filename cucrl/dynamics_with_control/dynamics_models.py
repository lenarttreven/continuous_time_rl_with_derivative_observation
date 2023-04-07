from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Tuple, Sequence, List

import jax.numpy as jnp
import jax.random
import jax.tree_util as jtu
from flax.core import FrozenDict
from jax import random, vmap, jit
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cho_solve, cho_factor
from jax.scipy.stats import norm, multivariate_normal
from jax.tree_util import tree_map

from cucrl.main.config import MeasurementCollectionConfig
from cucrl.main.data_stats import DataStats, DynamicsData, Normalizer
from cucrl.utils.classes import DynamicsModel, MeasurementSelection
from cucrl.utils.ensembles import DeterministicEnsemble
from cucrl.utils.euler_angles import euler_to_rotation
from cucrl.utils.fSVGD import FSVGD, DataStatsFSVGD, DataRepr
from cucrl.utils.greedy_point_selection import greedy_largest_subdeterminant_jit, greedy_distance_maximization_jit
from cucrl.utils.helper_functions import AngleLayerDynamics
from cucrl.utils.helper_functions import squared_l2_norm, MLP, make_positive
from cucrl.utils.representatives import Dynamics, BNNTypes, BatchStrategy
from cucrl.utils.splines import MultivariateSpline

pytree = Any


def get_dynamics(dynamics_model: Dynamics, state_dim: int, action_dim: int, normalizer: Normalizer,
                 angle_layer: AngleLayerDynamics, dynamics_config,
                 measurement_collection_config: MeasurementCollectionConfig):
    if dynamics_model == Dynamics.FSVGD_VAN_DER_POOL:
        return fSVGDDynamicsVanDerPoolOscilator(state_dim, action_dim, normalizer, dynamics_config.features,
                                                dynamics_config.bandwidth_gp_prior,
                                                dynamics_config.num_particles)
    elif dynamics_model == Dynamics.FSVGD_MOUNTAIN_CART:
        return fSVGDDynamicsMountainCar(state_dim, action_dim, normalizer, dynamics_config.features,
                                        dynamics_config.bandwidth_gp_prior, dynamics_config.num_particles)
    elif dynamics_model == Dynamics.FSVGD_CARTPOLE:
        return fSVGDDynamicsCartpole(state_dim, action_dim, normalizer, dynamics_config.features,
                                     dynamics_config.bandwidth_gp_prior, dynamics_config.num_particles)
    elif dynamics_model == Dynamics.FSVGD_BICYCLE:
        return fSVGDDynamicsBicycle(state_dim, action_dim, normalizer, dynamics_config.features,
                                    dynamics_config.bandwidth_gp_prior, dynamics_config.num_particles)
    elif dynamics_model == Dynamics.FSVGD_PENDULUM:
        return fSVGDDynamicsPendulum(state_dim, action_dim, normalizer, dynamics_config.features,
                                     dynamics_config.bandwidth_gp_prior, dynamics_config.num_particles)
    elif dynamics_model == Dynamics.FSVGD_LV:
        return fSVGDDynamicsLV(state_dim, action_dim, normalizer, dynamics_config.features,
                               dynamics_config.bandwidth_gp_prior, dynamics_config.num_particles)
    elif dynamics_model == Dynamics.FSVGD_FURUTA_PENDULUM:
        return fSVGDDynamicsFurutaPendulum(state_dim, action_dim, normalizer, dynamics_config.features,
                                           dynamics_config.bandwidth_gp_prior, dynamics_config.num_particles)
    elif dynamics_model == Dynamics.FSVGD_QUADROTOR_EULER:
        return fSVGDDynamicsQuadrotorEuler(state_dim, action_dim, normalizer, dynamics_config.features,
                                           dynamics_config.bandwidth_gp_prior, dynamics_config.num_particles)
    elif dynamics_model == Dynamics.FSVGD_GENERAL_AFFINE:
        return fSVGDAffineDynamics(state_dim, action_dim, normalizer, dynamics_config.features,
                                   angle_layer, dynamics_config.bandwidth_gp_prior, dynamics_config.num_particles)
    elif dynamics_model == Dynamics.GP:
        return GPDynamics(state_dim, action_dim, normalizer, angle_layer,
                          measurement_collection_config=measurement_collection_config)
    elif dynamics_model == Dynamics.BNN:
        return BNNDynamics(state_dim, action_dim, normalizer, dynamics_config.features,
                           bnn_type=dynamics_config.bnn_type, bandwidth_prior=dynamics_config.bandwidth_prior,
                           bandwidth_svgd=dynamics_config.bandwidth_svgd,
                           measurement_collection_config=measurement_collection_config)
    else:
        raise NotImplementedError("Chosen dynamics model has not been implemented yet.")


class AbstractDynamics(ABC):
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mean_and_std_eval_batch = vmap(self.mean_and_std_eval_one, in_axes=(None, 0, 0), out_axes=(0, 0),
                                            axis_name='batch')

    @abstractmethod
    def mean_and_std_eval_one(self, dynamics_model: DynamicsModel, x: jax.Array,
                              u: jax.Array) -> Tuple[jax.Array, jax.Array]:
        pass

    @abstractmethod
    def mean_eval_one(self, dynamics_model: DynamicsModel, x: jax.Array, u: jax.Array) -> jax.Array:
        pass

    @abstractmethod
    def initialize_parameters(self, key: jax.random.PRNGKey) -> Tuple[FrozenDict, FrozenDict]:
        pass

    @abstractmethod
    def loss(self, params, stats, state, control, state_der, std_state_der, data_stats: DataStats,
             num_train_points: int, key):
        pass

    @abstractmethod
    def calculate_calibration_alpha(self, dynamics_model: DynamicsModel, xs: jax.Array, us: jax.Array,
                                    xs_dot: jax.Array, xs_dot_std) -> jax.Array:
        pass

    @abstractmethod
    def propose_measurement_times(self, dynamics_model: DynamicsModel, xs_potential: jax.Array, us_potential: jax.Array,
                                  ts_potential: jax.Array, noise_std: float,
                                  num_meas_array: jax.Array) -> MeasurementSelection:
        pass


class BNNDynamics(AbstractDynamics):
    def __init__(self, state_dim: int, action_dim: int, normalizer: Normalizer, features: List[int],
                 bandwidth_prior: float = 1.0, bandwidth_svgd: float = 0.2, num_particles: int = 5,
                 bnn_type: BNNTypes = BNNTypes.DETERMINISTIC_ENSEMBLE,
                 measurement_collection_config: MeasurementCollectionConfig = MeasurementCollectionConfig()):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.measurement_collection_config = measurement_collection_config
        in_shape = normalizer.angle_layer.angle_layer(jnp.zeros((state_dim,))).shape[0]
        if bnn_type == BNNTypes.DETERMINISTIC_ENSEMBLE:
            self.BNN = DeterministicEnsemble(input_dim=in_shape + action_dim, output_dim=state_dim, features=features,
                                             num_particles=num_particles, normalizer=normalizer)
        else:
            # TODO: domain, num measurement points
            self.BNN = FSVGD(input_dim=in_shape + action_dim, output_dim=state_dim, bandwidth_prior=bandwidth_prior,
                             features=features, bandwidth_svgd=bandwidth_svgd, num_particles=num_particles,
                             domain_l=-1.5, domain_u=1.5, num_measurement_points=0, normalizer=normalizer)

    def calculate_calibration_alpha(self, dynamics_model: DynamicsModel, xs: jax.Array, us: jax.Array,
                                    xs_dot: jax.Array, xs_dot_std) -> jax.Array:
        expanded_xs = vmap(self.BNN.normalizer.angle_layer.angle_layer)(xs)
        zs = jnp.concatenate([expanded_xs, us], axis=1)
        input_stats = jtu.tree_map(lambda *w: jnp.concatenate(w), dynamics_model.data_stats.xs_after_angle_layer,
                                   dynamics_model.data_stats.us_stats)
        f_data_stats = DataStatsFSVGD(input_stats=input_stats, output_stats=dynamics_model.data_stats.xs_dot_noise_stats)
        # Todo: pass ps as argument
        num_ps = 10
        ps = jnp.linspace(0, 1, num_ps + 1)[1:]
        return self.BNN.calculate_calibration_alpha(dynamics_model.params, dynamics_model.model_stats, zs, xs_dot,
                                                    xs_dot_std, ps, f_data_stats)

    def _particle_prediction(self, dynamics_model: DynamicsModel, x: jax.Array,
                             u: jax.Array):
        expanded_x = self.BNN.normalizer.angle_layer.angle_layer(x)
        z = jnp.concatenate([expanded_x, u])
        input_stats = jtu.tree_map(lambda *w: jnp.concatenate(w), dynamics_model.data_stats.xs_after_angle_layer,
                                   dynamics_model.data_stats.us_stats)
        f_data_stats = DataStatsFSVGD(input_stats=input_stats, output_stats=dynamics_model.data_stats.xs_dot_noise_stats)
        xs_dot = vmap(self.BNN.apply_eval, in_axes=(0, 0, None, None))(dynamics_model.params,
                                                                       dynamics_model.model_stats, z, f_data_stats)
        return xs_dot

    def mean_and_std_eval_one(self, dynamics_model: DynamicsModel, x: jax.Array,
                              u: jax.Array) -> Tuple[jax.Array, jax.Array]:
        xs_dot = self._particle_prediction(dynamics_model, x, u)
        mean, std = jnp.mean(xs_dot, axis=0), jnp.std(xs_dot, axis=0)
        assert mean.shape == std.shape == (self.state_dim,)
        return mean, std * dynamics_model.calibration_alpha

    def mean_eval_one(self, dynamics_model: DynamicsModel, x: jax.Array, u: jax.Array, ) -> jax.Array:
        return self.mean_and_std_eval_one(dynamics_model, x, u)[0]

    def _covariance(self, dynamics_model, x1, u1, x2, u2):
        xs_dot_1 = self._particle_prediction(dynamics_model, x1, u1)
        xs_dot_2 = self._particle_prediction(dynamics_model, x2, u2)
        return jnp.sum((xs_dot_1 - jnp.mean(xs_dot_1, axis=0)[jnp.newaxis, ...]) * (
                xs_dot_2 - jnp.mean(xs_dot_2, axis=0)[jnp.newaxis, ...]), axis=0) / (self.BNN.num_particles - 1)

    @partial(jit, static_argnums=0)
    def propose_measurement_times(self, dynamics_model: DynamicsModel, xs_potential: jax.Array, us_potential: jax.Array,
                                  ts_potential: jax.Array, noise_std: float,
                                  num_meas_array: jax.Array) -> MeasurementSelection:

        posterior_kernel_v = vmap(self._covariance, in_axes=(None, 0, 0, None, None), out_axes=1)
        posterior_kernel_m = vmap(posterior_kernel_v, in_axes=(None, None, None, 0, 0), out_axes=2)

        # The following does the interpolation of the trajectory through time so that we can select more precisely
        # the points we want to measure
        ts_potential = ts_potential.reshape(-1)

        xs_spline = MultivariateSpline(ts_potential, xs_potential)
        us_spline = MultivariateSpline(ts_potential, us_potential)

        ts_potential = jnp.linspace(jnp.min(ts_potential), jnp.max(ts_potential),
                                    self.measurement_collection_config.num_interpolated_values)

        xs_potential = xs_spline(ts_potential)
        us_potential = us_spline(ts_potential)

        covariance_matrix = posterior_kernel_m(dynamics_model, xs_potential, us_potential, xs_potential, us_potential)

        assert covariance_matrix.shape == (self.state_dim, xs_potential.shape[0], xs_potential.shape[0])
        covariance_matrix = covariance_matrix + noise_std ** 2 * jnp.eye(xs_potential.shape[0])[None, ...]

        if self.measurement_collection_config.batch_strategy == BatchStrategy.MAX_DETERMINANT_GREEDY:
            greedy_indices, potential_indices = greedy_largest_subdeterminant_jit(covariance_matrix, num_meas_array)
        elif self.measurement_collection_config.batch_strategy == BatchStrategy.MAX_KERNEL_DISTANCE_GREEDY:
            greedy_indices, potential_indices = greedy_distance_maximization_jit(covariance_matrix, num_meas_array)

        ts_potential = ts_potential.reshape(-1, 1)
        proposed_ts = ts_potential[greedy_indices]

        initial_variances = vmap(self._covariance, in_axes=(None, 0, 0, 0, 0), out_axes=0)(
            dynamics_model, xs_potential, us_potential, xs_potential, us_potential)
        assert initial_variances.shape == (xs_potential.shape[0], self.state_dim,)

        return MeasurementSelection(proposed_ts=proposed_ts, potential_ts=ts_potential, potential_us=us_potential,
                                    potential_xs=xs_potential, vars_before_collection=initial_variances,
                                    proposed_indices=greedy_indices)

    def loss(self, params, stats, state, control, state_der, std_state_der, data_stats: DataStats, num_train_points,
             key):
        expanded_state = vmap(self.BNN.normalizer.angle_layer.angle_layer)(state)
        z = jnp.concatenate([expanded_state, control], axis=1)
        data = DataRepr(xs=z, ys=state_der)
        input_stats = jtu.tree_map(lambda *x: jnp.concatenate(x), data_stats.xs_after_angle_layer, data_stats.us_stats)
        f_data_stats = DataStatsFSVGD(input_stats=input_stats, output_stats=data_stats.xs_dot_noise_stats)
        return self.BNN.loss(params, stats, data, f_data_stats, std_state_der, num_train_points, key)

    def initialize_parameters(self, key: jax.random.PRNGKey) -> Tuple[FrozenDict, FrozenDict]:
        keys = random.split(key, self.BNN.num_particles)
        return vmap(self.BNN.init_params)(keys)


class GPDynamics(AbstractDynamics):
    def __init__(self, state_dim, action_dim, normalizer, angle_layer: AngleLayerDynamics,
                 measurement_collection_config: MeasurementCollectionConfig):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.angle_layer = angle_layer
        self.normalizer = normalizer
        self.measurement_collection_config = measurement_collection_config
        self.v_kernel = vmap(self.kernel, in_axes=(0, None, None), out_axes=0)
        self.m_kernel = vmap(self.v_kernel, in_axes=(None, 0, None), out_axes=1)
        self.m_kernel_multiple_output = vmap(self.m_kernel, in_axes=(None, None, 0), out_axes=0)
        self.v_kernel_multiple_output = vmap(self.v_kernel, in_axes=(None, None, 0), out_axes=0)
        self.kernel_multiple_output = vmap(self.kernel, in_axes=(None, None, 0), out_axes=0)

    def kernel(self, x, y, params):
        assert x.ndim == y.ndim == 1
        assert params["lengthscale"].shape == x.shape
        return jnp.exp(- jnp.sum((x - y) ** 2 / make_positive(params["lengthscale"]) ** 2))

    def mean_and_std_eval_one(self, dynamics_model: DynamicsModel, x, u):
        assert x.ndim == u.ndim == 1
        expanded_x = self.angle_layer.angle_layer(x)
        expanded_x = self.normalizer.normalize(expanded_x, dynamics_model.data_stats.xs_after_angle_layer)
        u = self.normalizer.normalize(u, dynamics_model.data_stats.us_stats)

        expanded_xs = vmap(self.angle_layer.angle_layer)(dynamics_model.history.xs)
        expanded_xs = vmap(self.normalizer.normalize, in_axes=(0, None))(expanded_xs,
                                                                         dynamics_model.data_stats.xs_after_angle_layer)
        us = vmap(self.normalizer.normalize, in_axes=(0, None))(dynamics_model.history.us,
                                                                dynamics_model.data_stats.us_stats)

        dot_xs = vmap(self.normalizer.normalize, in_axes=(0, None))(dynamics_model.history.xs_dot,
                                                                    dynamics_model.data_stats.xs_dot_noise_stats)
        std_dot_xs = vmap(self.normalizer.normalize_std, in_axes=(0, None))(dynamics_model.history.xs_dot_std,
                                                                            dynamics_model.data_stats.xs_dot_noise_stats)

        cat_input = jnp.concatenate([expanded_x, u])
        inputs = jnp.concatenate([expanded_xs, us], axis=1)
        covariance_matrix = self.m_kernel_multiple_output(inputs, inputs, dynamics_model.params)
        noise_term = vmap(jnp.diag, in_axes=0)(std_dot_xs.T)
        noisy_covariance_matrix = covariance_matrix + noise_term

        k_x_X = vmap(self.v_kernel, in_axes=(None, None, 0), out_axes=0)(inputs, cat_input, dynamics_model.params)
        cholesky_tuples = vmap(jax.scipy.linalg.cho_factor)(noisy_covariance_matrix)

        # Compute std
        denoised_var = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 0))((cholesky_tuples[0], False), k_x_X)
        var = vmap(self.kernel, in_axes=(None, None, 0))(cat_input, cat_input, dynamics_model.params) - vmap(jnp.dot)(
            k_x_X, denoised_var)
        std = jnp.sqrt(var)

        # Compute mean
        denoised_mean = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 1))((cholesky_tuples[0], False),
                                                                                 dot_xs)
        mean = vmap(jnp.dot)(k_x_X, denoised_mean)

        # Denormalize
        mean = self.normalizer.denormalize(mean, dynamics_model.data_stats.xs_dot_noise_stats)
        std = self.normalizer.denormalize_std(std, dynamics_model.data_stats.xs_dot_noise_stats)

        return mean, std

    def loss(self, params, stats, xs, us, dot_xs, std_dot_xs, data_stats: DataStats, num_train_points, key):
        assert xs.shape[0] == us.shape[0] == dot_xs.shape[0] == std_dot_xs.shape[0]
        num_points = xs.shape[0]

        expanded_xs = vmap(self.angle_layer.angle_layer)(xs)
        expanded_xs = vmap(self.normalizer.normalize, in_axes=(0, None))(expanded_xs, data_stats.xs_after_angle_layer)
        us = vmap(self.normalizer.normalize, in_axes=(0, None))(us, data_stats.us_stats)

        dot_xs = vmap(self.normalizer.normalize, in_axes=(0, None))(dot_xs, data_stats.xs_dot_noise_stats)
        std_dot_xs = vmap(self.normalizer.normalize_std, in_axes=(0, None))(std_dot_xs, data_stats.xs_dot_noise_stats)

        inputs = jnp.concatenate([expanded_xs, us], axis=1)

        covariance_matrix = self.m_kernel_multiple_output(inputs, inputs, params)
        noise_term = vmap(jnp.diag, in_axes=0)(std_dot_xs.T)
        noisy_covariance_matrix = covariance_matrix + noise_term

        log_pdf = vmap(multivariate_normal.logpdf, in_axes=(1, None, 0))(dot_xs, jnp.zeros(num_points, ),
                                                                         noisy_covariance_matrix)
        return - jnp.sum(log_pdf), dict()

    def initialize_parameters(self, key):
        expanded_x = self.angle_layer.angle_layer(jnp.ones(shape=(self.state_dim,)))
        u = jnp.ones(shape=(self.action_dim,))
        cat_input = jnp.concatenate([expanded_x, u])
        parameters = dict()
        # Inout dimension is state_dim + action_dim, we have one lengthscale for each dimension
        # Ouput dimension is state_dim, because we have one GP per state dimension
        # So we have lengthscales for each state dimension
        parameters["lengthscale"] = random.normal(key=key, shape=(self.state_dim, cat_input.size))
        return parameters, dict()

    def mean_eval_one(self, dynamics_model: DynamicsModel, x: jax.Array, u: jax.Array) -> jax.Array:
        assert x.ndim == u.ndim == 1
        expanded_x = self.angle_layer.angle_layer(x)
        expanded_x = self.normalizer.normalize(expanded_x, dynamics_model.data_stats.xs_after_angle_layer)
        u = self.normalizer.normalize(u, dynamics_model.data_stats.us_stats)

        expanded_xs = vmap(self.angle_layer.angle_layer)(dynamics_model.history.xs)
        expanded_xs = vmap(self.normalizer.normalize, in_axes=(0, None))(expanded_xs,
                                                                         dynamics_model.data_stats.xs_after_angle_layer)
        us = vmap(self.normalizer.normalize, in_axes=(0, None))(dynamics_model.history.us,
                                                                dynamics_model.data_stats.us_stats)

        dot_xs = vmap(self.normalizer.normalize, in_axes=(0, None))(dynamics_model.history.xs_dot,
                                                                    dynamics_model.data_stats.xs_dot_noise_stats)
        std_dot_xs = vmap(self.normalizer.normalize_std, in_axes=(0, None))(dynamics_model.history.xs_dot_std,
                                                                            dynamics_model.data_stats.xs_dot_noise_stats)

        cat_input = jnp.concatenate([expanded_x, u])
        inputs = jnp.concatenate([expanded_xs, us], axis=1)
        covariance_matrix = self.m_kernel_multiple_output(inputs, inputs, dynamics_model.params)
        noise_term = vmap(jnp.diag, in_axes=0)(std_dot_xs.T)
        noisy_covariance_matrix = covariance_matrix + noise_term

        k_x_X = vmap(self.v_kernel, in_axes=(None, None, 0), out_axes=0)(inputs, cat_input, dynamics_model.params)
        cholesky_tuples = vmap(jax.scipy.linalg.cho_factor)(noisy_covariance_matrix)

        # Compute mean
        denoised_mean = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 1))((cholesky_tuples[0], False),
                                                                                 dot_xs)
        mean = vmap(jnp.dot)(k_x_X, denoised_mean)

        # Denormalize
        mean = self.normalizer.denormalize(mean, dynamics_model.data_stats.xs_dot_noise_stats)
        return mean

    @partial(jit, static_argnums=0)
    def propose_measurement_times(self, dynamics_model: DynamicsModel, xs_potential: jax.Array, us_potential: jax.Array,
                                  ts_potential: jax.Array, noise_std: float,
                                  num_meas_array: jax.Array) -> MeasurementSelection:
        assert xs_potential.shape[0] == us_potential.shape[0] == ts_potential.shape[0]
        assert xs_potential.shape[1] == self.state_dim and us_potential.shape[1] == self.action_dim
        assert ts_potential.shape[1] == 1

        expanded_xs = vmap(self.angle_layer.angle_layer)(dynamics_model.history.xs)
        expanded_xs = vmap(self.normalizer.normalize, in_axes=(0, None))(expanded_xs,
                                                                         dynamics_model.data_stats.xs_after_angle_layer)
        us = vmap(self.normalizer.normalize, in_axes=(0, None))(dynamics_model.history.us,
                                                                dynamics_model.data_stats.us_stats)

        std_dot_xs = vmap(self.normalizer.normalize_std, in_axes=(0, None))(dynamics_model.history.xs_dot_std,
                                                                            dynamics_model.data_stats.xs_dot_noise_stats)

        history_inputs = jnp.concatenate([expanded_xs, us], axis=1)
        covariance_matrix_history = self.m_kernel_multiple_output(history_inputs, history_inputs, dynamics_model.params)

        noise_term = vmap(jnp.diag, in_axes=0)(std_dot_xs.T)
        noisy_covariance_matrix_history = covariance_matrix_history + noise_term
        cho_factor_k_XX = vmap(cho_factor, in_axes=0)(noisy_covariance_matrix_history)

        def posterior_kernel(input_1, input_2):
            k_X_x1 = self.v_kernel_multiple_output(history_inputs, input_1, dynamics_model.params)
            k_X_x2 = self.v_kernel_multiple_output(history_inputs, input_2, dynamics_model.params)
            k_x1_x_2 = self.kernel_multiple_output(input_1, input_2, dynamics_model.params)
            return k_x1_x_2 - vmap(jnp.dot)(k_X_x1,
                                            vmap(cho_solve, in_axes=((0, None), 0))((cho_factor_k_XX[0], False),
                                                                                    k_X_x2))

        # Prepare finely spaced potential measurements
        ts_potential = ts_potential.reshape(-1)

        xs_spline = MultivariateSpline(ts_potential, xs_potential)
        us_spline = MultivariateSpline(ts_potential, us_potential)

        ts_potential = jnp.linspace(jnp.min(ts_potential), jnp.max(ts_potential),
                                    self.measurement_collection_config.num_interpolated_values)

        xs_potential = xs_spline(ts_potential)
        us_potential = us_spline(ts_potential)

        expanded_xs_potential = vmap(self.angle_layer.angle_layer)(xs_potential)
        expanded_xs_potential = vmap(self.normalizer.normalize, in_axes=(0, None))(expanded_xs_potential,
                                                                                   dynamics_model.data_stats.xs_after_angle_layer)
        us_potential = vmap(self.normalizer.normalize, in_axes=(0, None))(us_potential,
                                                                          dynamics_model.data_stats.us_stats)
        potential_inputs = jnp.concatenate([expanded_xs_potential, us_potential], axis=1)

        posterior_kernel_v = vmap(posterior_kernel, in_axes=(0, None), out_axes=1)
        posterior_kernel_m = vmap(posterior_kernel_v, in_axes=(None, 0), out_axes=2)

        covariance_matrix = posterior_kernel_m(potential_inputs, potential_inputs)

        assert covariance_matrix.shape == (self.state_dim, xs_potential.shape[0], xs_potential.shape[0])
        covariance_matrix = covariance_matrix + noise_std ** 2 * jnp.eye(xs_potential.shape[0])[None, ...]

        if self.measurement_collection_config.batch_strategy == BatchStrategy.MAX_DETERMINANT_GREEDY:
            greedy_indices, potential_indices = greedy_largest_subdeterminant_jit(covariance_matrix, num_meas_array)
        elif self.measurement_collection_config.batch_strategy == BatchStrategy.MAX_KERNEL_DISTANCE_GREEDY:
            greedy_indices, potential_indices = greedy_distance_maximization_jit(covariance_matrix, num_meas_array)

        ts_potential = ts_potential.reshape(-1, 1)
        proposed_ts = ts_potential[greedy_indices]

        initial_variances = vmap(posterior_kernel, in_axes=(0, 0), out_axes=0)(potential_inputs, potential_inputs)
        assert initial_variances.shape == (xs_potential.shape[0], self.state_dim,)

        return MeasurementSelection(proposed_ts=proposed_ts, potential_ts=ts_potential, potential_us=us_potential,
                                    potential_xs=xs_potential, vars_before_collection=initial_variances,
                                    proposed_indices=greedy_indices)

    def calculate_calibration_alpha(self, dynamics_model: DynamicsModel, xs: jax.Array, us: jax.Array,
                                    xs_dot: jax.Array, xs_dot_std) -> jax.Array:
        return jnp.ones(shape=(self.state_dim,))


class AbstractfSVGDDynamics(AbstractDynamics):
    def __init__(self, state_dim, action_dim, normalizer, features: Sequence[int], prior_h=1.0, stein_h: float = 0.2,
                 num_particles: int = 5):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.num_particles = num_particles
        self.normalizer = normalizer
        self.features = features
        self.prior_h = prior_h
        self.stein_h = stein_h
        self.model = MLP(features=self.features, output_dim=self.state_dim)
        self.stein_kernel, self.stein_kernel_derivative = self._prepare_stein_kernel(h=self.stein_h ** 2)
        self.prior_kernel = self._prepare_prior_kernel(h=self.prior_h ** 2)
        self._unravel_one = self._prepare_unravel_one()
        self.ravel_ensemble = vmap(self.ravel_one)

    @abstractmethod
    def _control_affine_part(self, x, u):
        pass

    def _norm_control_affine_part(self, x, u, data_stats):
        return self.normalizer.normalize(self._control_affine_part(x, u), data_stats.xs_dot_noise_stats)

    def _state_dynamics_train_one(self, params, stats, x, data_stats):
        assert x.shape == (self.state_dim,)
        x = self.normalizer.normalize(x, data_stats.xs_stats)
        net_out, new_state = self.model.apply({'params': params, **stats}, x, mutable=list(stats.keys()), train=True)
        return net_out, new_state

    def _state_dynamics_eval_one(self, params, stats, x, data_stats):
        assert x.shape == (self.state_dim,)
        x = self.normalizer.normalize(x, data_stats.xs_stats)
        net_out = self.model.apply({'params': params, **stats}, x)
        return net_out

    def _dynamics_eval_one(self, params, stats, x, u, data_stats):
        assert x.shape == (self.state_dim,) and u.shape == (self.action_dim,)
        f = self._state_dynamics_eval_one(params, stats, x, data_stats)
        return f + self._norm_control_affine_part(x, u, data_stats)

    def eval_index(self, params: pytree, stats: FrozenDict, x: jax.Array, u: jax.Array,
                   data_stats: DataStats, dynamics_idx: int) -> jax.Array:
        params_particle = tree_map(lambda z: z[dynamics_idx, ...], params)
        stats_particle = tree_map(lambda z: z[dynamics_idx, ...], params)
        x_dot = self._dynamics_eval_one(params_particle, stats_particle, x, u, data_stats)
        denormalize = self.normalizer.denormalize
        return denormalize(x_dot, data_stats.xs_dot_noise_stats)

    def _dynamics_train_one(self, params, stats, x, u, data_stats):
        assert x.shape == (self.state_dim,) and u.shape == (self.action_dim,)
        net_out, new_state = self._state_dynamics_train_one(params, stats, x, data_stats)
        return net_out + self._norm_control_affine_part(x, u, data_stats), new_state

    def mean_train_one(self, params, stats, x, u, data_stats):
        means, new_stats = vmap(self._state_dynamics_train_one, in_axes=(0, 0, None, None), out_axes=0)(params, stats,
                                                                                                        x, data_stats)
        return jnp.mean(means, axis=0) + self._norm_control_affine_part(x, u, data_stats), new_stats

    def std_train_one(self, params, stats, x, data_stats):
        means, new_stats = vmap(self._state_dynamics_train_one, in_axes=(0, 0, None, None), out_axes=0)(params, stats,
                                                                                                        x, data_stats)
        return jnp.std(means, axis=0), new_stats

    def mean_and_std_train_one(self, params, stats, x, u, data_stats):
        means, new_stats = vmap(self._state_dynamics_train_one, in_axes=(0, 0, None, None), out_axes=0)(params, stats,
                                                                                                        x, data_stats)
        return jnp.mean(means, axis=0) + self._norm_control_affine_part(x, u, data_stats), jnp.std(means,
                                                                                                   axis=0), new_stats

    def mean_eval_one(self, params, stats, x, u, dynamics_data: DynamicsData, data_stats):
        means = vmap(self._state_dynamics_eval_one, in_axes=(0, 0, None, None), out_axes=0)(params, stats, x,
                                                                                            data_stats)
        denormalize = self.normalizer.denormalize
        mean = jnp.mean(means, axis=0) + self._norm_control_affine_part(x, u, data_stats)
        return denormalize(mean, data_stats.xs_dot_noise_stats)

    def std_eval_one(self, params, stats, x, data_stats):
        means = vmap(self._state_dynamics_eval_one, in_axes=(0, 0, None, None), out_axes=0)(params, stats, x,
                                                                                            data_stats)
        return jnp.std(means, axis=0)

    def mean_and_std_eval_one(self, params, stats, x, u, dynamics_data: DynamicsData, data_stats):
        means = vmap(self._state_dynamics_eval_one, in_axes=(0, 0, None, None), out_axes=0)(params, stats, x,
                                                                                            data_stats)
        mean = jnp.mean(means, axis=0) + self._norm_control_affine_part(x, u, data_stats)
        std = jnp.std(means, axis=0)

        denormalize = self.normalizer.denormalize
        denormalize_std = self.normalizer.denormalize_std

        return denormalize(mean, data_stats.xs_dot_noise_stats), denormalize_std(std, data_stats.xs_dot_noise_stats)

    def _initialize_parameters(self, key):
        variables = self.model.init(key, jnp.ones(shape=(self.state_dim,)))
        # Split state and params (which are updated by optimizer).
        if 'params' in variables:
            state, params = variables.pop('params')
        else:
            state, params = variables, FrozenDict({})
        del variables  # Delete variables to avoid wasting resources
        return params, state

    def initialize_parameters(self, key):
        key, *subkeys = random.split(key, self.num_particles + 1)
        return vmap(self._initialize_parameters)(jnp.stack(subkeys))

    def regularization(self, params, weights) -> jax.Array:
        # We use mean instead of sum!
        # return weights.wd_dynamics * jnp.mean((ravel_pytree(params)[0] ** 2))
        return jnp.array(0.0)

    def sample_dynamics_model(self, params, stats, key, data_stats):
        sample = random.randint(key, minval=0, maxval=self.num_particles, shape=())

        def _one_dynamics(x, u):
            x_dot = self._dynamics_eval_one(tree_map(lambda x: x[sample, ...], params),
                                            tree_map(lambda x: x[sample, ...], stats), x, u, data_stats)
            denormalize = self.normalizer.denormalize
            return denormalize(x_dot, data_stats.xs_dot_noise_stats)

        return _one_dynamics

    def dynamics_model_idx(self, params, stats, idx, data_stats):

        def _one_dynamics(x, u):
            x_dot = self._dynamics_eval_one(tree_map(lambda x: x[idx, ...], params),
                                            tree_map(lambda x: x[idx, ...], stats), x, u, data_stats)
            denormalize = self.normalizer.denormalize
            return denormalize(x_dot, data_stats.xs_dot_noise_stats)

        return _one_dynamics

    def _neg_log_posterior(self, pred_raw: jax.Array, x_stacked: jax.Array, y_batch: jax.Array,
                           scale: jax.Array, num_train_points):
        assert pred_raw.shape == y_batch.shape == scale.shape
        nll = self._nll(pred_raw, y_batch, scale)
        neg_log_prior = - self._gp_prior_log_prob(x_stacked, pred_raw) / num_train_points
        neg_log_post = nll + neg_log_prior
        return neg_log_post

    @staticmethod
    def _nll(pred_raw: jax.Array, y_batch: jax.Array, scale: jax.Array):
        log_prob = norm.logpdf(y_batch, loc=pred_raw, scale=scale)
        return - jnp.mean(log_prob)

    def _gp_prior_log_prob(self, x: jnp.array, y: jnp.array, eps: float = 1e-6) -> jax.Array:
        # Multiple dimension outputs are handled independently per dimension
        k = self.prior_kernel(x) + eps * jnp.eye(x.shape[0])

        def evaluate_fs(fs):
            assert fs.shape == (x.shape[0],) and fs.ndim == 1
            return multivariate_normal.logpdf(fs, mean=jnp.zeros(x.shape[0]), cov=k)

        evaluate_fs_multiple_dims = vmap(evaluate_fs, in_axes=1, out_axes=0)
        evaluate_ensemble = vmap(evaluate_fs_multiple_dims, in_axes=0, out_axes=0)
        return jnp.mean(evaluate_ensemble(y))

    def loss(self, params, stats, xs, us, dot_xs, std_dot_xs, data_stats: DataStats):
        assert xs.shape == dot_xs.shape == std_dot_xs.shape and xs.shape[0] == us.shape[0]
        assert xs.shape[1] == self.state_dim and us.shape[1] == self.action_dim

        mean_batch = vmap(self._dynamics_train_one, in_axes=(None, None, 0, 0, None), out_axes=(0, 0))
        mean_batch_ensemble = vmap(mean_batch, in_axes=(0, 0, None, None, None), out_axes=(0, None),
                                   axis_name='batch')
        x_dot_pred, new_stats = mean_batch_ensemble(params, stats, xs, us, data_stats)

        normalize = vmap(self.normalizer.normalize, in_axes=(0, None))
        xs_norm = normalize(xs, data_stats.xs_stats)
        us_norm = normalize(us, data_stats.us_stats)
        x_batch = jnp.concatenate([xs_norm, us_norm], axis=1)

        norm_stds = vmap(self.normalizer.normalize_std, in_axes=(0, None))(std_dot_xs, data_stats.xs_dot_noise_stats)
        norm_stds_repeated = jnp.repeat(norm_stds[jnp.newaxis, ...], repeats=self.num_particles, axis=0)

        norm_xs_dot = vmap(self.normalizer.normalize, in_axes=(0, None))(dot_xs, data_stats.xs_dot_noise_stats)
        norm_xs_dot_repeated = jnp.repeat(norm_xs_dot[jnp.newaxis, ...], repeats=self.num_particles, axis=0)

        assert x_dot_pred.shape == (self.num_particles, xs.shape[0], self.state_dim)
        assert x_dot_pred.shape == norm_xs_dot_repeated.shape == norm_stds_repeated.shape
        num_train_points = xs.shape[0]
        grad_post = jax.grad(self._neg_log_posterior)(x_dot_pred, x_batch, norm_xs_dot_repeated, norm_stds_repeated,
                                                      num_train_points)
        # kernel
        k = self.stein_kernel(x_dot_pred)
        k_x = self.stein_kernel_derivative(x_dot_pred)
        grad_k = jnp.mean(k_x, axis=0)
        surrogate_loss = jnp.sum(x_dot_pred * jax.lax.stop_gradient(
            jnp.einsum('ij,jkm', k, grad_post) - grad_k)) + 1e-4 * squared_l2_norm(params)
        return surrogate_loss, new_stats

    @staticmethod
    def _prepare_prior_kernel(h=1.0 ** 2):
        def k(x, y):
            return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h))

        v_k = vmap(k, in_axes=(0, None), out_axes=0)
        m_k = vmap(v_k, in_axes=(None, 0), out_axes=1)

        def kernel(fs):
            kernel_matrix = m_k(fs, fs)
            return kernel_matrix

        return kernel

    @staticmethod
    def _prepare_stein_kernel(h=0.2 ** 2):
        def k(x, y):
            return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h))

        v_k = vmap(k, in_axes=(0, None), out_axes=0)
        m_k = vmap(v_k, in_axes=(None, 0), out_axes=1)

        def kernel(fs):
            kernel_matrix = m_k(fs, fs)
            return kernel_matrix

        k_x = jax.grad(k, argnums=0)

        v_k_der = vmap(k_x, in_axes=(0, None), out_axes=0)
        m_k_der = vmap(v_k_der, in_axes=(None, 0), out_axes=1)

        def kernel_derivative(fs):
            return m_k_der(fs, fs)

        return kernel, kernel_derivative

    @staticmethod
    def multiply_by_kernel(params, k):
        return jnp.einsum('ij,i...->j...', k, params) / k.shape[0]

    @staticmethod
    def ravel_one(params):
        return ravel_pytree(params)[0]

    def _prepare_unravel_one(self):
        return ravel_pytree(self._initialize_parameters(random.PRNGKey(0))[0])[1]


class fSVGDAffineDynamics(AbstractDynamics):
    """
    We reprensent dynamics as \dot x = \a(x) + b(x)u
    """

    def __init__(self, state_dim, action_dim, normalizer, features: Sequence[int], angle_layer: AngleLayerDynamics,
                 prior_h=1.0, num_particles: int = 5):
        super().__init__(state_dim=state_dim, action_dim=action_dim)
        self.angle_layer = angle_layer
        self.num_particles = num_particles
        self.normalizer = normalizer
        self.features = features
        self.prior_h = prior_h
        # Model outputs are a (first self.state_dim dimensions) and b (last self.state_dim * self.action_dim dimensions)
        self.model = MLP(features=self.features, output_dim=self.state_dim + self.state_dim * self.action_dim)

        self.stein_kernel, self.stein_kernel_derivative = self._prepare_stein_kernel()
        self.prior_kernel = self._prepare_prior_kernel(h=self.prior_h)

        self._unravel_one = self._prepare_unravel_one()
        self.ravel_ensemble = vmap(self.ravel_one)

    def _dynamics_train_one(self, params, stats, x, u, data_stats: DataStats):
        assert x.shape == (self.state_dim,) and u.shape == (self.action_dim,)
        expanded_x = self.angle_layer.angle_layer(x)
        expanded_x = self.normalizer.normalize(expanded_x, data_stats.xs_after_angle_layer)
        net_out, new_state = self.model.apply({'params': params, **stats}, expanded_x, mutable=list(stats.keys()),
                                              train=True)
        a = net_out[:self.state_dim]
        b = net_out[self.state_dim:].reshape(self.state_dim, self.action_dim)
        return a + b @ u, new_state

    def _dynamics_eval_one(self, params, stats, x, u, data_stats):
        assert x.shape == (self.state_dim,) and u.shape == (self.action_dim,)
        expanded_x = self.angle_layer.angle_layer(x)
        expanded_x = self.normalizer.normalize(expanded_x, data_stats.xs_after_angle_layer)
        net_out = self.model.apply({'params': params, **stats}, expanded_x)
        a = net_out[:self.state_dim]
        b = net_out[self.state_dim:].reshape(self.state_dim, self.action_dim)
        return a + b @ u

    def eval_index(self, params: pytree, stats: FrozenDict, x: jax.Array, u: jax.Array,
                   data_stats: DataStats, dynamics_idx: int) -> jax.Array:
        params_particle = tree_map(lambda z: z[dynamics_idx, ...], params)
        stats_particle = tree_map(lambda z: z[dynamics_idx, ...], params)
        x_dot = self._dynamics_eval_one(params_particle, stats_particle, x, u, data_stats)
        denormalize = self.normalizer.denormalize
        return denormalize(x_dot, data_stats.xs_dot_noise_stats)

    def mean_train_one(self, params, stats, x, u, data_stats):
        means, new_stats = vmap(self._dynamics_train_one, in_axes=(0, 0, None, None, None), out_axes=0)(params, stats,
                                                                                                        x, u,
                                                                                                        data_stats)
        return jnp.mean(means, axis=0), new_stats

    def std_train_one(self, params, stats, x, u, data_stats):
        means, new_stats = vmap(self._dynamics_train_one, in_axes=(0, 0, None, None, None), out_axes=0)(params, stats,
                                                                                                        x, u,
                                                                                                        data_stats)
        return jnp.std(means, axis=0), new_stats

    def mean_and_std_train_one(self, params, stats, x, u, data_stats):
        means, new_stats = vmap(self._dynamics_train_one, in_axes=(0, 0, None, None, None), out_axes=0)(params, stats,
                                                                                                        x, u,
                                                                                                        data_stats)
        return jnp.mean(means, axis=0), jnp.std(means, axis=0), new_stats

    def mean_eval_one(self, params, stats, x, u, dynamics_data: DynamicsData, data_stats):
        means = vmap(self._dynamics_eval_one, in_axes=(0, 0, None, None, None), out_axes=0)(params, stats, x, u,
                                                                                            data_stats)
        denormalize = self.normalizer.denormalize
        mean = jnp.mean(means, axis=0)
        return denormalize(mean, data_stats.xs_dot_noise_stats)

    def std_eval_one(self, params, stats, x, u, data_stats):
        means = vmap(self._dynamics_eval_one, in_axes=(0, 0, None, None, None), out_axes=0)(params, stats, x, u,
                                                                                            data_stats)
        return jnp.std(means, axis=0)

    def mean_and_std_eval_one(self, params, stats, x, u, dynamics_data: DynamicsData, data_stats):
        means = vmap(self._dynamics_eval_one, in_axes=(0, 0, None, None, None), out_axes=0)(params, stats, x, u,
                                                                                            data_stats)
        mean = jnp.mean(means, axis=0)
        std = jnp.std(means, axis=0)

        denormalize = self.normalizer.denormalize
        denormalize_std = self.normalizer.denormalize_std

        return denormalize(mean, data_stats.xs_dot_noise_stats), denormalize_std(std, data_stats.xs_dot_noise_stats)

    def _initialize_parameters(self, key):
        init_input = self.angle_layer.angle_layer(jnp.ones(shape=(self.state_dim,)))
        variables = self.model.init(key, init_input)
        # Split state and params (which are updated by optimizer).
        if 'params' in variables:
            state, params = variables.pop('params')
        else:
            state, params = variables, FrozenDict({})
        del variables  # Delete variables to avoid wasting resources
        return params, state

    def initialize_parameters(self, key):
        key, *subkeys = random.split(key, self.num_particles + 1)
        return vmap(self._initialize_parameters)(jnp.stack(subkeys))

    def regularization(self, params, weights) -> jax.Array:
        # We use mean instead of sum!
        # return weights.wd_dynamics * jnp.mean((ravel_pytree(params)[0] ** 2))
        return jnp.array(0.0)

    def sample_dynamics_model(self, params, stats, key, data_stats):
        sample = random.randint(key, minval=0, maxval=self.num_particles, shape=())

        def _one_dynamics(x, u):
            x_dot = self._dynamics_eval_one(tree_map(lambda x: x[sample, ...], params),
                                            tree_map(lambda x: x[sample, ...], stats), x, u, data_stats)
            denormalize = self.normalizer.denormalize
            return denormalize(x_dot, data_stats.xs_dot_noise_stats)

        return _one_dynamics

    def dynamics_model_idx(self, params, stats, idx, data_stats):

        def _one_dynamics(x, u):
            x_dot = self._dynamics_eval_one(tree_map(lambda x: x[idx, ...], params),
                                            tree_map(lambda x: x[idx, ...], stats), x, u, data_stats)
            denormalize = self.normalizer.denormalize
            return denormalize(x_dot, data_stats.xs_dot_noise_stats)

        return _one_dynamics

    def _neg_log_posterior(self, pred_raw: jax.Array, x_stacked: jax.Array, y_batch: jax.Array,
                           scale: jax.Array, num_train_points):
        assert pred_raw.shape == y_batch.shape == scale.shape
        nll = self._nll(pred_raw, y_batch, scale)
        neg_log_prior = - self._gp_prior_log_prob(x_stacked, pred_raw) / num_train_points
        neg_log_post = nll + neg_log_prior
        return neg_log_post

    @staticmethod
    def _nll(pred_raw: jax.Array, y_batch: jax.Array, scale: jax.Array):
        log_prob = norm.logpdf(y_batch, loc=pred_raw, scale=scale)
        return - jnp.mean(log_prob)

    def _gp_prior_log_prob(self, x: jnp.array, y: jnp.array, eps: float = 1e-6) -> jax.Array:
        # Multiple dimension outputs are handled independently per dimension
        k = self.prior_kernel(x) + eps * jnp.eye(x.shape[0])

        def evaluate_fs(fs):
            assert fs.shape == (x.shape[0],) and fs.ndim == 1
            return multivariate_normal.logpdf(fs, mean=jnp.zeros(x.shape[0]), cov=k)

        evaluate_fs_multiple_dims = vmap(evaluate_fs, in_axes=1, out_axes=0)
        evaluate_ensemble = vmap(evaluate_fs_multiple_dims, in_axes=0, out_axes=0)
        return jnp.mean(evaluate_ensemble(y))

    def loss(self, params, stats, xs, us, dot_xs, std_dot_xs, data_stats: DataStats):
        assert xs.shape == dot_xs.shape == std_dot_xs.shape and xs.shape[0] == us.shape[0]
        assert xs.shape[1] == self.state_dim and us.shape[1] == self.action_dim

        mean_batch = vmap(self._dynamics_train_one, in_axes=(None, None, 0, 0, None), out_axes=(0, 0))
        mean_batch_ensemble = vmap(mean_batch, in_axes=(0, 0, None, None, None), out_axes=(0, None),
                                   axis_name='batch')

        x_dot_pred, new_stats = mean_batch_ensemble(params, stats, xs, us, data_stats)

        normalize = vmap(self.normalizer.normalize, in_axes=(0, None))

        expanded_xs = vmap(self.angle_layer.angle_layer)(xs)
        expanded_xs_norm = normalize(expanded_xs, data_stats.xs_after_angle_layer)
        us_norm = normalize(us, data_stats.us_stats)

        x_batch = jnp.concatenate([expanded_xs_norm, us_norm], axis=1)

        norm_stds = vmap(self.normalizer.normalize_std, in_axes=(0, None))(std_dot_xs, data_stats.xs_dot_noise_stats)
        norm_stds_repeated = jnp.repeat(norm_stds[jnp.newaxis, ...], repeats=self.num_particles, axis=0)

        norm_xs_dot = vmap(self.normalizer.normalize, in_axes=(0, None))(dot_xs, data_stats.xs_dot_noise_stats)
        norm_xs_dot_repeated = jnp.repeat(norm_xs_dot[jnp.newaxis, ...], repeats=self.num_particles, axis=0)

        assert x_dot_pred.shape == (self.num_particles, xs.shape[0], self.state_dim)
        assert x_dot_pred.shape == norm_xs_dot_repeated.shape == norm_stds_repeated.shape

        num_train_points = xs.shape[0]
        grad_post = jax.grad(self._neg_log_posterior)(x_dot_pred, x_batch, norm_xs_dot_repeated, norm_stds_repeated,
                                                      num_train_points)
        # kernel
        k = self.stein_kernel(x_dot_pred)
        k_x = self.stein_kernel_derivative(x_dot_pred)
        grad_k = jnp.mean(k_x, axis=0)
        surrogate_loss = jnp.sum(x_dot_pred * jax.lax.stop_gradient(
            jnp.einsum('ij,jkm', k, grad_post) - grad_k)) + 1e-4 * squared_l2_norm(params)
        return surrogate_loss, new_stats

    @staticmethod
    def _prepare_prior_kernel(h=1.0 ** 2):
        def k(x, y):
            return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h))

        v_k = vmap(k, in_axes=(0, None), out_axes=0)
        m_k = vmap(v_k, in_axes=(None, 0), out_axes=1)

        def kernel(fs):
            kernel_matrix = m_k(fs, fs)
            return kernel_matrix

        return kernel

    @staticmethod
    def _prepare_stein_kernel(h=0.2 ** 2):
        def k(x, y):
            return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h))

        v_k = vmap(k, in_axes=(0, None), out_axes=0)
        m_k = vmap(v_k, in_axes=(None, 0), out_axes=1)

        def kernel(fs):
            kernel_matrix = m_k(fs, fs)
            return kernel_matrix

        k_x = jax.grad(k, argnums=0)

        v_k_der = vmap(k_x, in_axes=(0, None), out_axes=0)
        m_k_der = vmap(v_k_der, in_axes=(None, 0), out_axes=1)

        def kernel_derivative(fs):
            return m_k_der(fs, fs)

        return kernel, kernel_derivative

    @staticmethod
    def multiply_by_kernel(params, k):
        return jnp.einsum('ij,i...->j...', k, params) / k.shape[0]

    @staticmethod
    def ravel_one(params):
        return ravel_pytree(params)[0]

    def _prepare_unravel_one(self):
        return ravel_pytree(self._initialize_parameters(random.PRNGKey(0))[0])[1]


class fSVGDDynamicsMountainCar(AbstractfSVGDDynamics):
    def __init__(self, state_dim, action_dim, normalizer, features: Sequence[int], prior_h=1.0, num_particles: int = 5):
        super().__init__(state_dim=state_dim, action_dim=action_dim, normalizer=normalizer, features=features,
                         num_particles=num_particles, prior_h=prior_h, )

    def _control_affine_part(self, x, u):
        assert u.shape == (self.action_dim,)
        return 100 * u @ jnp.array([[0.0, 0.001]], dtype=jnp.float64)


class fSVGDDynamicsQuadrotorEuler(AbstractfSVGDDynamics):
    def __init__(self, state_dim, action_dim, normalizer, features: Sequence[int], prior_h=1.0, num_particles: int = 5):
        super().__init__(state_dim=state_dim, action_dim=action_dim, normalizer=normalizer, features=features,
                         num_particles=num_particles, prior_h=prior_h, )
        self.mass = 0.18  # kg
        self.g = 9.81  # m/s^2
        self.arm_length = 0.086  # meter
        self.height = 0.05

        self.I = jnp.array([(0.00025, 0, 2.55e-6),
                            (0, 0.000232, 0),
                            (2.55e-6, 0, 0.0003738)])

        self.invI = jnp.linalg.inv(self.I)

        self.minF = 0.0
        self.maxF = 2.0 * self.mass * self.g

        self.km = 1.5e-9
        self.kf = 6.11e-8
        self.r = self.km / self.kf

        self.L = self.arm_length
        self.H = self.height
        #  [ F  ]         [ F1 ]
        #  | M1 |  = A *  | F2 |
        #  | M2 |         | F3 |
        #  [ M3 ]         [ F4 ]
        self.A = jnp.array([[1, 1, 1, 1],
                            [0, self.L, 0, -self.L],
                            [-self.L, 0, self.L, 0],
                            [self.r, -self.r, self.r, -self.r]])

        self.invA = jnp.linalg.inv(self.A)

        self.body_frame = jnp.array([(self.L, 0, 0, 1),
                                     (0, self.L, 0, 1),
                                     (-self.L, 0, 0, 1),
                                     (0, -self.L, 0, 1),
                                     (0, 0, 0, 1),
                                     (0, 0, self.H, 1)])

        self.B = jnp.array([[0, self.L, 0, -self.L],
                            [-self.L, 0, self.L, 0]])

        self.internal_control_scaling_inv = jnp.diag(jnp.array([1, 2 * 1e-4, 2 * 1e-4, 1e-3], dtype=jnp.float64))
        self.internal_state_scaling = jnp.diag(jnp.array([1, 1, 1, 1, 1, 1, 10, 10, 1, 10, 10, 1], dtype=jnp.float64))
        self.internal_state_scaling_inv = jnp.diag(
            jnp.array([1, 1, 1, 1, 1, 1, 1 / 10, 1 / 10, 1, 1 / 10, 1 / 10, 1], dtype=jnp.float64))

    def _control_affine_part(self, x, u):
        assert u.shape == (self.action_dim,)
        x = self.internal_state_scaling_inv @ x
        u = self.internal_control_scaling_inv @ u
        F, M = u[0], u[1:]
        phi, theta, psi = x[6:9]
        angles = jnp.array([phi, theta, psi])
        wRb = euler_to_rotation(angles)
        # acceleration - Newton's second law of motion
        accel = 1.0 / self.mass * (wRb.dot(jnp.array([[0, 0, F]]).T))
        pqrdot = self.invI.dot(M.flatten())
        state_dot_0 = 0
        state_dot_1 = 0
        state_dot_2 = 0
        state_dot_3 = accel[0].reshape()
        state_dot_4 = accel[1].reshape()
        state_dot_5 = accel[2].reshape()
        state_dot_6 = 0
        state_dot_7 = 0
        state_dot_8 = 0
        state_dot_9 = pqrdot[0]
        state_dot_10 = pqrdot[1]
        state_dot_11 = pqrdot[2]
        return self.internal_state_scaling @ jnp.array([state_dot_0, state_dot_1, state_dot_2, state_dot_3, state_dot_4,
                                                        state_dot_5, state_dot_6, state_dot_7, state_dot_8, state_dot_9,
                                                        state_dot_10, state_dot_11])


class fSVGDDynamicsFurutaPendulum(AbstractfSVGDDynamics):
    def __init__(self, state_dim, action_dim, normalizer, features: Sequence[int], prior_h=1.0, num_particles: int = 5,
                 system_params=jnp.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0]), g=0.2):
        super().__init__(state_dim=state_dim, action_dim=action_dim, normalizer=normalizer, features=features,
                         num_particles=num_particles, prior_h=prior_h)
        self.system_params = system_params
        self.g = g
        (J, M, m_a, m_p, l_a, l_p) = system_params
        self.alpha = J + (M + 1 / 3 * m_a + m_p) * l_a ** 2
        self.beta = (M + 1 / 3 * m_p) * l_p ** 2
        self.gamma = (M + 1 / 2 * m_p) * l_a * l_p
        self.delta = (M + 1 / 2 * m_p) * g * l_p

    def _control_affine_part(self, x, u):
        assert u.shape == (self.action_dim,)
        t_phi = u.reshape()
        x0_dot = 0

        denom = self.alpha * self.beta - self.gamma ** 2 + (self.beta ** 2 + self.gamma ** 2) * jnp.sin(x[2]) ** 2
        num = self.beta * t_phi
        x1_dot = num / denom

        x2_dot = 0

        num = -1 * self.gamma * jnp.cos(x[2]) * t_phi
        x3_dot = num / denom
        return jnp.array([x0_dot, x1_dot, x2_dot, x3_dot])


class fSVGDDynamicsCartpole(AbstractfSVGDDynamics):
    def __init__(self, state_dim, action_dim, normalizer, features: Sequence[int], prior_h=1.0, num_particles: int = 5,
                 system_params=jnp.array([0.5, 1.0, 0.5]), g=0.2):
        super().__init__(state_dim=state_dim, action_dim=action_dim, normalizer=normalizer, features=features,
                         num_particles=num_particles, prior_h=prior_h)
        self.system_params = system_params
        self.g = g
        L, M, m_p = system_params
        self.L = L
        self.M = M
        self.m_p = m_p

    def _control_affine_part(self, x, u):
        assert u.shape == (self.action_dim,)
        x0_dot = 0
        num = jnp.cos(x[0]) * u[0]
        denom = (self.M + self.m_p * (1 - jnp.cos(x[0]) ** 2)) * self.L
        x1_dot = num / denom
        x2_dot = 0
        num = u[0]
        denom = self.M + self.m_p * (1 - jnp.cos(x[0]) ** 2)
        x3_dot = num / denom
        return jnp.array([x0_dot, x1_dot, x2_dot, x3_dot])


class fSVGDDynamicsPendulum(AbstractfSVGDDynamics):
    def __init__(self, state_dim, action_dim, normalizer, features: Sequence[int], prior_h=1.0, num_particles: int = 5):
        super().__init__(state_dim=state_dim, action_dim=action_dim, normalizer=normalizer, features=features,
                         num_particles=num_particles, prior_h=prior_h)

    def _control_affine_part(self, x, u):
        assert u.shape == (self.action_dim,)
        return u @ jnp.array([[0.0, 1.0]])


class fSVGDDynamicsBicycle(AbstractfSVGDDynamics):
    def __init__(self, state_dim, action_dim, normalizer, features: Sequence[int], prior_h=1.0, num_particles: int = 5,
                 system_params=jnp.array([1.0])):
        super().__init__(state_dim=state_dim, action_dim=action_dim, normalizer=normalizer, features=features,
                         num_particles=num_particles, prior_h=prior_h)
        self.system_params = system_params

    def _control_affine_part(self, x, u):
        assert u.shape == (self.action_dim,)
        x0_dot = 0
        x1_dot = 0
        x2_dot = x[3] * u[0] / self.system_params[0]
        x3_dot = u[1]
        return jnp.array([x0_dot, x1_dot, x2_dot, x3_dot], dtype=jnp.float64)


class fSVGDDynamicsVanDerPoolOscilator(AbstractfSVGDDynamics):
    def __init__(self, state_dim, action_dim, normalizer, features: Sequence[int], prior_h=1.0, num_particles: int = 5):
        super().__init__(state_dim=state_dim, action_dim=action_dim, normalizer=normalizer, features=features,
                         num_particles=num_particles, prior_h=prior_h)

    def _control_affine_part(self, x, u):
        assert u.shape == (self.action_dim,)
        return u @ jnp.array([[0.0, 1.0]])


class fSVGDDynamicsLV(AbstractfSVGDDynamics):
    def __init__(self, state_dim, action_dim, normalizer, features: Sequence[int], prior_h=1.0, num_particles: int = 5):
        super().__init__(state_dim=state_dim, action_dim=action_dim, normalizer=normalizer, features=features,
                         num_particles=num_particles, prior_h=prior_h)

    def _control_affine_part(self, x, u):
        assert u.shape == (self.action_dim,)
        return u


if __name__ == '__main__':
    from jax.config import config

    config.update("jax_enable_x64", True)

    state_dim = 2
    control_dim = 1

    noise = 0.01
