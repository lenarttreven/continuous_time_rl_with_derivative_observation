from typing import Any, List

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax import random
from jax import vmap, jacrev

from cucrl.main.data_stats import Normalizer, DataStats
from cucrl.smoother.abstract_smoother import AbstractSmoother
from cucrl.utils.classes import SmootherPosterior, SmootherApply, SampledData
from cucrl.utils.fSVGD import FSVGD, DataStatsFSVGD, DataRepr

pytree = Any


class FSVGDTimeOnly(FSVGD, AbstractSmoother):
    def __init__(self, state_dim: int, num_particles: int, noise_stds: jnp.ndarray, normalizer: Normalizer,
                 features: List[int] = (50, 50, 20), bandwidth_prior: float = 1.0, bandwidth_svgd: float = 0.2,
                 numerical_correction: float = 1e-3):
        self.noise_stds = noise_stds
        # Todo: incorporate domain_l and domain_u
        FSVGD.__init__(self, input_dim=1, output_dim=state_dim, bandwidth_prior=bandwidth_prior,
                       bandwidth_svgd=bandwidth_svgd, features=features, num_particles=num_particles, domain_l=0.0,
                       domain_u=1.0, normalizer=normalizer)
        AbstractSmoother.__init__(self, state_dim=state_dim, numerical_correction=numerical_correction)

    def _apply_model_eval(self, params, stats, t: jnp.ndarray, x: jnp.ndarray, data_stats: DataStats):
        data_stats = DataStatsFSVGD(input_stats=data_stats.ts_stats, output_stats=data_stats.ys_stats)
        return self.apply_eval(params, stats, t, data_stats)

    def _apply_ensemble_eval(self, params, stats, t, x, data_stats):
        return vmap(self._apply_model_eval, in_axes=(0, 0, None, None, None))(params, stats, t, x, data_stats)

    def _apply_model_derivative(self, params, stats, t, x, data_stats):
        assert t.shape == (1,) and x.shape == (self.state_dim,)
        return jacrev(self._apply_model_eval, argnums=2)(params, stats, t, x, data_stats)

    def _apply_ensemble_derivative(self, params, stats, t, x, data_stats):
        return vmap(self._apply_model_derivative, in_axes=(0, 0, None, None, None))(params, stats, t, x, data_stats)

    def compute_state_mean(self, params, stats, t, x, data_stats):
        means = self._apply_ensemble_eval(params, stats, t, x, data_stats)
        assert means.shape == (self.num_particles, self.state_dim)
        return jnp.mean(means, axis=0)

    def compute_state_mean_batch(self, params, stats, ts, xs, data_stats):
        return vmap(self.compute_state_mean, in_axes=(None, None, 0, 0, None), axis_name='batch')(params, stats, ts, xs,
                                                                                                  data_stats)

    def compute_state_covariance(self, params, stats, t1, x1, t2, x2, data_stats):
        means_1 = self._apply_ensemble_eval(params, stats, t1, x1, data_stats)
        means_2 = self._apply_ensemble_eval(params, stats, t2, x2, data_stats)
        return jnp.sum((means_1 - jnp.mean(means_1, axis=0)[jnp.newaxis, ...]) * (
                means_2 - jnp.mean(means_2, axis=0)[jnp.newaxis, ...]), axis=0) / (self.num_particles - 1)

    def compute_state_covariance_batch(self, params, stats, ts1, xs1, ts2, xs2, data_stats):
        return vmap(self.compute_state_covariance, in_axes=(None, None, 0, 0, 0, 0, None), axis_name='batch')(
            params, stats, ts1, xs1, ts2, xs2, data_stats)

    def compute_state_variance(self, params, stats, t, x, data_stats):
        return self.compute_state_covariance(params, stats, t, x, t, x, data_stats)

    def compute_state_variance_batch(self, params, stats, ts, xs, data_stats):
        return vmap(self.compute_state_variance, in_axes=(None, None, 0, 0, None), axis_name='batch')(
            params, stats, ts, xs, data_stats)

    def compute_derivative_mean(self, params, stats, t, x, data_stats):
        der_means = self._apply_ensemble_derivative(params, stats, t, x, data_stats)
        assert der_means.shape == (self.num_particles, self.state_dim, 1)
        der_means = der_means.reshape(self.num_particles, self.state_dim)
        return jnp.mean(der_means, axis=0)

    def compute_derivative_mean_batch(self, params, stats, ts, xs, data_stats):
        return vmap(self.compute_derivative_mean, in_axes=(None, None, 0, 0, None), axis_name='batch')(
            params, stats, ts, xs, data_stats)

    def compute_derivative_covariance(self, params, stats, t1, x1, t2, x2, data_stats):
        der_means_1 = self._apply_ensemble_derivative(params, stats, t1, x1, data_stats)
        der_means_2 = self._apply_ensemble_derivative(params, stats, t2, x2, data_stats)
        der_means_1, der_means_2 = der_means_1.reshape(-1, self.state_dim), der_means_2.reshape(-1, self.state_dim)
        return jnp.sum((der_means_1 - jnp.mean(der_means_1, axis=0)[jnp.newaxis, ...]) * (
                der_means_2 - jnp.mean(der_means_2, axis=0)[jnp.newaxis, ...]), axis=0) / (self.num_particles - 1)

    def compute_derivative_covariance_batch(self, params, stats, ts1, xs1, ts2, xs2, data_stats):
        return vmap(self.compute_derivative_covariance, in_axes=(None, None, 0, 0, 0, 0, None), axis_name='batch')(
            params, stats, ts1, xs1, ts2, xs2, data_stats)

    def compute_derivative_variance(self, params, stats, t, x, data_stats):
        return self.compute_derivative_covariance(params, stats, t, x, t, x, data_stats)

    def compute_derivative_variance_batch(self, params, stats, ts, xs, data_stats):
        return vmap(self.compute_derivative_variance, in_axes=(None, None, 0, 0, None), axis_name='batch')(
            params, stats, ts, xs, data_stats)

    def compute_derivative_state_covariance(self, params, stats, t1, x1, t2, x2, data_stats):
        der_means_1 = self._apply_ensemble_derivative(params, stats, t1, x1, data_stats)
        der_means_1 = der_means_1.reshape(-1, self.state_dim)
        means_2 = self._apply_ensemble_eval(params, stats, t2, x2, data_stats)
        return jnp.sum((der_means_1 - jnp.mean(der_means_1, axis=0)[jnp.newaxis, ...]) * (
                means_2 - jnp.mean(means_2, axis=0)[jnp.newaxis, ...]), axis=0) / (self.num_particles - 1)

    def compute_derivative_state_covariance_batch(self, params, stats, ts1, xs1, ts2, xs2, data_stats):
        assert ts1.shape == ts2.shape and xs1.shape == xs2.shape
        return vmap(self.compute_derivative_state_covariance, in_axes=(None, None, 0, 0, 0, 0, None),
                    axis_name='batch')(params, stats, ts1, xs1, ts2, xs2, data_stats)

    def apply(self, parameters: pytree, stats: FrozenDict, observation_times: jnp.array, matching_times: jnp.array,
              ic_for_observation_times: jnp.array, ic_for_matching_times: jnp.array, observations: jnp.array,
              key: jnp.ndarray, data_stats: DataStats, num_train_points: int):
        assert observation_times.shape[1] == 1 and matching_times.shape[1] == 1
        assert ic_for_observation_times.shape[1] == ic_for_matching_times.shape[1] == self.state_dim
        assert ic_for_observation_times.shape == observations.shape

        data = DataRepr(xs=observation_times, ys=observations)
        data_stats_loss = DataStatsFSVGD(input_stats=data_stats.ts_stats, output_stats=data_stats.ys_stats)
        data_std = jnp.repeat(self.noise_stds[jnp.newaxis, ...], repeats=observation_times.shape[0], axis=0)
        loss, updated_stats_mean = self.loss(parameters, stats, data, data_stats_loss, data_std,
                                             num_train_points, key)

        smoothed_mean = self.compute_state_mean_batch(parameters, stats, matching_times, ic_for_matching_times,
                                                      data_stats)
        state_variances = self.compute_state_variance_batch(parameters, stats, matching_times, ic_for_matching_times,
                                                            data_stats)
        derivative_mean = self.compute_derivative_mean_batch(parameters, stats, matching_times, ic_for_matching_times,
                                                             data_stats)
        derivative_variances = self.compute_derivative_variance_batch(parameters, stats, matching_times,
                                                                      ic_for_matching_times, data_stats)
        derivative_state_covariances = self.compute_derivative_state_covariance_batch(parameters, stats,
                                                                                      matching_times,
                                                                                      ic_for_matching_times,
                                                                                      matching_times,
                                                                                      ic_for_matching_times, data_stats)

        posterior_der_variances = derivative_variances - derivative_state_covariances ** 2 / state_variances
        return SmootherApply(smoothed_mean, state_variances, derivative_mean, derivative_variances,
                             posterior_der_variances, loss, updated_stats_mean)

    def posterior(self, parameters: pytree, stats: FrozenDict, evaluation_times: jnp.array,
                  ic_for_evaluation_times: jnp.array, observation_times, ic_for_observation_times, observations,
                  data_stats: DataStats):
        assert evaluation_times.shape[1] == 1
        assert ic_for_evaluation_times.shape[1] == self.state_dim
        smoothed_mean = self.compute_state_mean_batch(parameters, stats, evaluation_times, ic_for_evaluation_times,
                                                      data_stats)
        state_variances = self.compute_state_variance_batch(parameters, stats, evaluation_times,
                                                            ic_for_evaluation_times, data_stats)
        derivative_mean = self.compute_derivative_mean_batch(parameters, stats, evaluation_times,
                                                             ic_for_evaluation_times, data_stats)
        derivative_variances = self.compute_derivative_variance_batch(parameters, stats, evaluation_times,
                                                                      ic_for_evaluation_times, data_stats)
        return SmootherPosterior(smoothed_mean, state_variances, derivative_mean, derivative_variances)

    def sample_vector_field_data(self, params, stats, observation_times, observations, ic_for_observation_times,
                                 data_stats: DataStats, key: jax.Array):
        observation_times = observation_times.reshape(-1, 1)
        assert ic_for_observation_times.shape[1] == self.state_dim
        assert observation_times.shape[1] == 1
        xs_mean = self.compute_state_mean_batch(params, stats, observation_times, ic_for_observation_times,
                                                data_stats)
        xs_var = self.compute_state_variance_batch(params, stats, observation_times, ic_for_observation_times,
                                                   data_stats)
        xs_dot_mean = self.compute_derivative_mean_batch(params, stats, observation_times, ic_for_observation_times,
                                                         data_stats)
        xs_dot_var = self.compute_derivative_variance_batch(params, stats, observation_times, ic_for_observation_times,
                                                            data_stats)
        xs_dot_vs_xs_cov = self.compute_derivative_state_covariance_batch(params, stats, observation_times,
                                                                          ic_for_observation_times,
                                                                          observation_times,
                                                                          ic_for_observation_times, data_stats)

        def sample_from_2d_normal(key, mean_1, var_1, mean_2, var_2, cov):
            assert mean_1.shape == var_1.shape == mean_2.shape == var_2.shape == cov.shape == ()
            mean = jnp.array([mean_1, mean_2])
            cov_matrix = jnp.array([[var_1, cov], [cov, var_2]])
            sample = random.multivariate_normal(key=key, mean=mean, cov=cov_matrix)
            return sample

        keys = random.split(key, xs_mean.size).reshape(*xs_mean.shape, 2)
        sample = vmap(vmap(sample_from_2d_normal))(keys, xs_mean, xs_var, xs_dot_mean, xs_dot_var, xs_dot_vs_xs_cov)

        sample_xs = sample[..., 0]
        sample_xs_dot = sample[..., 1]
        variance_estimate = xs_var + xs_dot_var - 2 * xs_dot_vs_xs_cov

        return SampledData(sample_xs, sample_xs_dot, jnp.sqrt(variance_estimate))

    def initialize_parameters(self, key):
        keys = random.split(key, self.num_particles)
        return vmap(self.init_params)(keys)


if __name__ == '__main__':
    pass
