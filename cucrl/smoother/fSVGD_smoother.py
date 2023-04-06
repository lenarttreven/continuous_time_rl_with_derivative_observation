from typing import Any, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax import random
from jax import vmap, jacrev
from jax.scipy.stats import norm, multivariate_normal

from cucrl.main.data_stats import Normalizer, DataStats
from cucrl.smoother.abstract_smoother import AbstractSmoother
from cucrl.utils.classes import SmootherPosterior, SmootherApply, SampledData
from cucrl.utils.helper_functions import squared_l2_norm, MLP

pytree = Any


class FSVGDEnsemble(AbstractSmoother):
    def __init__(self, state_dim: int, num_members: int, noise_stds: jnp.ndarray, normalizer: Normalizer,
                 features: Sequence[int] = (50, 50, 20), prior_h: float = 1.0, numerical_correction: float = 1e-3):
        super().__init__(state_dim, numerical_correction)
        self.normalizer = normalizer
        self.noise_stds = noise_stds
        self.num_particles = num_members
        self.features = features
        self.prior_h = prior_h
        self.model = MLP(features=self.features, output_dim=self.state_dim)
        self.stein_kernel, self.stein_kernel_derivative = self.prepare_stein_kernel()
        self.prior_kernel = self.prepare_prior_kernel(self.prior_h ** 2)

    @staticmethod
    def prepare_prior_kernel(h=1.0 ** 2):
        def k(x, y):
            return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h))

        v_k = vmap(k, in_axes=(0, None), out_axes=0)
        m_k = vmap(v_k, in_axes=(None, 0), out_axes=1)

        def kernel(fs):
            kernel_matrix = m_k(fs, fs)
            return kernel_matrix

        return kernel

    @staticmethod
    def prepare_stein_kernel(h=0.2 ** 2):
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

    def _apply_model_train(self, params, stats, t: jnp.ndarray, x: jnp.ndarray, data_stats: DataStats):
        assert t.shape == (1,) and x.shape == (self.state_dim,)
        x = self.normalizer.normalize(x, data_stats.ic_stats)
        t = self.normalizer.normalize(t, data_stats.ts_stats)
        pred, stats = self.model.apply({'params': params, **stats}, jnp.concatenate([t, x]), train=True,
                                       mutable=list(stats.keys()))
        assert pred.shape == (self.state_dim,)
        return pred, stats

    def _apply_model_eval(self, params, stats, t: jnp.ndarray, x: jnp.ndarray, data_stats: DataStats):
        assert t.shape == (1,) and x.shape == (self.state_dim,)
        x = self.normalizer.normalize(x, data_stats.ic_stats)
        t = self.normalizer.normalize(t, data_stats.ts_stats)
        pred = self.model.apply({'params': params, **stats}, jnp.concatenate([t, x]))
        assert pred.shape == (self.state_dim,)
        return pred

    def _apply_ensemble_train(self, params, stats, t, x, data_stats):
        return vmap(self._apply_model_train, in_axes=(0, 0, None, None, None))(params, stats, t, x, data_stats)

    def _apply_ensemble_train_batch(self, params, stats, ts, xs, data_stats):
        return vmap(self._apply_ensemble_train, in_axes=(None, None, 0, 0, None), out_axes=(1, 0), axis_name='batch')(
            params, stats, ts, xs, data_stats)

    def _apply_ensemble_eval(self, params, stats, t, x, data_stats):
        return vmap(self._apply_model_eval, in_axes=(0, 0, None, None, None))(params, stats, t, x, data_stats)

    def _apply_model_derivative(self, params, stats, t, x, data_stats):
        assert t.shape == (1,) and x.shape == (self.state_dim,)
        return jacrev(self._apply_model_eval, argnums=2)(params, stats, t, x, data_stats)

    def _apply_ensemble_derivative(self, params, stats, t, x, data_stats):
        return vmap(self._apply_model_derivative, in_axes=(0, 0, None, None, None))(params, stats, t, x, data_stats)

    def compute_state_mean(self, params, stats, t, x, data_stats):
        assert t.shape == (1,) and x.shape == (self.state_dim,)
        means = self._apply_ensemble_eval(params, stats, t, x, data_stats)
        assert means.shape == (self.num_particles, self.state_dim)
        return jnp.mean(means, axis=0)

    def compute_state_mean_batch(self, params, stats, ts, xs, data_stats):
        assert ts.shape[0] == xs.shape[0] and ts.shape[1] == 1 and xs.shape[1] == self.state_dim
        return vmap(self.compute_state_mean, in_axes=(None, None, 0, 0, None), axis_name='batch')(params, stats, ts, xs,
                                                                                                  data_stats)

    def compute_state_covariance(self, params, stats, t1, x1, t2, x2, data_stats):
        assert t1.shape == t2.shape == (1,) and x1.shape == x2.shape == (self.state_dim,)
        means_1 = self._apply_ensemble_eval(params, stats, t1, x1, data_stats)
        means_2 = self._apply_ensemble_eval(params, stats, t2, x2, data_stats)
        return jnp.sum((means_1 - jnp.mean(means_1, axis=0)[jnp.newaxis, ...]) * (
                means_2 - jnp.mean(means_2, axis=0)[jnp.newaxis, ...]), axis=0) / (self.num_particles - 1)

    def compute_state_covariance_batch(self, params, stats, ts1, xs1, ts2, xs2, data_stats):
        assert ts1.shape[0] == xs1.shape[0] and ts1.shape[1] == 1 and xs1.shape[1] == self.state_dim
        assert ts1.shape == ts2.shape and xs1.shape == xs2.shape
        return vmap(self.compute_state_covariance, in_axes=(None, None, 0, 0, 0, 0, None), axis_name='batch')(
            params, stats, ts1, xs1, ts2, xs2, data_stats)

    def compute_state_variance(self, params, stats, t, x, data_stats):
        assert t.shape == (1,) and x.shape == (self.state_dim,)
        return self.compute_state_covariance(params, stats, t, x, t, x, data_stats)

    def compute_state_variance_batch(self, params, stats, ts, xs, data_stats):
        assert ts.shape[0] == xs.shape[0] and ts.shape[1] == 1 and xs.shape[1] == self.state_dim
        return vmap(self.compute_state_variance, in_axes=(None, None, 0, 0, None), axis_name='batch')(
            params, stats, ts, xs, data_stats)

    def compute_derivative_mean(self, params, stats, t, x, data_stats):
        assert t.shape == (1,) and x.shape == (self.state_dim,)
        der_means = self._apply_ensemble_derivative(params, stats, t, x, data_stats)
        assert der_means.shape == (self.num_particles, self.state_dim, 1)
        der_means = der_means.reshape(self.num_particles, self.state_dim)
        return jnp.mean(der_means, axis=0)

    def compute_derivative_mean_batch(self, params, stats, ts, xs, data_stats):
        assert ts.shape[0] == xs.shape[0] and ts.shape[1] == 1 and xs.shape[1] == self.state_dim
        return vmap(self.compute_derivative_mean, in_axes=(None, None, 0, 0, None), axis_name='batch')(
            params, stats, ts, xs, data_stats)

    def compute_derivative_covariance(self, params, stats, t1, x1, t2, x2, data_stats):
        assert t1.shape == t2.shape == (1,) and x1.shape == x2.shape == (self.state_dim,)
        der_means_1 = self._apply_ensemble_derivative(params, stats, t1, x1, data_stats)
        der_means_2 = self._apply_ensemble_derivative(params, stats, t2, x2, data_stats)
        der_means_1, der_means_2 = der_means_1.reshape(-1, self.state_dim), der_means_2.reshape(-1, self.state_dim)
        return jnp.sum((der_means_1 - jnp.mean(der_means_1, axis=0)[jnp.newaxis, ...]) * (
                der_means_2 - jnp.mean(der_means_2, axis=0)[jnp.newaxis, ...]), axis=0) / (self.num_particles - 1)

    def compute_derivative_covariance_batch(self, params, stats, ts1, xs1, ts2, xs2, data_stats):
        assert ts1.shape[0] == xs1.shape[0] and ts1.shape[1] == 1 and xs1.shape[1] == self.state_dim
        assert ts1.shape == ts2.shape and xs1.shape == xs2.shape
        return vmap(self.compute_derivative_covariance, in_axes=(None, None, 0, 0, 0, 0, None), axis_name='batch')(
            params, stats, ts1, xs1, ts2, xs2, data_stats)

    def compute_derivative_variance(self, params, stats, t, x, data_stats):
        assert t.shape == (1,) and x.shape == (self.state_dim,)
        return self.compute_derivative_covariance(params, stats, t, x, t, x, data_stats)

    def compute_derivative_variance_batch(self, params, stats, ts, xs, data_stats):
        assert ts.shape[0] == xs.shape[0] and ts.shape[1] == 1 and xs.shape[1] == self.state_dim
        return vmap(self.compute_derivative_variance, in_axes=(None, None, 0, 0, None), axis_name='batch')(
            params, stats, ts, xs, data_stats)

    def compute_derivative_state_covariance(self, params, stats, t1, x1, t2, x2, data_stats):
        assert t1.shape == t2.shape == (1,) and x1.shape == x2.shape == (self.state_dim,)
        der_means_1 = self._apply_ensemble_derivative(params, stats, t1, x1, data_stats)
        der_means_1 = der_means_1.reshape(-1, self.state_dim)
        means_2 = self._apply_ensemble_eval(params, stats, t2, x2, data_stats)
        return jnp.sum((der_means_1 - jnp.mean(der_means_1, axis=0)[jnp.newaxis, ...]) * (
                means_2 - jnp.mean(means_2, axis=0)[jnp.newaxis, ...]), axis=0) / (self.num_particles - 1)

    def compute_derivative_state_covariance_batch(self, params, stats, ts1, xs1, ts2, xs2, data_stats):
        assert ts1.shape[0] == xs1.shape[0] and ts1.shape[1] == 1 and xs1.shape[1] == self.state_dim
        assert ts1.shape == ts2.shape and xs1.shape == xs2.shape
        return vmap(self.compute_derivative_state_covariance, in_axes=(None, None, 0, 0, 0, 0, None),
                    axis_name='batch')(params, stats, ts1, xs1, ts2, xs2, data_stats)

    def _neg_log_posterior(self, pred_raw: jnp.ndarray, x_stacked: jnp.ndarray, y_batch: jnp.ndarray,
                           scale: jnp.ndarray, num_train_points):
        assert pred_raw.shape == y_batch.shape == scale.shape
        nll = self._nll(pred_raw, y_batch, scale)
        neg_log_prior = - self._gp_prior_log_prob(x_stacked, pred_raw) / num_train_points
        neg_log_post = nll + neg_log_prior
        return neg_log_post

    @staticmethod
    def _nll(pred_raw: jnp.ndarray, y_batch: jnp.ndarray, scale: jnp.ndarray):
        log_prob = norm.logpdf(y_batch, loc=pred_raw, scale=scale)
        return - jnp.mean(log_prob)

    def _gp_prior_log_prob(self, x: jnp.array, y: jnp.array, eps: float = 1e-4) -> jnp.ndarray:
        # Multiple dimension outputs are handled independently per dimension
        k = self.prior_kernel(x) + eps * jnp.eye(x.shape[0])

        def evaluate_fs(fs):
            assert fs.shape == (x.shape[0],) and fs.ndim == 1
            return multivariate_normal.logpdf(fs, mean=jnp.zeros(x.shape[0]), cov=k)

        evaluate_fs_multiple_dims = vmap(evaluate_fs, in_axes=1, out_axes=0)
        evaluate_ensemble = vmap(evaluate_fs_multiple_dims, in_axes=0, out_axes=0)
        return jnp.mean(evaluate_ensemble(y))

    def _surrogate_loss(self, params: jnp.array, stats, ts, xs, obs, data_stats: DataStats) -> Tuple[float, FrozenDict]:
        assert ts.shape[1] == 1 and xs.shape[1] == self.state_dim and ts.shape[0] == xs.shape[0]
        assert xs.shape == obs.shape
        # likelihood
        f_raw, stats = self._apply_ensemble_train_batch(params, stats, ts, xs, data_stats)

        normalize = vmap(self.normalizer.normalize, in_axes=(0, None))

        xs_norm = normalize(xs, data_stats.ic_stats)
        ts_norm = normalize(ts, data_stats.ts_stats)
        ys_norm = normalize(obs, data_stats.ys_stats)

        x_batch = jnp.concatenate([ts_norm, xs_norm], axis=1)
        y_batch = jnp.repeat(ys_norm[jnp.newaxis, ...], repeats=self.num_particles, axis=0)

        scale = self.normalizer.normalize_std(self.noise_stds, data_stats.ys_stats)
        scale = jnp.repeat(scale[jnp.newaxis, ...], obs.shape[0], axis=0)
        scale = jnp.repeat(scale[jnp.newaxis, ...], self.num_particles, axis=0)

        assert x_batch.shape == (ts.shape[0], self.state_dim + 1)
        assert f_raw.shape == (self.num_particles, ts.shape[0], self.state_dim) == y_batch.shape == scale.shape
        num_train_points = ts.shape[0]
        # f_raw, stats = apply_ensemble_train(params, stats, x_batch)
        grad_post = jax.grad(self._neg_log_posterior)(f_raw, x_batch, y_batch, scale, num_train_points)

        # kernel
        k = self.stein_kernel(f_raw)
        k_x = self.stein_kernel_derivative(f_raw)
        grad_k = jnp.mean(k_x, axis=0)

        surrogate_loss = jnp.sum(f_raw * jax.lax.stop_gradient(
            jnp.einsum('ij,jkm', k, grad_post) - grad_k)) + 1e-4 * squared_l2_norm(params)
        return surrogate_loss, stats

    def apply(self, parameters: pytree, stats: FrozenDict, observation_times: jnp.array, matching_times: jnp.array,
              ic_for_observation_times: jnp.array, ic_for_matching_times: jnp.array, observations: jnp.array,
              key: jnp.ndarray, data_stats: DataStats):
        # TODO: change handling of times to have shape (-1, 1) at the input already
        observation_times = observation_times.reshape(-1, 1)
        matching_times = matching_times.reshape(-1, 1)

        assert observation_times.shape[1] == 1 and matching_times.shape[1] == 1
        assert ic_for_observation_times.shape[1] == ic_for_matching_times.shape[1] == self.state_dim
        assert ic_for_observation_times.shape == observations.shape
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
        loss, updated_stats_mean = self._surrogate_loss(parameters, stats, observation_times, ic_for_observation_times,
                                                        observations, data_stats)

        denormalize = vmap(self.normalizer.denormalize, in_axes=(0, None))
        denormalize_std = vmap(self.normalizer.denormalize_std, in_axes=(0, None))
        denormalize_smoother_der = vmap(self.normalizer.denormalize_smoother_der, in_axes=(0, None))
        denormalize_smoother_der_var = vmap(self.normalizer.denormalize_smoother_der_var, in_axes=(0, None))

        posterior_der_variances = derivative_variances - derivative_state_covariances ** 2 / state_variances

        smoothed_mean = denormalize(smoothed_mean, data_stats.ys_stats)
        state_variances = denormalize_std(jnp.sqrt(state_variances), data_stats.ys_stats) ** 2
        derivative_mean = denormalize_smoother_der(derivative_mean, data_stats.ys_stats)
        derivative_variances = denormalize_smoother_der_var(derivative_variances, data_stats.ys_stats)

        posterior_der_variances = denormalize_smoother_der_var(posterior_der_variances, data_stats.ys_stats)

        return SmootherApply(smoothed_mean, state_variances, derivative_mean, derivative_variances,
                             posterior_der_variances, loss, updated_stats_mean)

    def posterior(self, parameters: pytree, stats: FrozenDict, evaluation_times: jnp.array,
                  ic_for_evaluation_times: jnp.array, data_stats: DataStats):
        # TODO: change handling of times to have shape (-1, 1) at the input already
        evaluation_times = evaluation_times.reshape(-1, 1)

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
        denormalize = vmap(self.normalizer.denormalize, in_axes=(0, None))
        denormalize_std = vmap(self.normalizer.denormalize_std, in_axes=(0, None))
        denormalize_smoother_der = vmap(self.normalizer.denormalize_smoother_der, in_axes=(0, None))

        smoothed_mean = denormalize(smoothed_mean, data_stats.ys_stats)
        state_variances = denormalize_std(jnp.sqrt(state_variances), data_stats.ys_stats) ** 2
        derivative_mean = denormalize_smoother_der(derivative_mean, data_stats.ys_stats)
        derivative_variances = denormalize_smoother_der(jnp.sqrt(derivative_variances), data_stats.ys_stats) ** 2

        return SmootherPosterior(smoothed_mean, state_variances, derivative_mean, derivative_variances)

    def sample_vector_field_data(self, params, stats, observation_times, ic_for_observation_times,
                                 data_stats: DataStats):
        observation_times = observation_times.reshape(-1, 1)
        assert ic_for_observation_times.shape[1] == self.state_dim
        assert observation_times.shape[1] == 1
        smoothed_state_mean = self.compute_state_mean_batch(params, stats, observation_times, ic_for_observation_times,
                                                            data_stats)
        xs_var = self.compute_state_variance_batch(params, stats, observation_times, ic_for_observation_times,
                                                   data_stats)

        xs_dot_mean = self.compute_derivative_mean_batch(params, stats, observation_times, ic_for_observation_times,
                                                         data_stats)
        xs_dot_var = self.compute_derivative_variance_batch(params, stats, observation_times, ic_for_observation_times,
                                                            data_stats)

        xs_dot_vs_xs_cov = self.compute_derivative_state_covariance_batch(params, stats, observation_times,
                                                                          ic_for_observation_times, observation_times,
                                                                          ic_for_observation_times, data_stats)
        xs_dot_given_xs_var = xs_dot_var - xs_dot_vs_xs_cov ** 2 / xs_var
        xs_dot_given_xs_std = jnp.sqrt(xs_dot_given_xs_var)

        smoothed_state_mean = vmap(self.normalizer.denormalize, in_axes=(0, None))(smoothed_state_mean,
                                                                                   data_stats.ys_stats)
        xs_dot_mean = vmap(self.normalizer.denormalize_smoother_der, in_axes=(0, None))(xs_dot_mean,
                                                                                        data_stats.ys_stats)
        xs_dot_given_xs_std = vmap(self.normalizer.denormalize_smoother_der, in_axes=(0, None))(xs_dot_given_xs_std,
                                                                                                data_stats.ys_stats)
        return SampledData(smoothed_state_mean, xs_dot_mean, xs_dot_given_xs_std)

    def regularization(self, parameters, weights):
        # return 0.01 * squared_l2_norm(parameters)
        return 0.0

    def initialize_parameters(self, key):
        key, *subkeys = random.split(key, self.num_particles + 1)
        return vmap(self._initialize_parameters, in_axes=0)(jnp.stack(subkeys))

    def _initialize_parameters(self, key):
        variables = self.model.init(key, jnp.ones(shape=(self.state_dim + 1)))
        state, params = variables.pop('params')
        del variables  # Delete variables to avoid wasting resources
        return params, state


def training_test():
    key = random.PRNGKey(0)
    key, subkey = random.split(key)

    test_state_dim = 2
    test_num_samples = 17
    test_num_members = 10


def general_test():
    test_key = random.PRNGKey(0)
    test_key, test_subkey = random.split(test_key)

    test_state_dim = 3
    test_num_samples = 17
    test_num_members = 10

    test_gp = FSVGDEnsemble(state_dim=test_state_dim, num_members=test_num_members, features=[30, 20])
    test_params, test_stats = test_gp.initialize_parameters(test_subkey)

    test_t1 = jnp.array([1.0])
    test_x1 = jnp.array([1.0, 2.0, 3.0])
    test_t2 = jnp.array([-1.0])
    test_x2 = jnp.array([1.0, -2.0, 3.0])
    test_ts = random.normal(test_key, shape=(test_num_samples, 1))
    test_xs = random.normal(test_key, shape=(test_num_samples, test_state_dim))
    test_obs = random.normal(test_key, shape=(test_num_samples, test_state_dim))

    out = test_gp.compute_state_mean(test_params, test_stats, test_t1, test_x1)
    assert out.shape == (test_state_dim,)
    out = test_gp.compute_state_variance(test_params, test_stats, test_t1, test_x1)
    assert out.shape == (test_state_dim,)
    out = test_gp.compute_derivative_mean(test_params, test_stats, test_t1, test_x1)
    assert out.shape == (test_state_dim,)
    out = test_gp.compute_derivative_variance(test_params, test_stats, test_t1, test_x1)
    assert out.shape == (test_state_dim,)
    test_loss = test_gp._surrogate_loss(test_params, test_stats, test_ts, test_xs, test_obs)
    assert test_loss[0].shape == ()


if __name__ == '__main__':
    general_test()
