from functools import partial
from typing import Any, Tuple, Dict

import jax.numpy as jnp
from flax.core import FrozenDict
from jax import vmap, random, jit
from jax.scipy.stats import norm

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.main.data_stats import DataStats, DataLearn
from cucrl.smoother.abstract_smoother import AbstractSmoother
from cucrl.utils.classes import NumberTrainPoints, DynamicsModel

pytree = Any


class Objectives:
    def __init__(self, smoother: AbstractSmoother, dynamics: AbstractDynamics):
        self.smoother = smoother
        self.dynamics = dynamics

    def pretraining_smoother(self, parameters, stats, data: DataLearn, data_stats: DataStats, keys,
                             num_train_points: NumberTrainPoints):
        smoother_apply = self.smoother.apply(parameters["smoother"], stats['smoother'],
                                             data.smoothing_data.ts, data.matching_data.ts,
                                             data.smoothing_data.x0s, data.matching_data.x0s,
                                             data.smoothing_data.ys, keys.episode_key, data_stats,
                                             num_train_points.smoother)

        smoother_loss, updated_states_smoother = smoother_apply.loss, smoother_apply.updated_stats
        objective = jnp.sum(smoother_loss)
        stats['smoother'] = updated_states_smoother
        return objective, stats

    def pretraining_dynamics(self, parameters, stats, data: DataLearn, data_stats: DataStats, keys,
                             num_train_points: NumberTrainPoints):
        dynamics_pretraining_loss, updated_states_dynamics = self.dynamics.loss(
            parameters["dynamics"], stats['dynamics'], data.dynamics_data.xs, data.dynamics_data.us,
            data.dynamics_data.xs_dot, data.dynamics_data.xs_dot_std, data_stats, num_train_points.dynamics,
            keys.step_key)
        objective = dynamics_pretraining_loss
        stats['dynamics'] = updated_states_dynamics
        return objective, stats

    @partial(jit, static_argnums=0)
    def joint_training(self, parameters, stats, data: DataLearn, data_stats: DataStats,
                       betas, keys, num_train_points: NumberTrainPoints) -> Tuple[jnp.array, Dict[str, FrozenDict]]:
        smoother_apply = self.smoother.apply(parameters["smoother"], stats['smoother'], data.smoothing_data.ts,
                                             data.matching_data.ts, data.smoothing_data.x0s, data.matching_data.x0s,
                                             data.smoothing_data.ys, keys.episode_key, data_stats,
                                             num_train_points.smoother)

        # Compute marginal log likelihood of the data
        mll_terms = jnp.sum(smoother_apply.loss)
        objective = mll_terms

        dynamics_model = DynamicsModel(
            params=parameters["dynamics"],
            model_stats=stats['dynamics'],
            data_stats=data_stats,
            episode=1,
            beta=betas,
            history=data.dynamics_data,
            calibration_alpha=jnp.ones(shape=(self.dynamics.state_dim,))
        )
        dynamics_der_means, dynamics_der_stds = self.dynamics.mean_and_std_eval_batch(
            dynamics_model, smoother_apply.xs_mean, data.matching_data.us)

        # Compute the distribution distance
        distribution_distance = jnp.mean(
            betas[jnp.newaxis, ...] * vv_wasserstein_2_distance(smoother_apply.xs_dot_mean, dynamics_der_means,
                                                                smoother_apply.xs_dot_var_given_x,
                                                                dynamics_der_stds ** 2))
        objective += distribution_distance

        # Compute marginal log likelihood of the vector field
        dynamics_pretraining_loss, updated_states_dynamics = self.dynamics.loss(
            parameters["dynamics"], stats['dynamics'], data.dynamics_data.xs, data.dynamics_data.us,
            data.dynamics_data.xs_dot, data.dynamics_data.xs_dot_std, data_stats, num_train_points.dynamics,
            keys.step_key)

        objective += dynamics_pretraining_loss
        stats['dynamics'] = updated_states_dynamics
        stats['smoother'] = smoother_apply.updated_stats
        return objective, stats


def pointwise_wasserstein_2_distance(smoother_mean, dynamics_mean, smoother_covariance, dynamics_covariance):
    assert smoother_mean.shape == dynamics_mean.shape == smoother_covariance.shape == dynamics_covariance.shape
    smoother_std = jnp.sqrt(jnp.clip(smoother_covariance, 0.0))
    dynamics_std = jnp.sqrt(jnp.clip(dynamics_covariance, 0.0))
    return (smoother_mean - dynamics_mean) ** 2 + (smoother_std - dynamics_std) ** 2


def wasserstein_2_distance_1d(smoother_mean, dynamics_mean, smoother_covariance, dynamics_covariance):
    assert smoother_mean.shape == () and dynamics_mean.shape == ()
    assert smoother_covariance.shape == () and dynamics_covariance.shape == ()
    smoother_std = jnp.sqrt(jnp.clip(smoother_covariance, 0.0))
    dynamics_std = jnp.sqrt(jnp.clip(dynamics_covariance, 0.0))
    return (smoother_mean - dynamics_mean) ** 2 + (smoother_std - dynamics_std) ** 2


v_wasserstein_2_distance = vmap(wasserstein_2_distance_1d, in_axes=(0, 0, 0, 0))
vv_wasserstein_2_distance = vmap(v_wasserstein_2_distance, in_axes=(0, 0, 0, 0))
wasserstein_distance_gmm_gaussian = vmap(vv_wasserstein_2_distance, in_axes=(None, 0, None, 0))


def sinkhorn_distance(smoother_means, dynamics_means, smoother_covariances, dynamics_covariances, betas, sigma=0.1):
    distance = jnp.sum(betas * jnp.sum((smoother_means - dynamics_means) ** 2, axis=0))
    d_sigma = jnp.sqrt(4 * smoother_covariances * dynamics_covariances + sigma ** 4)
    distance += jnp.sum(betas * jnp.sum(smoother_covariances + dynamics_covariances - d_sigma, axis=0))
    distance += jnp.size(smoother_means) * sigma ** 2 * (1 - jnp.log(2 * sigma ** 2))
    distance += sigma ** 2 * jnp.sum(betas * jnp.sum(jnp.log(d_sigma + sigma ** 2), axis=0))
    return 2 * distance


def kl(mean_a, mean_b, covariance_a, covariance_b, betas):
    distance = jnp.sum(betas * jnp.sum(covariance_a / covariance_b, axis=0))
    distance += jnp.sum(betas * jnp.sum((mean_a - mean_b) ** 2 / covariance_b, axis=0))
    distance += jnp.sum(betas * jnp.sum(jnp.log(covariance_b / covariance_a), axis=0))
    distance -= jnp.sum(betas * jnp.array(mean_a.shape[1] * [mean_a.shape[0]]))
    return 0.5 * distance


def forward_kl(smoother_means, dynamics_means, smoother_covariances, dynamics_covariances, betas):
    return 2 * kl(smoother_means, dynamics_means, smoother_covariances, dynamics_covariances, betas)


def backward_kl(smoother_means, dynamics_means, smoother_covariances, dynamics_covariances, betas):
    return 2 * kl(dynamics_means, smoother_means, dynamics_covariances, smoother_covariances, betas)


def sample_posterior(mean: jnp.ndarray, std: jnp.ndarray, num_samples: int, key: jnp.ndarray):
    samples = mean[jnp.newaxis, ...] + std[jnp.newaxis, ...] * random.normal(key=key, shape=(
        num_samples, mean.shape[0], mean.shape[1]))
    weights = norm.pdf(samples, loc=mean[jnp.newaxis, ...], scale=std[jnp.newaxis, ...])

    sums_over_samples = weights.sum(axis=0)
    weights = weights / sums_over_samples[jnp.newaxis, ...]
    return samples, weights


if __name__ == '__main__':
    test_key = random.PRNGKey(0)
    test_key, *subkey = random.split(test_key, 3)
    test_num_samples = 4

    means = jnp.ones((7, 2))
    stds = jnp.ones((7, 2))

    test_samples, test_weights = sample_posterior(means, stds, test_num_samples, test_key)
