import time
from functools import partial
from typing import List, Sequence, Optional, Union, Dict, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn
from flax.core import FrozenDict
from jax import random, vmap, grad, jit
from jax.scipy.stats import norm, multivariate_normal

from cucrl.main.data_stats import Normalizer, Stats


class DataRepr(NamedTuple):
    xs: jax.Array
    ys: jax.Array


class DataStatsFSVGD(NamedTuple):
    input_stats: Stats
    output_stats: Stats


class FSVGD:
    def __init__(self, input_dim: int, output_dim: int, bandwidth_prior: float, bandwidth_svgd: float,
                 features: List[int], num_particles: int, domain_l: float, domain_u: float,
                 num_measurement_points: int = 0, normalizer: Normalizer = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_particles = num_particles
        self.num_measurement_points = num_measurement_points
        self.bandwidth_prior = bandwidth_prior
        self.bandwidth_svgd = bandwidth_svgd
        self.features = features
        self.prior_kernel = self.prepare_prior_kernel(h=self.bandwidth_prior)
        self.svgd_kernel, self.svgd_kernel_der = self.prepare_svgd_kernel(h=self.bandwidth_svgd)
        self.model = MLP(features=features, output_dim=self.output_dim)
        self.key = random.PRNGKey(0)
        self.domain_l, self.domain_u = domain_l, domain_u
        self.tx = optax.adam(learning_rate=optax.piecewise_constant_schedule(1e-2, {1000: 1e-1}))
        self.normalizer = normalizer

    def _apply_train(self, params, stats, x, data_stats: DataStatsFSVGD):
        assert x.shape == (self.input_dim,)
        x = self.normalizer.normalize(x, data_stats.input_stats)
        return self.model.apply({'params': params, **stats}, x, mutable=stats.keys(), train=True)

    def apply_eval(self, params, stats, x, data_stats: DataStatsFSVGD):
        assert x.shape == (self.input_dim,)
        x = self.normalizer.normalize(x, data_stats.input_stats)
        out = self.model.apply({'params': params, **stats}, x)
        return self.normalizer.denormalize(out, data_stats.output_stats)

    @staticmethod
    def prepare_prior_kernel(h=1.0):
        def k(x, y):
            return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h ** 2))

        v_k = vmap(k, in_axes=(0, None), out_axes=0)
        m_k = vmap(v_k, in_axes=(None, 0), out_axes=1)

        def kernel(fs):
            kernel_matrix = m_k(fs, fs)
            return kernel_matrix

        return kernel

    @staticmethod
    def prepare_svgd_kernel(h=1.0):
        def k(x, y):
            return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h ** 2))

        v_k = vmap(k, in_axes=(0, None), out_axes=0)
        m_k = vmap(v_k, in_axes=(None, 0), out_axes=1)

        k_x = grad(k, argnums=0)

        v_k_der = vmap(k_x, in_axes=(0, None), out_axes=0)
        m_k_der = vmap(v_k_der, in_axes=(None, 0), out_axes=1)

        def kernel(fs):
            kernel_matrix = m_k(fs, fs)
            return kernel_matrix

        def kernel_derivative(fs):
            return m_k_der(fs, fs)

        return kernel, kernel_derivative

    ## This is an old implementation of the prior log prob, check why it is not working
    # def _gp_prior_log_prob(self, x: jnp.array, y: jnp.array, data_stats, eps: float = 1e-4, ) -> jax.Array:
    #     # Multiple dimension outputs are handled independently per dimension
    #     x = vmap(self.normalizer.normalize, in_axes=(0, None))(x, data_stats.input_stats)
    #     k = self.prior_kernel(x) + eps * jnp.eye(x.shape[0])
    #
    #     def evaluate_fs(fs):
    #         assert fs.shape == (x.shape[0],) and fs.ndim == 1
    #         return multivariate_normal.logpdf(fs, mean=jnp.zeros(x.shape[0]), cov=k)
    #
    #     evaluate_fs_multiple_dims = vmap(evaluate_fs, in_axes=1, out_axes=0)
    #     evaluate_ensemble = vmap(evaluate_fs_multiple_dims, in_axes=0, out_axes=0)
    #     # Next line can be potentially sum instead of mean!!
    #     return jnp.mean(evaluate_ensemble(y))

    def _gp_prior_log_prob(self, x: jnp.array, y: jnp.array, data_stats, eps: float = 1e-4, ) -> jnp.ndarray:
        x = vmap(self.normalizer.normalize, in_axes=(0, None))(x, data_stats.input_stats)
        k = self.prior_kernel(x) + eps * jnp.eye(x.shape[0])
        dist = tfd.MultivariateNormalFullCovariance(jnp.zeros(x.shape[0]), k)
        return jnp.mean(jnp.sum(dist.log_prob(jnp.swapaxes(y, -1, -2)), axis=-1)) / x.shape[0]

    def _nll(self, pred_raw: jax.Array, y_batch: jax.Array, train_data_till_idx: int, data_std_batch: jax.Array):
        assert y_batch.shape == data_std_batch.shape
        log_prob = norm.logpdf(y_batch[jnp.newaxis, :], loc=pred_raw[:, :train_data_till_idx, :],
                               scale=data_std_batch[jnp.newaxis, :])
        return - jnp.mean(log_prob)

    def _neg_log_posterior(self, pred_raw: jax.Array, x_stacked: jax.Array, y_batch: jax.Array,
                           train_data_till_idx: int, data_std_batch: jax.Array, data_stats: DataStatsFSVGD,
                           num_train_points: Union[float, int]):
        nll = self._nll(pred_raw, y_batch, train_data_till_idx, data_std_batch)
        neg_log_prior = - self._gp_prior_log_prob(x_stacked, pred_raw, data_stats) / num_train_points
        neg_log_post = nll + neg_log_prior
        return neg_log_post

    def _sample_measurement_points(self, key: jax.random.PRNGKey, num_points: int = 10) -> jax.Array:
        x_domain = jax.random.uniform(key, shape=(num_points, self.input_dim),
                                      minval=self.domain_l, maxval=self.domain_u)
        assert x_domain.shape == (num_points, self.input_dim)
        return x_domain

    def loss(self, param_vec_stack: jnp.array, stats, data: DataRepr, data_stats: DataStatsFSVGD,
             ys_std: jax.Array, num_train_points: int, key: jax.random.PRNGKey) -> [jax.Array, Dict]:

        # combine the training data batch with a batch of sampled measurement points
        train_batch_size = data.xs.shape[0]
        x_domain = self._sample_measurement_points(key, num_points=self.num_measurement_points)
        x_stacked = jnp.concatenate([data.xs, x_domain], axis=0)

        # likelihood
        apply_ensemble_one = vmap(self._apply_train, in_axes=(0, 0, None, None), out_axes=0)
        apply_ensemble = vmap(apply_ensemble_one, in_axes=(None, None, 0, None), out_axes=(1, None), axis_name='batch')
        f_raw, new_stats = apply_ensemble(param_vec_stack, stats, x_stacked, data_stats)

        ys_batch_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(data.ys, data_stats.output_stats)
        ys_std_norm = vmap(self.normalizer.normalize_std, in_axes=(0, None))(ys_std, data_stats.output_stats)

        log_likelihood, grad_post = jax.value_and_grad(self._neg_log_posterior)(f_raw, x_stacked, ys_batch_norm,
                                                                                train_batch_size, ys_std_norm,
                                                                                data_stats, num_train_points)

        # kernel
        k = self.svgd_kernel(f_raw)
        k_x = self.svgd_kernel_der(f_raw)
        grad_k = jnp.mean(k_x, axis=0)

        surrogate_loss = jnp.sum(f_raw * jax.lax.stop_gradient(
            jnp.einsum('ij,jkm', k, grad_post) - grad_k))
        return surrogate_loss, new_stats

    @partial(jit, static_argnums=0)
    def eval_ll(self, param_vec_stack: jnp.array, stats, data: DataRepr, data_stats: DataStatsFSVGD,
                data_std: jax.Array, num_train_points) -> jax.Array:
        apply_ensemble_one = vmap(self.apply_eval, in_axes=(0, 0, None, None), out_axes=0)
        apply_ensemble = vmap(apply_ensemble_one, in_axes=(None, None, 0, None), out_axes=1, axis_name='batch')
        f_raw = apply_ensemble(param_vec_stack, stats, data.xs, data_stats)

        nll = self._nll(f_raw, data.ys, data.xs.shape[0], data_std)
        neg_log_prior = - self._gp_prior_log_prob(data.xs, f_raw, data_stats) / num_train_points
        neg_log_post = nll + neg_log_prior
        return nll, neg_log_prior, neg_log_post

    @partial(jit, static_argnums=0)
    def _step_jit(self, opt_state: optax.OptState, param_vec_stack: jnp.array, stats, data: DataRepr,
                  data_stats, data_std, key: jax.random.PRNGKey, num_train_points: Union[float, int]):
        (loss, stats), grads = jax.value_and_grad(self.loss, has_aux=True)(
            param_vec_stack, stats, data, data_stats, data_std, num_train_points, key)
        updates, opt_state = self.tx.update(grads, opt_state)
        param_vec_stack = optax.apply_updates(param_vec_stack, updates)
        return opt_state, param_vec_stack, stats, loss

    def init_params(self, key):
        variables = self.model.init(key, jnp.ones(shape=(self.input_dim,)))
        if 'params' in variables:
            stats, params = variables.pop('params')
        else:
            stats, params = variables, FrozenDict({})
        del variables  # Delete variables to avoid wasting resources
        return params, stats

    def fit_model(self, dataset: DataRepr, num_epochs, data_stats: DataStatsFSVGD, data_std: jax.Array, batch_size):
        self.key, key, *subkeys = random.split(self.key, self.num_particles + 2)
        params, stats = vmap(self.init_params)(jnp.stack(subkeys))
        opt_state = self.tx.init(params)

        num_train_points = dataset.ys.shape[0]
        train_loader = self._create_data_loader(dataset, data_std, batch_size=batch_size)

        for step, (data_batch, data_std_batch) in enumerate(train_loader, 1):
            key, subkey = random.split(key)
            opt_state, params, stats, loss = self._step_jit(opt_state, params, stats, data_batch,
                                                            data_stats, data_std_batch, subkey,
                                                            num_train_points)

            if step >= num_epochs:
                break

            if step % 100 == 0 or step == 1:
                nll, nll_prior, nll_posterior = self.eval_ll(params, stats, data_batch, data_stats, data_std_batch,
                                                             num_train_points)

                print(f'Step {step}, nll_posterior: {nll_posterior}, nll_prior: {nll_prior}, nll: {nll}')

        return params, stats

    def _create_data_loader(self, dataset: DataRepr, data_std: jax.Array,
                            batch_size: int = 64, shuffle: bool = True,
                            infinite: bool = True) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((dataset, data_std))
        if shuffle:
            seed = int(jax.random.randint(self.key, (1,), 0, 10 ** 8))
            ds = ds.shuffle(batch_size * 4, seed=seed, reshuffle_each_iteration=True)
        if infinite:
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = tfds.as_numpy(ds)
        return ds

    def calculate_calibration_alpha(self, params, stats, xs, ys, ys_stds, ps, data_stats: DataStatsFSVGD) -> jax.Array:
        # We flip so that we rather take more uncertainty model than less
        test_alpha = jnp.flip(jnp.linspace(0, 10, 100)[1:])
        test_alphas = jnp.repeat(test_alpha[..., jnp.newaxis], repeats=self.output_dim, axis=1)
        errors = vmap(self._calibration_errors, in_axes=(None, None, None, None, None, None, None, 0))(
            params, stats, xs, ys, ys_stds, ps, data_stats, test_alphas)
        indices = jnp.argmin(errors, axis=0)
        best_alpha = test_alpha[indices]
        assert best_alpha.shape == (self.output_dim,)
        return best_alpha

    def _calibration_errors(self, params, stats, xs, ys, ys_stds, ps, data_stats: DataStatsFSVGD, alpha) -> jax.Array:
        ps_hat = self.calculate_calibration_score(params, stats, xs, ys, ys_stds, ps, data_stats, alpha)
        ps = jnp.repeat(ps[..., jnp.newaxis], repeats=self.output_dim, axis=1)
        return jnp.mean((ps - ps_hat) ** 2, axis=0)

    def calculate_calibration_score(self, params, stats, xs, ys, ys_stds, ps, data_stats: DataStatsFSVGD, alpha):
        assert alpha.shape == (self.output_dim,)

        def calculate_score(x, y, y_std):
            assert x.shape == (self.input_dim,) and y.shape == y_std.shape == (self.output_dim,)
            preds = vmap(self.apply_eval, in_axes=(0, 0, None, None), out_axes=0)(params, stats, x, data_stats)
            means, stds = preds.mean(axis=0), preds.std(axis=0)
            assert stds.shape == (self.output_dim,)
            cdfs = vmap(norm.cdf)(y, means, stds * alpha + y_std)

            def check_cdf(cdf):
                assert cdf.shape == ()
                return cdf <= ps

            return vmap(check_cdf, out_axes=1)(cdfs)

        cdfs = vmap(calculate_score)(xs, ys, ys_stds)
        return jnp.mean(cdfs, axis=0)


class MLP(nn.Module):
    features: Sequence[int]
    output_dim: Optional[int]

    @nn.compact
    def __call__(self, x, train: bool = False):
        for feat in self.features:
            x = nn.Dense(features=feat)(x)
            x = nn.swish(x)
        if self.output_dim is not None:
            x = nn.Dense(features=self.output_dim)(x)
        return x


if __name__ == '__main__':
    from cucrl.utils.helper_functions import AngleLayerDynamics

    key = random.PRNGKey(0)
    input_dim = 1
    output_dim = 2

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 200).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
    ys = ys + noise_level * random.normal(key=random.PRNGKey(0), shape=ys.shape)
    data_std = noise_level * jnp.ones(shape=ys.shape)
    data_stats = DataStatsFSVGD(input_stats=Stats(mean=jnp.mean(xs, axis=0), std=jnp.std(xs, axis=0)),
                                output_stats=Stats(mean=jnp.mean(ys, axis=0), std=jnp.std(ys, axis=0)))

    angle_layer = AngleLayerDynamics(state_dim=input_dim, control_dim=0, angles_dim=[],
                                     state_scaling=jnp.eye(input_dim), )
    normalizer = Normalizer(state_dim=input_dim, action_dim=output_dim, angle_layer=angle_layer)

    num_particles = 10
    model = FSVGD(input_dim=input_dim, output_dim=output_dim, bandwidth_prior=0.5, features=[64, 64, 64, 64],
                  bandwidth_svgd=0.3, num_particles=num_particles, domain_l=d_l,
                  domain_u=d_u, num_measurement_points=20, normalizer=normalizer)

    train_data = DataRepr(xs=xs, ys=ys)
    start_time = time.time()

    model_params, model_stats = model.fit_model(dataset=train_data, num_epochs=4000, data_stats=data_stats,
                                                data_std=data_std, batch_size=32)
    print(f'Training time: {time.time() - start_time:.2f} seconds')

    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)

    test_ys_noisy = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1) + noise_level * random.normal(
        key=random.PRNGKey(0), shape=test_ys.shape)

    test_stds = noise_level * jnp.ones(shape=test_ys.shape)
    num_ps = 10
    ps = jnp.linspace(0, 1, num_ps + 1)[1:]
    alpha_best = model.calculate_calibration_alpha(model_params, model_stats, test_xs, test_ys, test_ys_noisy, ps,
                                                   data_stats)

    test_ps = model.calculate_calibration_score(model_params, model_stats, test_xs, test_ys, test_ys_noisy, ps,
                                                data_stats, alpha_best)
    print(test_ps.shape)
    print('Test ps: ', test_ps)
    print('Target ps: ', ps.reshape(-1, 1))

    apply_ens = vmap(model.apply_eval, in_axes=(None, None, 0, None))
    preds = vmap(apply_ens, in_axes=(0, 0, None, None))(model_params, model_stats, test_xs, data_stats)

    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        for i in range(num_particles):
            plt.plot(test_xs, preds[i, :, j], label='NN prediction', color='black', alpha=0.3)
        plt.plot(test_xs, jnp.mean(preds[..., j], axis=0), label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (jnp.mean(preds[..., j], axis=0) - 2 * jnp.std(preds[..., j], axis=0)).reshape(-1),
                         (jnp.mean(preds[..., j], axis=0) + 2 * jnp.std(preds[..., j], axis=0)).reshape(-1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()

    for j in range(output_dim):
        # plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        for i in range(num_particles):
            plt.plot(test_xs, preds[i, :, j], label='NN prediction', color='black', alpha=0.3)
        plt.plot(test_xs, jnp.mean(preds[..., j], axis=0), label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (jnp.mean(preds[..., j], axis=0) - 2 * jnp.std(preds[..., j], axis=0)).reshape(-1),
                         (jnp.mean(preds[..., j], axis=0) + 2 * jnp.std(preds[..., j], axis=0)).reshape(-1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()

    for j in range(output_dim):
        # plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        for i in range(num_particles):
            plt.plot(test_xs, preds[i, :, j], label='NN prediction', color='black', alpha=0.3)
        plt.plot(test_xs, jnp.mean(preds[..., j], axis=0), label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (jnp.mean(preds[..., j], axis=0) - 2 * alpha_best[j] * jnp.std(preds[..., j], axis=0)).reshape(
                             -1),
                         (jnp.mean(preds[..., j], axis=0) + 2 * alpha_best[j] * jnp.std(preds[..., j], axis=0)).reshape(
                             -1),
                         label=r'$2\sigma$', alpha=0.3, color='blue')
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()
