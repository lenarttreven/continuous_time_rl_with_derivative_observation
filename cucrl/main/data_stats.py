from functools import partial
from typing import NamedTuple, Optional

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import numpy as jnp
from jax import vmap, random

from cucrl.main.config import BatchSize
from cucrl.utils.helper_functions import AngleLayerDynamics


class Stats(NamedTuple):
    mean: jnp.ndarray
    std: jnp.ndarray


class DataStats(NamedTuple):
    ic_stats: Stats
    ts_stats: Stats
    ys_stats: Stats
    xs_stats: Optional[Stats] = None
    us_stats: Optional[Stats] = None
    dot_xs_stats: Optional[Stats] = None
    xs_after_angle_layer: Optional[Stats] = None


class DynamicsData(NamedTuple):
    xs: jnp.ndarray
    us: jnp.ndarray
    xs_dot: jnp.ndarray
    xs_dot_std: jnp.ndarray


class SmoothingData(NamedTuple):
    ts: jnp.ndarray
    x0s: jnp.ndarray
    ys: jnp.ndarray
    us: jnp.ndarray


class MatchingData(NamedTuple):
    ts: jnp.ndarray
    x0s: jnp.ndarray
    us: jnp.ndarray


class DataLearn(NamedTuple):
    smoothing_data: SmoothingData
    matching_data: MatchingData | None = None
    dynamics_data: DynamicsData | None = None


class Normalizer:
    def __init__(self, state_dim: int, action_dim: int, angle_layer: AngleLayerDynamics,
                 tracking_c: None | jnp.ndarray = None, num_correction=1e-6):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_correction = num_correction
        self.angle_layer = angle_layer
        if tracking_c is None:
            tracking_c = jnp.eye(self.state_dim)
        self.tracking_c = tracking_c
        self.y_dim = self.tracking_c.shape[0]

    def compute_stats(self, data: DataLearn):
        if data.dynamics_data is not None:
            state_after_angle_layer = vmap(self.angle_layer.angle_layer)(data.dynamics_data.xs)
            return DataStats(ic_stats=self.normalize_stats(data.smoothing_data.x0s),
                             ts_stats=self.normalize_stats(data.smoothing_data.ts),
                             ys_stats=self.normalize_stats(data.smoothing_data.ys),
                             xs_stats=self.normalize_stats(data.dynamics_data.xs),
                             us_stats=self.normalize_stats(data.dynamics_data.us),
                             dot_xs_stats=self.normalize_stats(data.dynamics_data.xs_dot),
                             xs_after_angle_layer=self.normalize_stats(state_after_angle_layer))
        else:
            return DataStats(self.normalize_stats(data.smoothing_data.x0s),
                             self.normalize_stats(data.smoothing_data.ts), self.normalize_stats(data.smoothing_data.ys))

    @partial(jax.jit, static_argnums=0)
    def normalize_c_scale(self, to_normalize: jnp.ndarray, stats: Stats):
        assert to_normalize.ndim == 1 and to_normalize.shape == (self.y_dim,)
        x = jnp.dot(jnp.linalg.pinv(self.tracking_c), to_normalize)
        x_norm = x / stats.std
        return jnp.dot(self.tracking_c, x_norm)

    @partial(jax.jit, static_argnums=0)
    def denormalize_c_scale(self, to_denormalize: jnp.ndarray, stats: Stats):
        assert to_denormalize.ndim == 1 and to_denormalize.shape == (self.y_dim,)
        x = jnp.dot(jnp.linalg.pinv(self.tracking_c), to_denormalize)
        x_denorm = x * stats.std
        return jnp.dot(self.tracking_c, x_denorm)

    @partial(jax.jit, static_argnums=0)
    def normalize_stats(self, to_normalize: jnp.ndarray):
        assert to_normalize.ndim == 2
        mean = jnp.mean(to_normalize, axis=0)
        std = jnp.std(to_normalize, axis=0) + self.num_correction
        return Stats(mean, std)

    @partial(jax.jit, static_argnums=0)
    def normalize(self, to_normalize: jnp.ndarray, stats: Stats):
        assert to_normalize.ndim == 1
        return (to_normalize - stats.mean) / stats.std

    @partial(jax.jit, static_argnums=0)
    def normalize_std(self, to_normalize: jnp.ndarray, stats: Stats):
        assert to_normalize.ndim == 1
        return to_normalize / stats.std

    @partial(jax.jit, static_argnums=0)
    def denormalize(self, to_denormalize: jnp.ndarray, stats: Stats):
        assert to_denormalize.ndim == 1
        return to_denormalize * stats.std + stats.mean

    @partial(jax.jit, static_argnums=0)
    def denormalize_std(self, to_denormalize: jnp.ndarray, stats: Stats):
        assert to_denormalize.ndim == 1
        return to_denormalize * stats.std

    @partial(jax.jit, static_argnums=0)
    def denormalize_smoother_der(self, smoother_der, ys_stats: Stats, ts_stats: Stats):
        assert smoother_der.shape == (self.state_dim,)
        sigma_ys = ys_stats.std
        assert sigma_ys.shape == (self.state_dim,)
        return (sigma_ys / ts_stats.std) * smoother_der

    @partial(jax.jit, static_argnums=0)
    def denormalize_smoother_der_var(self, smoother_der, ys_stats: Stats):
        assert smoother_der.shape == (self.state_dim,)
        sigma_ys = ys_stats.std
        assert sigma_ys.shape == (self.state_dim,)
        return sigma_ys ** 2 * smoother_der

    @partial(jax.jit, static_argnums=0)
    def denormalize_smoother_der_std(self, smoother_der, ys_stats: Stats, ts_stats: Stats):
        assert smoother_der.shape == (self.state_dim,)
        sigma_ys = ys_stats.std
        assert sigma_ys.shape == (self.state_dim,)
        return (sigma_ys / ts_stats.std) * smoother_der


class DataLoader:
    def __init__(self, batch_size: BatchSize, no_batching: bool = False):
        self.batch_size = batch_size
        self.no_batching = no_batching

    @staticmethod
    def _create_data_loader(data, batch_size: int, key, shuffle: bool = True, infinite: bool = True):
        ds = tf.data.Dataset.from_tensor_slices(data)
        if shuffle:
            seed = int(jax.random.randint(key, (1,), 0, 10 ** 8))
            ds = ds.shuffle(buffer_size=4 * batch_size, seed=seed, reshuffle_each_iteration=True)
        if infinite:
            ds = ds.repeat()
        ds = ds.batch(batch_size)
        ds = tfds.as_numpy(ds)
        return ds

    @staticmethod
    def _non_gen():
        while True:
            yield None

    def prepare_loader(self, dataset: DataLearn, key: random.PRNGKey, no_dynamics_data: bool = False):
        if self.no_batching:
            def _return_all():
                while True:
                    yield dataset.smoothing_data, dataset.matching_data, dataset.dynamics_data

            return _return_all()
        if no_dynamics_data:
            ds_smoothing = self._create_data_loader(dataset.smoothing_data, self.batch_size.smoothing, key)
            ds_matching = self._create_data_loader(dataset.matching_data, self.batch_size.matching, key)
            ds_dynamics = self._non_gen()
            return zip(ds_smoothing, ds_matching, ds_dynamics)
        else:
            ds_smoothing = self._create_data_loader(dataset.smoothing_data, self.batch_size.smoothing, key)
            ds_matching = self._create_data_loader(dataset.matching_data, self.batch_size.matching, key)
            ds_dynamics = self._create_data_loader(dataset.dynamics_data, self.batch_size.dynamics, key)
            return zip(ds_smoothing, ds_matching, ds_dynamics)


if __name__ == '__main__':
    num_smooth = 10
    num_vf = 20
    xs_dim = 3
    us_dim = 2

    data = DataLearn(
        smoothing_data=SmoothingData(
            x0s=jnp.ones(shape=(num_smooth, xs_dim)),
            us=jnp.ones(shape=(num_smooth, us_dim)),
            ts=jnp.ones(shape=(num_smooth, 1)),
            ys=jnp.ones(shape=(num_smooth, xs_dim))),
        dynamics_data=None,
        matching_data=MatchingData(
            ts=jnp.ones(shape=(num_smooth, 1)),
            us=jnp.ones(shape=(num_smooth, us_dim)),
            x0s=jnp.ones(shape=(num_smooth, xs_dim)), )
    )

    batch_size = BatchSize(smoothing=13, dynamics=5, matching=3)
