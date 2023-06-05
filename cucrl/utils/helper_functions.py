import os
import sys
from functools import partial
from math import comb
from typing import Any, List, Sequence, Optional, Callable

import jax
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from jax import jit, vmap, random, numpy as jnp
from jax.scipy.linalg import cho_solve, cho_factor
from jax.tree_util import register_pytree_node_class
from jax.tree_util import tree_flatten
from scipy.stats import chi2

from cucrl.utils.representatives import Norm, BetaType
from cucrl.utils.splines import MultivariateSpline

pytree = Any
type_kernel = Callable[[jax.Array, jax.Array], jax.Array]
type_mean = Callable[[jax.Array], jax.Array]


class BetaExploration:
    def __init__(self, delta, state_dim, rkhs_bound, type: BetaType):
        self.delta = delta
        self.state_dim = state_dim
        self.rkhs_bound = rkhs_bound
        self.info_gain = jnp.zeros(shape=(state_dim,), dtype=jnp.float64)
        self.type = type

    def get_beta_bnn(self, num_episodes):
        beta_squared = chi2.isf(self.delta / (self.state_dim * num_episodes), df=1)
        beta = jnp.sqrt(beta_squared)
        return beta

    def __call__(self, num_episodes):
        if self.type == BetaType.GP:
            return self.rkhs_bound + jnp.sqrt(2 * (self.info_gain + jnp.log(self.state_dim) / self.delta))
        elif self.type == BetaType.BNN:
            return jnp.repeat(self.get_beta_bnn(num_episodes), self.state_dim)

    def update_info_gain(self, stds: jax.Array, taus: jax.Array):
        assert stds.shape == taus.shape and stds.ndim == 2 and stds.shape[1] == self.state_dim
        new_info_gain = 0.5 * jnp.log(1 + stds ** 2 / taus ** 2).sum(axis=0)
        self.info_gain += new_info_gain


@register_pytree_node_class
class GPFunc:
    def __init__(self, dim_in, dim_out, kernel, mean):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel = kernel
        self.mean = mean
        self.eps = 1e-6
        self.xs = jnp.ones(shape=(0, dim_in))
        self.f_vec = jnp.ones(shape=(0, dim_out))

    @jit
    def _compute_f_vec(self, xs, ys):
        cov_matrix = vmap(vmap(self.kernel, in_axes=(0, None)),
                          in_axes=(None, 0))(xs, xs) + self.eps * jnp.eye(xs.shape[0])
        centered_data = ys - vmap(self.mean)(xs)
        cov_matrix_sqrt = cho_factor(cov_matrix)
        scaled_vector = vmap(cho_solve, in_axes=(None, 1), out_axes=1)(cov_matrix_sqrt, centered_data)
        return scaled_vector

    def prepare_function_vec(self, xs, ys):
        self.xs = xs
        self.f_vec = self._compute_f_vec(xs, ys)

    @jit
    def eval(self, x):
        first_term = self.mean(x)
        k_x = vmap(self.kernel, in_axes=(None, 0))(x, self.xs)
        second_term = jnp.einsum('i, ij -> j', k_x, self.f_vec)
        return first_term + second_term

    def tree_flatten(self):
        children = (self.xs, self.f_vec)  # arrays / dynamic values
        aux_data = {'dim_in': self.dim_in, 'dim_out': self.dim_out,
                    'kernel': self.kernel, 'mean': self.mean}  # static values
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        xs, f_vec = children
        new_class = cls(dim_in=aux_data['dim_in'], dim_out=aux_data['dim_out'],
                        kernel=aux_data['kernel'], mean=aux_data['mean'])
        new_class.xs = xs
        new_class.f_vec = f_vec
        return new_class


class RandomFunctionQuantile:
    def __init__(self, dim_in: int, dim_out: int, mu: type_mean, k: type_kernel, quantile: float, outer_dim_norm: Norm):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mu = mu
        self.kernel = k
        self.quantile = quantile
        self.outer_dim_norm = outer_dim_norm
        self.eps = 1e-6

    def evaluate_function(self, xs, ys):
        # Here we treat multiple outputs independently
        assert xs.ndim == ys.ndim == 2 and xs.shape[1] == self.dim_in and ys.shape[1] == self.dim_out
        mean = vmap(self.mu)(xs)
        cov_matrix = vmap(vmap(self.kernel, in_axes=(0, None)), in_axes=(None, 0))(xs, xs) + self.eps * jnp.eye(
            xs.shape[0])
        centered_sample = ys - mean
        cov_matrix_sqrt = cho_factor(cov_matrix)
        scaled_vector = vmap(cho_solve, in_axes=(None, 1), out_axes=1)(cov_matrix_sqrt, centered_sample)
        assert centered_sample.ndim == scaled_vector.ndim
        assert centered_sample.shape[1] == scaled_vector.shape[1] == self.dim_out
        return jnp.einsum('ij, ij -> j', centered_sample, scaled_vector)

    # @partial(jit, static_argnums=0)
    def inequality(self, xs, ys):
        if self.outer_dim_norm == Norm.L_INF:
            quantile = chi2.isf(df=xs.shape[0], q=1 - self.quantile ** (1 / self.dim_out))
            # Here we need another layer of quantile propagation
            output_dim_eval = self.evaluate_function(xs, ys)
            return jnp.concatenate([output_dim_eval + quantile, quantile - output_dim_eval])

    def passes_test(self, xs, ys):
        should_be_nonnegative = self.inequality(xs, ys)
        return jnp.all(should_be_nonnegative >= 0)


def sample_func(kernel_func, key, n_dim, t_min, t_max, num_samples=100, max_value: float = 1.0,
                decay_factor: float = 1.0):
    v_k = vmap(kernel_func, in_axes=(0, None), out_axes=0)
    m_k = vmap(v_k, in_axes=(None, 0), out_axes=1)

    t = jnp.linspace(t_min, t_max, num_samples, dtype=jnp.float64)
    key, subkey = random.split(key)
    covariance_matrix = m_k(t, t) + 1e-10 * jnp.eye(num_samples)
    factor = jnp.linalg.cholesky(covariance_matrix)
    # Get n_dim samples from normal distribution
    z = random.normal(key=subkey, shape=(num_samples, n_dim))
    sample_fun = factor @ z

    # Here we scale by max_value
    def scale(xs: jnp.ndarray):
        assert xs.ndim == 1
        return xs / jnp.max(jnp.abs(xs)) * max_value

    sample_fun = vmap(scale, in_axes=1, out_axes=1)(sample_fun)

    @jit
    def sampled_fun(t_eval):
        assert t_eval.shape == ()
        return MultivariateSpline(t, sample_fun)(t_eval.reshape(1)).reshape(n_dim, ) * jnp.exp(-decay_factor * t_eval)

    return sampled_fun


# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


class AngleLayerDynamics:
    def __init__(self, state_dim: int, control_dim: int, angles_dim: List[int], state_scaling: jnp.ndarray):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.angles_dim = angles_dim
        self.rest_dim = []
        self.state_scaling = state_scaling
        assert self.state_scaling.shape == (self.state_dim, self.state_dim)
        state_scaling_inv = 1 / jnp.diag(self.state_scaling)
        self.angles_inv_scaling = state_scaling_inv[jnp.array(self.angles_dim, dtype=int)]

        for i in range(self.state_dim):
            if i not in self.angles_dim:
                self.rest_dim.append(i)
        self.new_state_dim = self.state_dim + len(angles_dim)

    @staticmethod
    def _transform_angle(angle: jnp.ndarray, inverse_scale: jnp.ndarray):
        assert angle.shape == () and inverse_scale.shape == ()
        angle = angle * inverse_scale
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        return jnp.array([cos, sin])

    @partial(jit, static_argnums=0)
    def angle_layer(self, x):
        angles = x[jnp.array(self.angles_dim, dtype=int)]
        rest = x[jnp.array(self.rest_dim, dtype=int)]
        assert angles.ndim == 1 and rest.ndim == 1
        transformed_angles = vmap(self._transform_angle)(angles, self.angles_inv_scaling)
        to_return = jnp.concatenate([transformed_angles.reshape(-1), rest])
        assert to_return.shape == (self.new_state_dim,)
        return to_return


class AngleNormalizer:
    def __init__(self, state_dim: int, control_dim: int, angles_dim: List[int], state_scaling: jnp.ndarray):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.angles_dim = angles_dim
        self.state_scaling = state_scaling
        assert self.state_scaling.shape == (self.state_dim, self.state_dim)
        state_scaling = jnp.diag(self.state_scaling)
        self.angles_scaling = state_scaling[jnp.array(self.angles_dim, dtype=int)]

    @staticmethod
    def _transform_angle(angle: jnp.ndarray, scale: jnp.ndarray):
        assert angle.shape == () and scale.shape == ()
        angle = angle / scale
        cos, sin = jnp.cos(angle), jnp.sin(angle)
        normalized_angle = jnp.arctan2(sin, cos)
        return normalized_angle * scale

    def _transform_x(self, x, angles_dim):
        angles = x[jnp.array(angles_dim, dtype=int)]
        assert angles.ndim == 1
        transformed_angles = vmap(self._transform_angle)(angles, self.angles_scaling)
        transformed_x = x.at[jnp.array(angles_dim, dtype=int)].set(transformed_angles)
        return transformed_x

    @partial(jit, static_argnums=0)
    def transform_x(self, x):
        assert x.ndim == 1
        return self._transform_x(x, self.angles_dim)

    @partial(jit, static_argnums=0)
    def transform_xs(self, xs):
        return vmap(self._transform_x, in_axes=(0, None))(xs, self.angles_dim)


class MLP(nn.Module):
    features: Sequence[int]
    output_dim: Optional[int]

    @nn.compact
    def __call__(self, x, train=False):
        for feat in self.features:
            x = nn.Dense(features=feat)(x)
            x = nn.swish(x)
        if self.output_dim is not None:
            x = nn.Dense(features=self.output_dim)(x)
        return x


class MLPWithBN(nn.Module):
    features: Sequence[int]
    output_dim: Optional[int]

    @nn.compact
    def __call__(self, x, train=False):
        for feat in self.features:
            x = nn.Dense(features=feat)(x)
            x = nn.BatchNorm(
                use_running_average=not train,
                momentum=0.01,
                epsilon=1e-10,
                axis_name='batch'
            )(x)
            x = nn.swish(x)
        if self.output_dim is not None:
            x = nn.Dense(features=self.output_dim)(x)
        return x


@jit
def make_positive(x):
    return jnp.logaddexp(x, 0)


@jax.jit
def squared_l2_norm(tree: pytree) -> jnp.array:
    """Compute the squared l2 norm of a pytree of arrays. Useful for weight decay."""
    leaves, _ = tree_flatten(tree)
    return sum(jnp.vdot(x, x) for x in leaves)


def isnamedtupleinstance(x):
    _type = type(x)
    bases = _type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(_type, '_fields', None)
    if not isinstance(fields, tuple):
        return False
    return all(type(i) == str for i in fields)


def namedtuple_to_dict(obj):
    if isinstance(obj, dict):
        return {key: namedtuple_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [namedtuple_to_dict(value) for value in obj]
    elif isnamedtupleinstance(obj):
        return {key: namedtuple_to_dict(value) for key, value in obj._asdict().items()}
    elif isinstance(obj, tuple):
        return tuple(namedtuple_to_dict(value) for value in obj)
    else:
        return obj


def derivative_coefficient(n, k):
    coefficients = np.zeros(shape=(n + 1,))
    for j in range(n + 1):
        if j == k:
            term = 0
            for i in range(n + 1):
                if i != k:
                    term += 1 / (k - i)
            coefficients[j] = term
        else:
            term = (-1) ** (k + j) * comb(n, j) / (comb(n, k) * (k - j))
            coefficients[j] = term
    return coefficients


def derivative_coefficients(n):
    coeffs = []
    for k in range(n + 1):
        coeffs.append(derivative_coefficient(n, k))
    return np.stack(coeffs)


@partial(jit, static_argnums=(1,))
def moving_window(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice_in_dim(a, start, size, axis=0))(starts)


def angle_layer_dynamics():
    angle_layer = AngleLayerDynamics(state_dim=1, state_scaling=jnp.diag(jnp.array([2])), control_dim=1, angles_dim=[0])
    test_value = 0.8
    x = jnp.array([test_value])
    y = angle_layer.angle_layer(x)
    assert jnp.allclose(y, jnp.array([jnp.cos(test_value / 2), jnp.sin(test_value / 2)]))


def angle_normalizer():
    angle_normalizer = AngleNormalizer(state_dim=1, state_scaling=jnp.diag(jnp.array([2])), control_dim=1,
                                       angles_dim=[0])
    test_value = 0.8
    x = 2 * jnp.array([jnp.pi + test_value])
    y = angle_normalizer.transform_x(x)
    assert jnp.allclose(y, 2 * jnp.array([-jnp.pi + test_value]))


def my_gp_func():
    h = 1.0
    dim_in, dim_out = 1, 2
    xs = jnp.linspace(0, 5, 10)
    ys = jnp.stack([jnp.sin(xs), jnp.cos(xs)], axis=1)

    xs = xs.reshape(-1, dim_in)
    ys = ys.reshape(-1, dim_out)

    def test_k(x, y):
        return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h))

    def test_mu(x):
        return jnp.zeros(shape=(dim_out,))

    gp_func = GPFunc(dim_in=dim_in, dim_out=dim_out, kernel=test_k, mean=test_mu)
    gp_func.prepare_function_vec(xs, ys)
    # gp_func.xs, gp_func.f_vec = gp_func.prepare_function_vec(xs, ys)

    test_xs = jnp.linspace(-5, 10, 100).reshape(-1, 1)
    test_ys = vmap(gp_func.eval)(test_xs)

    plt.scatter(xs, ys[:, 0], color='red')
    plt.plot(test_xs, test_ys[:, 0], color='blue')
    plt.title('Dim 0')
    plt.show()

    plt.scatter(xs, ys[:, 1], color='red')
    plt.plot(test_xs, test_ys[:, 1], color='blue')
    plt.title('Dim 1')
    plt.show()


def norm_difference(z: jax.Array):
    assert z.ndim == 2
    differences = jnp.linalg.norm(z[:-1] - z[1:], axis=1)
    differences_with_0 = jnp.concatenate([jnp.array([0]), differences])
    return jnp.cumsum(differences_with_0)


if __name__ == '__main__':
    a = random.normal(key=random.PRNGKey(0), shape=(10, 3))
    print(norm_difference(a))
