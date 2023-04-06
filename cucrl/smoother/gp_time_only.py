from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax.core import FrozenDict
from jax import random, vmap, grad
from jax.scipy.stats import multivariate_normal

from cucrl.main.data_stats import DataStats
from cucrl.main.data_stats import Normalizer, DataLearn, SmoothingData, MatchingData
from cucrl.smoother.abstract_smoother import AbstractSmoother
from cucrl.utils.classes import SmootherPosterior, SmootherApply, SampledData
from cucrl.utils.helper_functions import AngleLayerDynamics
from cucrl.utils.helper_functions import make_positive

pytree = Any


class GPTimeOnly(AbstractSmoother):
    def __init__(self, state_dim: int, noise_stds: jnp.ndarray, normalizer: Normalizer,
                 numerical_correction: float = 1e-10):
        super().__init__(state_dim, numerical_correction)
        self.normalizer = normalizer
        self.noise_stds = noise_stds
        self.v_kernel = vmap(self.kernel, in_axes=(0, None, None), out_axes=0)
        self.m_kernel = vmap(self.v_kernel, in_axes=(None, 0, None), out_axes=1)
        self.m_kernel_multiple_output = vmap(self.m_kernel, in_axes=(None, None, 0), out_axes=0)

    def d_kernel(self, x, y, params):
        return grad(self.kernel, argnums=1)(x, y, params).reshape()

    def dd_kernel(self, x, y, params):
        return grad(self.d_kernel, argnums=0)(x, y, params).reshape()

    def kernel(self, x, y, params):
        assert x.shape == y.shape == (1,)
        assert params["lengthscale"].shape == ()
        return jnp.exp(- jnp.sum((x - y) ** 2) / make_positive(params["lengthscale"]) ** 2)

    def mean_and_std_der_one(self, t, observation_times, observations, params, data_stats):
        num_points = observation_times.shape[0]

        t = self.normalizer.normalize(t, data_stats.ts_stats)
        observation_times = vmap(self.normalizer.normalize, in_axes=(0, None))(observation_times, data_stats.ts_stats)
        observations = vmap(self.normalizer.normalize, in_axes=(0, None))(observations, data_stats.ys_stats)

        covariance_matrix = self.m_kernel_multiple_output(observation_times, observation_times, params)
        noise_term = (self.normalizer.normalize_std(self.noise_stds, data_stats.ys_stats)[:, jnp.newaxis,
                      jnp.newaxis] + self.numerical_correction) * jnp.eye(num_points)[jnp.newaxis, ...]

        noisy_covariance_matrix = covariance_matrix + noise_term

        cholesky_tuples = vmap(jax.scipy.linalg.cho_factor)(noisy_covariance_matrix)

        d_v_kernel = vmap(self.d_kernel, in_axes=(0, None, None))
        k_x_X_d = vmap(d_v_kernel, in_axes=(None, None, 0), out_axes=0)(observation_times, t, params)

        # Compute std
        denoised_var = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 0))((cholesky_tuples[0], False), k_x_X_d)
        var = vmap(self.dd_kernel, in_axes=(None, None, 0))(t, t, params) - vmap(jnp.dot)(k_x_X_d, denoised_var)
        std = jnp.sqrt(var)

        # Compute mean
        denoised_mean = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 1))((cholesky_tuples[0], False),
                                                                                 observations)
        mean = vmap(jnp.dot)(k_x_X_d, denoised_mean)

        # Denormalize
        mean = self.normalizer.denormalize_smoother_der(mean, data_stats.ys_stats, data_stats.ts_stats)
        std = self.normalizer.denormalize_smoother_der_std(std, data_stats.ys_stats, data_stats.ts_stats)

        return mean, std

    def mean_and_std_one(self, t, observation_times, observations, params, data_stats):
        num_points = observation_times.shape[0]
        t = self.normalizer.normalize(t, data_stats.ts_stats)
        observation_times = vmap(self.normalizer.normalize, in_axes=(0, None))(observation_times, data_stats.ts_stats)
        observations = vmap(self.normalizer.normalize, in_axes=(0, None))(observations, data_stats.ys_stats)

        covariance_matrix = self.m_kernel_multiple_output(observation_times, observation_times, params)

        noise_term = (self.normalizer.normalize_std(self.noise_stds, data_stats.ys_stats)[:, jnp.newaxis,
                      jnp.newaxis] + self.numerical_correction) * jnp.eye(num_points)[jnp.newaxis, ...]

        noisy_covariance_matrix = covariance_matrix + noise_term
        k_x_X = vmap(self.v_kernel, in_axes=(None, None, 0), out_axes=0)(observation_times, t, params)
        cholesky_tuples = vmap(jax.scipy.linalg.cho_factor)(noisy_covariance_matrix)

        # Compute std
        denoised_var = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 0))((cholesky_tuples[0], False), k_x_X)
        var = vmap(self.kernel, in_axes=(None, None, 0))(t, t, params) - vmap(jnp.dot)(k_x_X, denoised_var)
        std = jnp.sqrt(var)

        # Compute mean
        denoised_mean = vmap(jax.scipy.linalg.cho_solve, in_axes=((0, None), 1))((cholesky_tuples[0], False),
                                                                                 observations)
        mean = vmap(jnp.dot)(k_x_X, denoised_mean)

        # Denormalize
        mean = self.normalizer.denormalize(mean, data_stats.ys_stats)
        std = self.normalizer.denormalize_std(std, data_stats.ys_stats)

        return mean, std

    def loss(self, parameters, stats, observation_times, observations: jax.Array, data_stats: DataStats):
        # Normalize observation_times and observations
        num_points = observation_times.shape[0]
        observation_times = vmap(self.normalizer.normalize, in_axes=(0, None))(observation_times, data_stats.ts_stats)
        observations = vmap(self.normalizer.normalize, in_axes=(0, None))(observations, data_stats.ys_stats)

        covariance_matrix = self.m_kernel_multiple_output(observation_times, observation_times, parameters)

        noise_term = (self.normalizer.normalize_std(self.noise_stds, data_stats.ys_stats)[:, jnp.newaxis,
                      jnp.newaxis] + self.numerical_correction) * jnp.eye(num_points)[jnp.newaxis, ...]
        noisy_covariance_matrix = covariance_matrix + noise_term
        log_pdf = vmap(multivariate_normal.logpdf, in_axes=(1, None, 0))(observations, jnp.zeros(num_points, ),
                                                                         noisy_covariance_matrix)
        # jax.debug.print("Log Likelihood: {x}", x=log_pdf)
        return - jnp.sum(log_pdf)

    def apply(self, parameters: pytree, stats: FrozenDict, observation_times: jax.Array, matching_times: jax.Array,
              ic_for_observation_times: jax.Array, ic_for_matching_times: jax.Array, observations: jax.Array,
              key: jax.Array, data_stats: DataStats) -> SmootherApply:
        loss = self.loss(parameters, stats, observation_times, observations, data_stats)
        mean, std = vmap(self.mean_and_std_one, in_axes=(0, None, None, None, None))(matching_times, observation_times,
                                                                                     observations, parameters,
                                                                                     data_stats)
        mean_d, std_d = vmap(self.mean_and_std_der_one, in_axes=(0, None, None, None, None))(matching_times,
                                                                                             observation_times,
                                                                                             observations, parameters,
                                                                                             data_stats)
        return SmootherApply(loss=loss, xs_mean=mean, xs_var=std ** 2, xs_dot_mean=mean_d, xs_dot_var=std_d ** 2,
                             xs_dot_var_given_x=std_d, updated_stats=dict())

    def initialize_parameters(self, key):
        parameters = dict()
        # Inout dimension is one, because we only have one input dimension (time)
        # Ouput dimension is state_dim, because we have one GP per state dimension
        # So we have lengthscales for each state dimension
        parameters["lengthscale"] = random.normal(key=key, shape=(self.state_dim,))
        # parameters["lengthscale"] = -0.5 * jnp.ones(self.state_dim)
        return parameters, dict()

    def posterior(self, parameters: pytree, stats: FrozenDict, evaluation_times: jax.Array,
                  ic_for_evaluation_times: jax.Array, observation_times, ic_for_observation_times, observations,
                  data_stats: DataStats) -> SmootherPosterior:
        mean, std = vmap(self.mean_and_std_one, in_axes=(0, None, None, None, None))(evaluation_times,
                                                                                     observation_times, observations,
                                                                                     parameters, data_stats)
        mean_d, std_d = vmap(self.mean_and_std_der_one, in_axes=(0, None, None, None, None))(evaluation_times,
                                                                                             observation_times,
                                                                                             observations, parameters,
                                                                                             data_stats)
        return SmootherPosterior(xs_mean=mean, xs_var=std ** 2, xs_dot_mean=mean_d, xs_dot_var=std_d ** 2)

    def regularization(self, parameters, weights):
        return 0.0

    def sample_vector_field_data(self, params, stats, observation_times, observations, ic_for_observation_times,
                                 data_stats: DataStats, key: jax.Array) -> SampledData:
        observation_times = vmap(self.normalizer.normalize, in_axes=(0, None))(observation_times, data_stats.ts_stats)
        observations = vmap(self.normalizer.normalize, in_axes=(0, None))(observations, data_stats.ys_stats)

        num_points = observation_times.shape[0]
        K_T_T = self.m_kernel_multiple_output(observation_times, observation_times, params)
        noise_term = (self.normalizer.normalize_std(self.noise_stds, data_stats.ys_stats)[:, jnp.newaxis,
                      jnp.newaxis] + self.numerical_correction) * jnp.eye(num_points)[jnp.newaxis, ...]
        K_T_T = K_T_T + noise_term

        def _posterior_1d(t, params_one, K_T_T_one, obs):
            assert t.shape == (1,)
            k_t_T = self.v_kernel(observation_times, t, params_one)
            k_t_T_dot = vmap(self.d_kernel, in_axes=(0, None, None))(observation_times, t, params_one)

            k_t = jnp.stack([k_t_T, k_t_T_dot], axis=1)

            prior_k = jnp.array([[self.kernel(t, t, params_one), self.d_kernel(t, t, params_one)],
                                 [self.d_kernel(t, t, params_one), self.dd_kernel(t, t, params_one)]])

            cholesky_tuples = jax.scipy.linalg.cho_factor(K_T_T_one)

            # Compute mean
            denoised_obs = jax.scipy.linalg.cho_solve((cholesky_tuples[0], False), obs)
            mean = jnp.dot(k_t.T, denoised_obs)

            # Compute covariance
            denoised_var = jax.scipy.linalg.cho_solve((cholesky_tuples[0], False), k_t)
            cov = prior_k - jnp.dot(k_t.T, denoised_var)
            return mean, cov

        _posterior = vmap(_posterior_1d, in_axes=(0, None, None, 0))

        _posterior_v = vmap(_posterior_1d, in_axes=(0, None, None, None))
        _posterior = vmap(_posterior_v, in_axes=(None, 0, 0, 1), out_axes=1)

        means, covs = _posterior(observation_times, params, K_T_T, observations)
        samples = random.multivariate_normal(key=key, mean=means, cov=covs)

        xs_var = covs[..., 0, 0]
        xs_dot_var = covs[..., 1, 1]
        xs_dot_vs_xs_cov = covs[..., 1, 0]
        variance_estimate = xs_var + xs_dot_var - 2 * xs_dot_vs_xs_cov

        xs = vmap(self.normalizer.denormalize, in_axes=(0, None))(samples[..., 0], data_stats.ys_stats)
        xs_dot = vmap(self.normalizer.denormalize_smoother_der, in_axes=(0, None, None))(samples[..., 1],
                                                                                         data_stats.ys_stats,
                                                                                         data_stats.ts_stats)
        return SampledData(xs=xs, xs_dot=xs_dot, std_xs_dot=jnp.sqrt(variance_estimate))


def test_mean_and_std_one():
    from jax.config import config

    config.update("jax_enable_x64", True)
    state_dim = 3
    noise = 0.01

    ts = jnp.linspace(0, 2, 50, dtype=jnp.float64)
    ys = jnp.array([jnp.sin(ts), jnp.cos(ts), ts ** 2]) + noise * random.normal(key=random.PRNGKey(0),
                                                                                shape=(state_dim, ts.shape[0]))
    ys = ys.T
    angle_layer = AngleLayerDynamics(state_dim=state_dim, control_dim=1, angles_dim=[],
                                     state_scaling=jnp.eye(state_dim))
    normalizer = Normalizer(state_dim=state_dim, action_dim=1, angle_layer=angle_layer)

    test = GPTimeOnly(state_dim, 0.1 * jnp.ones(shape=(state_dim,)), normalizer)

    smoothing_data = SmoothingData(
        ts=ts.reshape(-1, 1),
        ys=ys,
        x0s=ys,
        us=jnp.zeros(shape=(1, ts.shape[0])),
    )
    matching_data = MatchingData(
        ts=ts.reshape(-1, 1),
        us=jnp.zeros(shape=(1, ts.shape[0])),
        x0s=ys,
    )
    data = DataLearn(smoothing_data=smoothing_data, matching_data=matching_data)

    data_stats = normalizer.compute_stats(data=data)

    params = test.initialize_parameters(random.PRNGKey(0))[0]
    # params = {'lengthscale': jnp.array([0.01, 0.01], dtype=jnp.float32)}
    test_ts = jnp.linspace(-0.5, 2 + 0.5, 100).reshape(-1, 1)

    mean, std = vmap(test.mean_and_std_one, in_axes=(0, None, None, None, None))(test_ts, ts.reshape(-1, 1), ys, params,
                                                                                 data_stats)

    mean_d, std_d = vmap(test.mean_and_std_der_one, in_axes=(0, None, None, None, None))(test_ts, ts.reshape(-1, 1), ys,
                                                                                         params,
                                                                                         data_stats)

    numeric_der = jnp.diff(mean, axis=0) / (test_ts[1] - test_ts[0]).reshape()

    tru_der = jnp.array([jnp.cos(test_ts).reshape(-1), -jnp.sin(test_ts.reshape(-1)), 2 * test_ts.reshape(-1)])
    xs = jnp.array([jnp.sin(test_ts).reshape(-1), jnp.cos(test_ts.reshape(-1)), (test_ts ** 2).reshape(-1)])

    out = test.sample_vector_field_data(params, None, ts.reshape(-1, 1), ys, None, data_stats, random.PRNGKey(0))

    for i in range(state_dim):
        plt.plot(test_ts.reshape(-1), mean[:, i], color="blue")
        plt.fill_between(test_ts.reshape(-1), mean[:, i] - std[:, i], mean[:, i] + std[:, i], color="blue", alpha=0.2)
        plt.plot(test_ts.reshape(-1), xs[i, :], color="black")
        plt.plot(ts, ys[:, i], color="red", linestyle="none", marker="x")
        plt.plot(ts, out.xs[:, i], color="green", linestyle="none", marker="x")
        plt.show()

    for i in range(state_dim):
        plt.plot(test_ts.reshape(-1), mean_d[:, i], color="blue")
        plt.plot(test_ts.reshape(-1), tru_der[i, :], color="black")
        plt.plot(test_ts.reshape(-1)[1:], numeric_der[:, i], color="red")
        plt.fill_between(test_ts.reshape(-1), mean_d[:, i] - std_d[:, i], mean_d[:, i] + std_d[:, i], color="blue",
                         alpha=0.2)
        plt.plot(ts, out.xs_dot[:, i], color="green", linestyle="none", marker="x")
        plt.show()


def debug_test():
    from jax.config import config

    config.update("jax_enable_x64", True)
    state_dim = 2

    ts = jnp.array([[0.],
                    [0.13739533],
                    [0.27479067],
                    [0.41239533],
                    [0.54958134],
                    [0.68739533],
                    [0.824372],
                    [0.96239533],
                    [1.09916267],
                    [1.23739533],
                    [1.37395334],
                    [1.51239533],
                    [1.64874401],
                    [1.78739533],
                    [1.92353468],
                    [2.06239533],
                    [2.19832535],
                    [2.33739533],
                    [2.47311601],
                    [2.61239533],
                    [2.74790668],
                    [2.88739533],
                    [3.02269735],
                    [3.16239533],
                    [3.29748802],
                    [3.43739533],
                    [3.57227869],
                    [3.71239533],
                    [3.84706936],
                    [3.98739533],
                    [4.12186002],
                    [4.26239533],
                    [4.39665069],
                    [4.53739533],
                    [4.67144136],
                    [4.81239533],
                    [4.94623203],
                    [5.08739533],
                    [5.2210227],
                    [5.36239533],
                    [5.49581336],
                    [5.63739533],
                    [5.77060403],
                    [5.91239533],
                    [6.0453947],
                    [6.18739533],
                    [6.32018537],
                    [6.46239533],
                    [6.59497604],
                    [6.73739533],
                    [6.86976671],
                    [7.01239533],
                    [7.14455737],
                    [7.28739533],
                    [7.41934804],
                    [7.56239533],
                    [7.69413871],
                    [7.83739533],
                    [7.96892938],
                    [8.11239533],
                    [8.24372005],
                    [8.38739533],
                    [8.51851072],
                    [8.66239533],
                    [8.79330138],
                    [8.93739533],
                    [9.06809205],
                    [9.21239533],
                    [9.34288272],
                    [9.48739533],
                    [9.61767339],
                    [9.76239533],
                    [9.89246406]], dtype=jnp.float64)
    ys = jnp.array([[1.57291144, -0.01600522],
                    [1.60852285, 0.56404594],
                    [1.67278726, 1.14729762],
                    [1.74003904, 1.78036429],
                    [1.88748474, 2.4342891],
                    [2.09547794, 3.08855132],
                    [2.31198956, 3.71127297],
                    [2.58924873, 4.25519978],
                    [2.90109339, 4.71646445],
                    [3.22947227, 5.00875847],
                    [3.583548, 5.12698946],
                    [3.92636495, 5.09225793],
                    [4.27100241, 4.95790587],
                    [4.61600127, 4.6770668],
                    [4.92764207, 4.40307937],
                    [5.22723334, 4.15680067],
                    [5.50189197, 3.94089811],
                    [5.75709563, 3.84094425],
                    [6.0293291, 3.82227958],
                    [6.30492428, 3.90329937],
                    [6.55163772, 4.12097225],
                    [6.83699952, 4.41132546],
                    [7.16582974, 4.80492927],
                    [7.51307427, 5.31360229],
                    [7.88549766, 5.80676255],
                    [8.34155257, 6.26209667],
                    [8.75646667, 6.58213893],
                    [9.23229446, 6.68792279],
                    [9.67064167, 6.5012544],
                    [10.11548182, 6.07404718],
                    [10.51798031, 5.44041665],
                    [10.84072565, 4.69290572],
                    [11.14829991, 3.92169797],
                    [11.37850911, 3.12551283],
                    [11.58329073, 2.36147593],
                    [11.71860061, 1.65258569],
                    [11.81309936, 1.0009237],
                    [11.85289709, 0.38014124],
                    [11.84363602, -0.20573217],
                    [11.81392411, -0.79783453],
                    [11.764552, -1.3834007],
                    [11.63398342, -1.97064182],
                    [11.47511249, -2.56387225],
                    [11.2840789, -3.20595042],
                    [11.05019051, -3.79431374],
                    [10.73949981, -4.37153148],
                    [10.44605688, -4.868795],
                    [10.09255488, -5.26953578],
                    [9.73199159, -5.45194875],
                    [9.33069709, -5.39746936],
                    [8.98216732, -5.12608724],
                    [8.64568491, -4.65630542],
                    [8.35338127, -4.0329514],
                    [8.09906032, -3.28701112],
                    [7.91793626, -2.53606441],
                    [7.76796084, -1.71863377],
                    [7.65004314, -0.9507957],
                    [7.62324593, -0.10763822],
                    [7.62071373, 0.63827126],
                    [7.70969195, 1.48473181],
                    [7.81932025, 2.25791056],
                    [8.03341938, 3.07768813],
                    [8.26914647, 3.76903785],
                    [8.5552086, 4.46985138],
                    [8.87064688, 4.98006437],
                    [9.2287228, 5.32768026],
                    [9.59053148, 5.45029028],
                    [9.96865839, 5.32891101],
                    [10.30409912, 5.05699382],
                    [10.65161529, 4.57642381],
                    [10.95408794, 3.99894888],
                    [11.21112527, 3.38114627],
                    [11.39894183, 2.80300023]], dtype=jnp.float64)
    noise_std = jnp.array([0.01, 0.01], dtype=jnp.float64)

    angle_layer = AngleLayerDynamics(state_dim=state_dim, control_dim=1, angles_dim=[],
                                     state_scaling=jnp.eye(state_dim))
    normalizer = Normalizer(state_dim=state_dim, action_dim=1, angle_layer=angle_layer)

    test = GPTimeOnly(state_dim, noise_std, normalizer)

    smoothing_data = SmoothingData(
        ts=ts.reshape(-1, 1),
        ys=ys,
        x0s=ys,
        us=jnp.zeros(shape=(1, ts.shape[0])),
    )
    matching_data = MatchingData(
        ts=ts.reshape(-1, 1),
        us=jnp.zeros(shape=(1, ts.shape[0])),
        x0s=ys,
    )
    data = DataLearn(smoothing_data=smoothing_data, matching_data=matching_data)

    data_stats = normalizer.compute_stats(data=data)

    # params = test.initialize_parameters(random.PRNGKey(0))[0]
    params = {'lengthscale': jnp.array([-0.5, -0.5], dtype=jnp.float32)}
    test_ts = jnp.linspace(0.0, 10, 100).reshape(-1, 1)

    mean, std = vmap(test.mean_and_std_one, in_axes=(0, None, None, None, None))(test_ts, ts.reshape(-1, 1), ys,
                                                                                 params,
                                                                                 data_stats)

    for i in range(state_dim):
        plt.plot(test_ts.reshape(-1), mean[:, i], color="blue")
        plt.fill_between(test_ts.reshape(-1), mean[:, i] - std[:, i], mean[:, i] + std[:, i], color="blue",
                         alpha=0.2)
        plt.plot(ts, ys[:, i], color="red", linestyle="none", marker="x")
        # plt.plot(ts, out.xs[:, i], color="green", linestyle="none", marker="x")
        plt.show()

    test.loss(params, None, ts, ys, data_stats)


if __name__ == '__main__':
    out = test_mean_and_std_one()