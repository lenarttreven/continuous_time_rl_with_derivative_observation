import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from jax.config import config

from cucrl.main.data_stats import Normalizer, Stats
from cucrl.utils.ensembles import DeterministicEnsemble, DataStatsBNN, DataRepr
from cucrl.utils.helper_functions import AngleLayerDynamics

config.update("jax_enable_x64", True)

key = random.PRNGKey(0)
input_dim = 1
output_dim = 1

noise_level = 0.1
d_l, d_u = 0, 10
xs = random.uniform(key=random.PRNGKey(0), shape=(10, input_dim), minval=d_l, maxval=d_u)
ys = jnp.sin(xs)
ys = ys + noise_level * random.normal(key=random.PRNGKey(0), shape=ys.shape)
data_std = noise_level * jnp.ones(shape=ys.shape)
data_stats = DataStatsBNN(input_stats=Stats(mean=jnp.mean(xs, axis=0), std=jnp.std(xs, axis=0)),
                          output_stats=Stats(mean=jnp.mean(ys, axis=0), std=jnp.std(ys, axis=0)))

angle_layer = AngleLayerDynamics(state_dim=input_dim, control_dim=0, angles_dim=[],
                                 state_scaling=jnp.eye(input_dim), )
normalizer = Normalizer(state_dim=input_dim, action_dim=output_dim, angle_layer=angle_layer)

num_particles = 5
model = DeterministicEnsemble(input_dim=input_dim, output_dim=output_dim, features=[64, 64, 64],
                              num_particles=num_particles, normalizer=normalizer)

train_data = DataRepr(xs=xs, ys=ys)
start_time = time.time()

model_params, model_stats = model.fit_model(dataset=train_data, num_epochs=4000, data_stats=data_stats,
                                            data_std=data_std, batch_size=32)
print(f"Training time: {time.time() - start_time:.2f} seconds")

test_xs = jnp.linspace(0, 10, 1000).reshape(-1, 1)
test_ys = jnp.sin(test_xs)

test_ys_noisy = jnp.sin(test_xs) + noise_level * random.normal(key=random.PRNGKey(0), shape=test_ys.shape)

test_stds = noise_level * jnp.ones(shape=test_ys.shape)
num_ps = 10
ps = jnp.linspace(0, 1, num_ps + 1)[1:]
alpha_best = model.calculate_calibration_alpha(model_params, model_stats, test_xs, test_ys, test_ys_noisy, ps,
                                               data_stats)

best_ps = model.calculate_calibration_score(model_params, model_stats, test_xs, test_ys, test_ys_noisy, ps,
                                            data_stats, alpha_best)
print(best_ps.shape)
print("Test ps: ", best_ps)
print("Target ps: ", ps.reshape(-1, 1))

apply_ens = vmap(model.apply_eval, in_axes=(None, None, 0, None))
preds = vmap(apply_ens, in_axes=(0, 0, None, None))(model_params, model_stats, test_xs, data_stats)

for j in range(output_dim):
    plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
    for i in range(num_particles):
        plt.plot(test_xs, preds[i, :, j], label='NN prediction', color='black', alpha=0.3)
    plt.plot(test_xs, jnp.mean(preds[..., j], axis=0), label='Mean', color='blue')
    plt.fill_between(test_xs.reshape(-1),
                     (jnp.mean(preds[..., j], axis=0) - 2 * alpha_best[j] * jnp.std(preds[..., j], axis=0)).reshape(-1),
                     (jnp.mean(preds[..., j], axis=0) + 2 * alpha_best[j] * jnp.std(preds[..., j], axis=0)).reshape(-1),
                     label=r'$2\sigma$', alpha=0.3, color='blue')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

plt.plot(test_xs, jnp.std(preds[..., 0], axis=0), label='Std')
plt.legend()
plt.show()


def apply_particles(x):
    return vmap(model.apply_eval, in_axes=(0, 0, None, None))(model_params, model_stats, x, data_stats)


from jax import jacrev
from functools import partial
from jax.flatten_util import ravel_pytree


def get_features(x):
    def get_gradient(param, x):
        grads = jacrev(model.apply_eval, argnums=0)(param, model_stats, x, data_stats)
        return ravel_pytree(grads)[0]

    _gradients_ensemble = vmap(get_gradient, in_axes=(0, None))
    gradients_ensemble = partial(_gradients_ensemble, model_params)
    return jnp.mean(gradients_ensemble(x), axis=0)


# def test_kernel(x1, x2):
#     feats_1 = get_features(x1)
#     feats_2 = get_features(x2)
#     # Here we can do sth smarter with the features, since we have distribution over the
#     # features (distribution comes from different ensemble members)
#     return (jnp.dot(feats_1, feats_2) / feats_1.size).reshape(1, )


def kernel(x1, x2):
    preds_1 = apply_particles(x1)
    preds_2 = apply_particles(x2)
    return jnp.sum((preds_1 - jnp.mean(preds_1, axis=0)[jnp.newaxis, ...]) * (
            preds_2 - jnp.mean(preds_2, axis=0)[jnp.newaxis, ...]), axis=0) / (model.num_particles - 1)


kernel_v = vmap(kernel, in_axes=(0, None), out_axes=1)
kernel_m = vmap(kernel_v, in_axes=(None, 0), out_axes=2)

potential_xs = jnp.linspace(0, 10, 1000).reshape(-1, 1)

from cucrl.utils.greedy_point_selection import greedy_largest_subdeterminant_jit, greedy_distance_maximization_jit

K = kernel_m(potential_xs, potential_xs)
K = jnp.asarray(K, dtype=jnp.float64)


# preds = vmap(apply_ens, in_axes=(0, 0, None, None))(model_params, model_stats, potential_xs, data_stats)
# preds = preds[..., 0]
# mean = jnp.mean(preds, axis=0)
# K = (preds - mean).T @ (preds - mean) / (model.num_particles - 1)
# K = jnp.asarray(K, dtype=jnp.float64)
# K = jnp.expand_dims(K, axis=0)


def rbf(x, y):
    return jnp.exp(-jnp.sum((x - y) ** 2))


rbf_v = vmap(rbf, in_axes=(0, None), out_axes=0)
rbf_m = vmap(rbf_v, in_axes=(None, 0), out_axes=1)

jitter_K = rbf_m(potential_xs, potential_xs)
# min_diagonal = jnp.min(jnp.diag(K[0]))
rank = jnp.linalg.matrix_rank(K[0])
largest_eigenvalue = jnp.linalg.eigvalsh(jitter_K)[-1]
jitter_size = jnp.linalg.eigvalsh(K[0])[-rank]
K = K  # + 1e-17 * jitter_K

best_indices, potential_indices = greedy_largest_subdeterminant_jit(K, jnp.arange(20))

preds = vmap(apply_ens, in_axes=(0, 0, None, None))(model_params, model_stats, potential_xs, data_stats)

plt.plot(potential_xs.reshape(-1), jnp.std(preds[..., 0], axis=0), label='Std')
plt.scatter(potential_xs, jnp.zeros_like(potential_xs), color='blue')
plt.scatter(potential_xs[best_indices], jnp.zeros_like(potential_xs[best_indices]), color='red')
plt.show()
