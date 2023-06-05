import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap, random, jit


def greedy_distance_maximization_1d_jit(K: np.ndarray, k: jax.Array):
    assert K.ndim == 2
    """
    :param K: Kernel matrix
    :param k: Number of samples to draw
    :return: Indices of the k samples
    """
    potential_indices = jnp.arange(K.shape[1], dtype=jnp.int32)
    current_indices = jnp.array([], dtype=jnp.int32)

    self_distances = jnp.diag(K)
    kernel_distances = -2 * K + self_distances[None, ...] + self_distances[..., None]

    initial_index = jnp.argmax(self_distances)
    current_indices = jnp.append(current_indices, potential_indices[initial_index])

    for i in range(len(k) - 1):

        def compute_distance(S, index):
            cur_distances = kernel_distances[index, S]
            return jnp.min(cur_distances)

        distances = vmap(compute_distance, in_axes=(None, 0))(
            current_indices, potential_indices
        )
        greedy_index = jnp.argmax(distances)
        current_indices = jnp.append(current_indices, potential_indices[greedy_index])

    return current_indices, potential_indices


def greedy_distance_maximization_jit(K: np.ndarray, k: jax.Array):
    assert K.ndim == 3
    """
    :param K: Kernel matrix
    :param k: Number of samples to draw
    :return: Indices of the k samples
    """
    potential_indices = jnp.arange(K.shape[1], dtype=jnp.int32)
    current_indices = jnp.array([], dtype=jnp.int32)

    self_distances = vmap(jnp.diag)(K)
    kernel_distances = -2 * K + self_distances[:, None, :] + self_distances[..., None]

    initial_index = jnp.argmax(self_distances)
    current_indices = jnp.append(current_indices, potential_indices[initial_index])

    for i in range(len(k) - 1):

        def compute_distance(S, index):
            def _distance_one_dim(K, S, index):
                return K[index, S]

            cur_distances = vmap(_distance_one_dim, in_axes=(0, None, None))(
                kernel_distances, S, index
            )
            # cur_distances = kernel_distances[index, S]
            cur_distances = jnp.sum(cur_distances, axis=0)
            return jnp.min(cur_distances)

        distances = vmap(compute_distance, in_axes=(None, 0))(
            current_indices, potential_indices
        )
        greedy_index = jnp.argmax(distances)
        current_indices = jnp.append(current_indices, potential_indices[greedy_index])

    return current_indices, potential_indices


def greedy_largest_subdeterminant(K: np.ndarray, k: int):
    assert K.ndim == 3
    """
    :param K: Kernel matrix
    :param k: Number of samples to draw
    :return: Indices of the k samples
    """
    potential_indices = jnp.arange(K.shape[1], dtype=jnp.int32)
    current_indices = jnp.array([], dtype=jnp.int32)
    for i in range(k):

        def compute_det(S, index):
            indices = jnp.append(S, index)

            def compute_one_det(K_one, indices):
                return jnp.linalg.slogdet(K_one[jnp.ix_(indices, indices)])[1]

            output_dets = vmap(compute_one_det, in_axes=(0, None))(K, indices)
            return jnp.sum(output_dets)

        log_dets = vmap(compute_det, in_axes=(None, 0))(
            current_indices, potential_indices
        )
        greedy_index = jnp.argmax(log_dets)
        current_indices = jnp.append(current_indices, potential_indices[greedy_index])
        potential_indices = jnp.delete(potential_indices, greedy_index)

    return current_indices, potential_indices


def greedy_largest_subdeterminant_jit(K: np.ndarray, k_array):
    assert K.ndim == 3
    """
    :param K: Kernel matrix
    :param k: Number of samples to draw
    :return: Indices of the k samples
    """
    potential_indices = jnp.arange(K.shape[1], dtype=jnp.int32)
    current_indices = jnp.array([], dtype=jnp.int32)
    for _ in range(k_array.shape[0]):

        def compute_det(S, index):
            indices = jnp.append(S, index)

            def compute_one_det(K_one, indices):
                return jnp.linalg.slogdet(K_one[jnp.ix_(indices, indices)])[1]

            output_dets = vmap(compute_one_det, in_axes=(0, None))(K, indices)
            return jnp.sum(output_dets)

        log_dets = vmap(compute_det, in_axes=(None, 0))(
            current_indices, potential_indices
        )
        greedy_index = jnp.argmax(log_dets)
        current_indices = jnp.append(current_indices, potential_indices[greedy_index])

    return current_indices, potential_indices


def greedy_largest_subdeterminant_1d(K: np.ndarray, k: int):
    assert K.ndim == 2
    """
    :param K: Kernel matrix
    :param k: Number of samples to draw
    :return: Indices of the k samples
    """
    potential_indices = jnp.arange(K.shape[0], dtype=jnp.int32)
    current_indices = jnp.array([], dtype=jnp.int32)
    for i in range(k):

        def compute_det(S, index):
            indices = jnp.append(S, index)
            return jnp.linalg.slogdet(K[jnp.ix_(indices, indices)])[1]

        log_dets = vmap(compute_det, in_axes=(None, 0))(
            current_indices, potential_indices
        )
        greedy_index = jnp.argmax(log_dets)
        current_indices = jnp.append(current_indices, potential_indices[greedy_index])
        potential_indices = jnp.delete(potential_indices, greedy_index)

    return current_indices, potential_indices


def greedy_largest_subdeterminant_1d_jit(K: np.ndarray, k_array):
    assert K.ndim == 2
    """
    :param K: Kernel matrix
    :param k: Number of samples to draw
    :return: Indices of the k samples
    """
    potential_indices = jnp.arange(K.shape[0], dtype=jnp.int32)
    current_indices = jnp.array([], dtype=jnp.int32)
    for i in range(k_array.shape[0]):

        def compute_det(S, index):
            indices = jnp.append(S, index)
            return jnp.linalg.slogdet(K[jnp.ix_(indices, indices)])[1]

        log_dets = vmap(compute_det, in_axes=(None, 0))(
            current_indices, potential_indices
        )
        greedy_index = jnp.argmax(log_dets)
        current_indices = jnp.append(current_indices, potential_indices[greedy_index])

    return current_indices, potential_indices


def onedim_fun_example_dist():
    import matplotlib.pyplot as plt

    # Define a Gaussian kernel
    input_dim = 1
    gamma = 2.0
    noise_std = 0.1

    x_obs = random.uniform(
        key=random.PRNGKey(0), shape=(5,), minval=-5, maxval=5
    ).reshape(-1, 1)

    def rbf(x, y):
        assert x.shape == y.shape == (input_dim,)
        return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * gamma**2))

    kernel_v = vmap(rbf, in_axes=(0, None), out_axes=0)
    kernel_m = vmap(kernel_v, in_axes=(None, 0), out_axes=1)

    K_obs = kernel_m(x_obs, x_obs)

    def posterior_kernel(x, y):
        assert x.shape == y.shape == (input_dim,)
        return rbf(x, y) - kernel_v(x_obs, x).T @ jnp.linalg.inv(
            K_obs + noise_std * jnp.eye(K_obs.shape[0])
        ) @ kernel_v(x_obs, y)

    posterior_v = vmap(posterior_kernel, in_axes=(0, None), out_axes=0)
    posterior_m = vmap(posterior_v, in_axes=(None, 0), out_axes=1)

    x = jnp.linspace(-4, 4, 1000)

    K = posterior_m(x.reshape(-1, 1), x.reshape(-1, 1))
    K = K + noise_std * jnp.eye(K.shape[0])

    # Here we compute distance matrix based on kernel matrix K

    K = jnp.expand_dims(K, axis=0)
    out = jit(greedy_distance_maximization_jit)(K, jnp.arange(7))

    best_indices = out[0]

    initial_variances = vmap(posterior_kernel, in_axes=(0, 0))(
        x.reshape(-1, 1), x.reshape(-1, 1)
    )
    fig, axs = plt.subplots(1, 1)
    axs = np.array(axs)
    axs = axs.reshape(1, 1)
    axs[0, 0].plot(x, initial_variances, color="green")
    axs[0, 0].scatter(
        x[best_indices], jnp.zeros_like(x[best_indices]), color="red", marker="x"
    )
    test = "2"
    axs[0, 0].plot(
        x,
        jnp.zeros_like(x),
        color="blue",
        alpha=0.3,
        label=r"$\sum_{}^{} \sigma_i^2(x(t))$".format("i=0", 2),
    )
    plt.legend()
    plt.show()


def onedim_fun_example():
    import matplotlib.pyplot as plt

    x = jnp.linspace(-5, 5, 1000)

    # Define a Gaussian kernel
    input_dim = 1
    gamma = 2.0
    noise_std = 0.1

    x_obs = random.uniform(
        key=random.PRNGKey(0), shape=(5,), minval=-5, maxval=5
    ).reshape(-1, 1)

    def rbf(x, y):
        assert x.shape == y.shape == (input_dim,)
        return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * gamma**2))

    kernel_v = vmap(rbf, in_axes=(0, None), out_axes=0)
    kernel_m = vmap(kernel_v, in_axes=(None, 0), out_axes=1)

    K_obs = kernel_m(x_obs, x_obs)

    def posterior_kernel(x, y):
        assert x.shape == y.shape == (input_dim,)
        return rbf(x, y) - kernel_v(x_obs, x).T @ jnp.linalg.inv(
            K_obs + noise_std * jnp.eye(K_obs.shape[0])
        ) @ kernel_v(x_obs, y)

    posterior_v = vmap(posterior_kernel, in_axes=(0, None), out_axes=0)
    posterior_m = vmap(posterior_v, in_axes=(None, 0), out_axes=1)

    K = posterior_m(x.reshape(-1, 1), x.reshape(-1, 1))
    K = K + noise_std * jnp.eye(K.shape[0])

    out = jit(greedy_largest_subdeterminant_1d_jit)(K, jnp.arange(20))

    best_indices = out[0]

    print(
        "Max determinant: ",
        jnp.linalg.slogdet(K[np.ix_(best_indices, best_indices)])[1],
    )

    initial_variances = vmap(posterior_kernel, in_axes=(0, 0))(
        x.reshape(-1, 1), x.reshape(-1, 1)
    )

    fig, axs = plt.subplots(1, 1)
    axs = np.array(axs)
    axs = axs.reshape(1, 1)
    axs[0, 0].plot(x, initial_variances, color="green")
    axs[0, 0].scatter(
        x[best_indices], jnp.zeros_like(x[best_indices]), color="red", marker="x"
    )
    test = "2"
    axs[0, 0].plot(
        x,
        jnp.zeros_like(x),
        color="blue",
        alpha=0.3,
        label=r"$\sum_{}^{} \sigma_i^2(x(t))$".format("i=0", 2),
    )
    plt.legend()
    plt.show()


def onedim_example():
    import matplotlib.pyplot as plt

    x = jnp.linspace(0, 10, 1000)

    # Define a Gaussian kernel
    input_dim = 1
    gamma = 1.0
    noise_std = 0.1

    def rbf(x, y):
        assert x.shape == y.shape == (input_dim,)
        return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * gamma**2))

    kernel_v = vmap(rbf, in_axes=(0, None), out_axes=0)
    kernel_m = vmap(kernel_v, in_axes=(None, 0), out_axes=1)

    K = kernel_m(x.reshape(-1, 1), x.reshape(-1, 1))
    K = K + noise_std * jnp.eye(K.shape[1])

    out = jit(greedy_distance_maximization_1d_jit)(K, jnp.arange(5))

    best_indices = out[0]

    print(
        "Max determinant: ",
        jnp.linalg.slogdet(K[jnp.ix_(best_indices, best_indices)])[1],
    )

    plt.scatter(x, jnp.zeros_like(x), color="blue")
    plt.scatter(x[best_indices], jnp.zeros_like(x[best_indices]), color="red")
    plt.show()


def multidim_example():
    import matplotlib.pyplot as plt

    x = jnp.linspace(0, 10, 1000)

    # Define a Gaussian kernel
    input_dim = 1
    gamma_0 = 1.0
    gamma_1 = 1.0
    noise_std = 0.1

    def rbf(x, y):
        assert x.shape == y.shape == (input_dim,)
        return jnp.array(
            [
                jnp.exp(-jnp.sum((x - y) ** 2) / (2 * gamma_0**2)),
                jnp.exp(-jnp.sum((x - y) ** 2) / (2 * gamma_1**2)),
            ]
        )

    kernel_v = vmap(rbf, in_axes=(0, None), out_axes=1)
    kernel_m = vmap(kernel_v, in_axes=(None, 0), out_axes=2)

    K = kernel_m(x.reshape(-1, 1), x.reshape(-1, 1))
    K = K + noise_std * jnp.eye(K.shape[1])[None, ...]

    out = jit(greedy_distance_maximization_jit)(K, jnp.arange(5))

    best_indices = out[0]

    def compute_one_det(K_one, indices):
        return jnp.linalg.slogdet(K_one[jnp.ix_(indices, indices)])[1]

    print(
        "Max determinant: ",
        jnp.sum(vmap(compute_one_det, in_axes=(0, None))(K, best_indices)),
    )

    plt.scatter(x, jnp.zeros_like(x), color="blue")
    plt.scatter(x[best_indices], jnp.zeros_like(x[best_indices]), color="red")
    plt.show()


if __name__ == "__main__":
    # onedim_example()
    # multidim_example()
    # onedim_fun_example()
    onedim_fun_example_dist()
