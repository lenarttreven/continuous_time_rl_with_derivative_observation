import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap, random
from jax.config import config
from jax.scipy.linalg import cho_factor, cho_solve

config.update("jax_enable_x64", True)


# Need to vectorize this to be able to handle multiple outputs (it can take several inputs already)


def select_most_uncertain_point(xs_history: jax.Array, xs_potential, kernel, noise_std):
    kernel_v = vmap(kernel, in_axes=(0, None), out_axes=1)
    kernel_m = vmap(kernel_v, in_axes=(None, 0), out_axes=2)

    k_XX = (
        kernel_m(xs_history, xs_history)
        + jnp.eye(xs_history.shape[0])[None, ...] * noise_std**2
    )
    cho_factor_k_XX = vmap(cho_factor, in_axes=0)(k_XX)

    def uncertainty_one_point(x):
        k_Xx = kernel_v(xs_history, x)
        k_xx = kernel(x, x)
        return k_xx - vmap(jnp.dot)(
            k_Xx,
            vmap(cho_solve, in_axes=((0, None), 0))((cho_factor_k_XX[0], False), k_Xx),
        )

    uncertainties = vmap(uncertainty_one_point)(xs_potential)
    squared_sum_uncertainties = jnp.sum(uncertainties, axis=1)
    most_uncertain_index = jnp.argmax(squared_sum_uncertainties)
    return xs_potential[most_uncertain_index], most_uncertain_index, uncertainties


def select_most_uncertain_points(
    xs_history: jax.Array, xs_potential, kernel, noise_std, num_points
):
    uncertain_points = []
    for i in range(num_points):
        uncertain_point, index, uncertainties = select_most_uncertain_point(
            xs_history, xs_potential, kernel, noise_std
        )

        plt.title("Iteration {}".format(i + 1))
        plt.scatter(xs_history, jnp.zeros_like(xs_history), color="red")
        plt.scatter(uncertain_point, 0, color="green")
        plt.plot(xs_potential, uncertainties[:, 0], color="blue")
        plt.plot(xs_potential, uncertainties[:, 1], color="blue")
        plt.plot(xs_potential, jnp.sum(uncertainties, axis=1), color="orange")
        plt.show()

        uncertain_points.append(uncertain_point)
        xs_potential = jnp.delete(xs_potential, index, axis=0)
        xs_history = jnp.concatenate(
            [xs_history, uncertain_point.reshape(1, -1)], axis=0
        )
    return jnp.stack(uncertain_points, axis=0), xs_history, xs_potential


if __name__ == "__main__":
    input_dim = 1
    output_dim = 2
    key = random.PRNGKey(0)
    xs_history = random.uniform(key, shape=(3, 1))
    noise_std = 0.1

    xs_potential = jnp.linspace(-1, 1, 100).reshape(-1, 1)

    def k(x1, x2):
        assert x1.shape == x2.shape == (input_dim,)
        var = jnp.array(
            [
                jnp.exp(-0.5 * jnp.linalg.norm(x1 - x2) ** 2 / 0.5**2),
                jnp.exp(-0.5 * jnp.linalg.norm(x1 - x2) ** 2 / 0.3**2),
            ]
        )
        assert var.shape == (output_dim,)
        return var

    out = k(xs_potential[0], xs_potential[1])

    sampling_points = select_most_uncertain_points(
        xs_history, xs_potential, k, noise_std, 20
    )
