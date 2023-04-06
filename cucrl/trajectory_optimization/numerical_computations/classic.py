import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from scipy import special

from cucrl.trajectory_optimization.numerical_computations.abstract_numerical_computation import NumericalComputation
from cucrl.utils.helper_functions import derivative_coefficients


class Classic(NumericalComputation):
    def __init__(self, num_nodes, time_horizon, k=2):
        super().__init__(num_nodes=num_nodes, time_horizon=time_horizon)
        self.time = jnp.linspace(self.time_horizon[0], self.time_horizon[1], self.num_nodes)
        self.k = k
        self.h = self.time[1] - self.time[0]
        self.der_matrix = jnp.array(self.derivative_matrix(self.num_nodes, self.h, self.k))
        self.integral_vector = self._integral_vector(self.num_nodes, self.h)
        self.der_coeffs = derivative_coefficients(k)

        weights = self._weight_lgl(num_nodes)
        self._weights = jnp.array(weights)

    @staticmethod
    def _integral_vector(n, h):
        vec = 2 * jnp.ones(shape=(n,), dtype=jnp.float64)
        vec = vec.at[0].set(1.0)
        vec = vec.at[-1].set(1.0)
        return h * 0.5 * vec

    @staticmethod
    def derivative_matrix(n, h, k):
        center_start = (k + 1) // 2
        der_coeffs = derivative_coefficients(k)
        # First rows
        first_rows = np.zeros(shape=(center_start, n))
        for i in range(center_start):
            first_rows[i, :k + 1] = der_coeffs[i, ...]
        # Center
        center = np.zeros(shape=(n - 2 * center_start, n))
        rows, cols = np.indices((n - 2 * center_start, n))
        for i in range(k + 1):
            row_vals, col_vals = np.diag(rows, k=i), np.diag(cols, k=i)
            center[row_vals, col_vals] = der_coeffs[center_start, i]
        # Last rows
        last_rows = np.zeros(shape=(center_start, n))
        for i in range(1, center_start + 1):
            last_rows[-i, -(k + 1):] = der_coeffs[-i, ...]
        full_matrix = np.concatenate([first_rows, center, last_rows])
        return full_matrix / h

    def _derivative_one_dim(self, states: jnp.ndarray) -> jnp.ndarray:
        assert states.ndim == 1
        return self.der_matrix @ states

    def numerical_derivative(self, states: jnp.ndarray) -> jnp.ndarray:
        assert states.ndim == 2
        return vmap(self._derivative_one_dim, in_axes=1, out_axes=1)(states)

    # def numerical_integral(self, integrand: jnp.ndarray) -> jnp.ndarray:
    #     assert integrand.ndim == 1
    #     return self.integral_vector @ integrand

    # def numerical_integral(self, integrand: jnp.ndarray) -> jnp.ndarray:
    #     integrand_spline = InterpolatedUnivariateSpline(self.time, integrand)
    #     return integrand_spline.integral(self.time_horizon[0], self.time_horizon[1])[0]

    def numerical_integral(self, integrand):
        return jnp.sum(integrand * self._weights) * (self.time_horizon[1] - self.time_horizon[0]) / 2

    @staticmethod
    def _nodes_lgl(n):
        roots, weight = special.j_roots(n - 2, 1, 1)
        nodes = np.hstack((-1, roots, 1))
        return nodes

    def _weight_lgl(self, n):
        nodes = self._nodes_lgl(n)
        w = np.zeros(0)
        for i in range(n):
            w = np.append(w, 2 / (n * (n - 1) * self._legendre_function(nodes[i], n - 1) ** 2))
        return w

    @staticmethod
    def _legendre_function(x, n):
        legendre, derivative = special.lpn(n, x)
        return legendre[-1]


if __name__ == '__main__':
    num_nodes = 100
    time_horizon = (0.0, jnp.pi)
    ts = jnp.linspace(time_horizon[0], time_horizon[1], num_nodes)
    xs = jnp.sin(ts)
    xs_der_true = jnp.cos(ts)

    model = Classic(num_nodes=num_nodes, time_horizon=time_horizon)
    xs_der_num = model.numerical_derivative(xs.reshape(-1, 1)).reshape(-1)
    plt.plot(ts, xs_der_true, label='True der')
    plt.plot(ts, xs_der_num, label='Num der')
    plt.legend()
    plt.show()

    integral = model.numerical_integral(xs)
    print(integral)
    print(2 * model.h * model.der_coeffs)
