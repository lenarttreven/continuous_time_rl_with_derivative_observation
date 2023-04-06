import jax.numpy as jnp
import numpy as np
from scipy import special

from cucrl.trajectory_optimization.numerical_computations.abstract_numerical_computation import NumericalComputation


class LegendreGaussLobatto(NumericalComputation):
    def __init__(self, num_nodes, time_horizon):
        super().__init__(num_nodes=num_nodes, time_horizon=time_horizon)
        nodes, weights, differentiation_matrix = self.method_lgl(num_nodes)
        self._nodes = jnp.array(nodes)
        self._weights = jnp.array(weights)
        self._differentiation_matrix = jnp.array(differentiation_matrix)
        self.time = (self.time_horizon[1] - self.time_horizon[0]) * 0.5 * self._nodes + (
                self.time_horizon[1] + self.time_horizon[0]) * 0.5

    def numerical_derivative(self, states):
        return self._differentiation_matrix.dot(states) / ((self.time_horizon[1] - self.time_horizon[0]) * 0.5)

    def numerical_integral(self, integrand):
        return jnp.sum(integrand * self._weights) * (self.time_horizon[1] - self.time_horizon[0]) / 2

    @staticmethod
    def _legendre_function(x, n):
        legendre, derivative = special.lpn(n, x)
        return legendre[-1]

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

    def _differentiation_matrix_lgl(self, n):
        tau = self._nodes_lgl(n)
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    d[i, j] = self._legendre_function(tau[i], n - 1) \
                              / self._legendre_function(tau[j], n - 1) \
                              / (tau[i] - tau[j])
                elif i == j and i == 0:
                    d[i, j] = -n * (n - 1) * 0.25
                elif i == j and i == n - 1:
                    d[i, j] = n * (n - 1) * 0.25
                else:
                    d[i, j] = 0.0
        return d

    def method_lgl(self, n):
        nodes = self._nodes_lgl(n)
        weight = self._weight_lgl(n)
        d = self._differentiation_matrix_lgl(n)
        return nodes, weight, d
