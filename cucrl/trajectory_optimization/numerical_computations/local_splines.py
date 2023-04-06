import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap

from cucrl.trajectory_optimization.numerical_computations.abstract_numerical_computation import NumericalComputation
from cucrl.utils.splines import MultivariateSpline, InterpolatedUnivariateSpline


class LocalSplines(NumericalComputation):
    """
     To perform numerical derivative we split trajectory in chunks of size num_points_per_spline, fit a spline through
     every chunk and take a derivate of that chunk
    """

    def __init__(self, num_nodes, time_horizon, num_points_per_spline=5):
        super().__init__(num_nodes=num_nodes, time_horizon=time_horizon)
        assert 4 <= num_points_per_spline and num_nodes % num_points_per_spline == 0
        self.num_points_per_spline = num_points_per_spline
        self.num_splines = self.num_nodes // self.num_points_per_spline
        self.time = jnp.linspace(self.time_horizon[0], self.time_horizon[1], self.num_nodes)
        self.reshaped_time = self.time.reshape(self.num_splines, self.num_points_per_spline)

    def numerical_derivative(self, states: jnp.ndarray) -> jnp.ndarray:
        state_dim = states.shape[1]
        assert states.ndim == 2 and states.shape == (self.num_nodes, state_dim)
        # Reshape
        reshaped_states = states.reshape(self.num_splines, self.num_points_per_spline, state_dim)

        # Fit plenty of splines and take derivative
        def _der(states, times):
            return MultivariateSpline(times, states).derivative(times)

        reshaped_ders = vmap(_der, in_axes=(0, 0))(reshaped_states, self.reshaped_time)
        return reshaped_ders.reshape(self.num_nodes, state_dim)

    def numerical_integral(self, integrand: jnp.ndarray) -> jnp.ndarray:
        integrand_spline = InterpolatedUnivariateSpline(self.time, integrand)
        return integrand_spline.integral(self.time_horizon[0], self.time_horizon[1])[0]


if __name__ == '__main__':
    num_nodes = 50
    time_horizon = (0, 10)
    ts = jnp.linspace(time_horizon[0], time_horizon[1], num_nodes)
    xs = jnp.sin(ts)
    xs_der_true = jnp.cos(ts)
    model = LocalSplines(num_nodes=num_nodes, time_horizon=time_horizon)
    xs_der_model = model.numerical_derivative(xs.reshape(-1, 1)).reshape(-1)

    plt.plot(ts, xs_der_true, label='True der')
    plt.plot(ts, xs_der_model, label='Model der')
    plt.legend()
    plt.show()
