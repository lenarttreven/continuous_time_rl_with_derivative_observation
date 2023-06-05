import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap

from cucrl.trajectory_optimization.numerical_computations.abstract_numerical_computation import (
    NumericalComputation,
)
from cucrl.utils.helper_functions import moving_window
from cucrl.utils.splines import InterpolatedUnivariateSpline, MultivariateSpline


class LocalPointSplines(NumericalComputation):
    """
    To perform numerical derivative at a certain point we fit a spline on num_points_per_spline around that point
    (num_points_per_spline // 2 points on the left, same on the right) and take a derivative at that point.
    """

    def __init__(self, num_nodes, time_horizon, num_points_per_spline=11):
        super().__init__(num_nodes=num_nodes, time_horizon=time_horizon)
        assert 4 <= num_points_per_spline and num_points_per_spline % 2 == 1
        self.num_points_per_spline = num_points_per_spline
        self.time = jnp.linspace(
            self.time_horizon[0], self.time_horizon[1], self.num_nodes
        )
        self.reshaped_time = self.prepare_reshaped_times()

    def prepare_reshaped_times(self):
        return moving_window(self.time, self.num_points_per_spline)

    def numerical_derivative(self, states: jnp.ndarray) -> jnp.ndarray:
        # First spline
        first_times = self.time[: self.num_points_per_spline]
        first_states = states[: self.num_points_per_spline, ...]
        first_der = MultivariateSpline(first_times, first_states).derivative(
            self.time[: self.num_points_per_spline // 2]
        )

        # Center spline
        reshaped_states = moving_window(states, self.num_points_per_spline)

        def _der(ts, xs):
            return (
                MultivariateSpline(ts, xs)
                .derivative(ts[self.num_points_per_spline // 2].reshape(1, 1))
                .reshape(-1)
            )

        center_der = vmap(_der)(self.reshaped_time, reshaped_states)

        # Last spline
        last_times = self.time[-self.num_points_per_spline :]
        last_states = states[-self.num_points_per_spline :, ...]
        last_der = MultivariateSpline(last_times, last_states).derivative(
            self.time[-(self.num_points_per_spline // 2) :]
        )

        return jnp.concatenate([first_der, center_der, last_der])

    def numerical_integral(self, integrand: jnp.ndarray) -> jnp.ndarray:
        integrand_spline = InterpolatedUnivariateSpline(self.time, integrand)
        return integrand_spline.integral(self.time_horizon[0], self.time_horizon[1])[0]


if __name__ == "__main__":
    test_num_nodes = 10
    test_time_horizon = (0, 10)
    test_ts = jnp.linspace(test_time_horizon[0], test_time_horizon[1], test_num_nodes)
    test_xs = jnp.sin(test_ts)
    xs_der_true = jnp.cos(test_ts)
    model = LocalPointSplines(num_nodes=test_num_nodes, time_horizon=test_time_horizon)
    xs_der_model = model.numerical_derivative(test_xs.reshape(-1, 1)).reshape(-1)

    plt.plot(test_ts, xs_der_true, label="True der")
    plt.plot(test_ts, xs_der_model, label="Model der")
    plt.legend()
    plt.show()
