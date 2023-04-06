import jax.numpy as jnp

from cucrl.trajectory_optimization.numerical_computations.abstract_numerical_computation import NumericalComputation
from cucrl.utils.splines import MultivariateSpline, InterpolatedUnivariateSpline


class Splines(NumericalComputation):
    def __init__(self, num_nodes, time_horizon):
        super().__init__(num_nodes=num_nodes, time_horizon=time_horizon)
        self.time = jnp.linspace(self.time_horizon[0], self.time_horizon[1], self.num_nodes)

    def numerical_derivative(self, states: jnp.ndarray) -> jnp.ndarray:
        spline = MultivariateSpline(self.time, states)
        return spline.derivative(self.time)

    def numerical_integral(self, integrand: jnp.ndarray) -> jnp.ndarray:
        integrand_spline = InterpolatedUnivariateSpline(self.time, integrand)
        return integrand_spline.integral(self.time_horizon[0], self.time_horizon[1])[0]
