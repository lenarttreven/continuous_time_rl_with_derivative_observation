import chex
from jax.lax import scan
from jaxtyping import PyTree

from cucrl.environment_interactor.discretization.equidistant_discretization import EquidistantDiscretization
from cucrl.environment_interactor.discretization.equidistant_discretization import TimeHorizon, DynamicsFn


class RungeKutta(EquidistantDiscretization):
    def __init__(self, time_horizon: TimeHorizon, num_control_steps: int,
                 buffer_control_steps: int = 1, num_int_step_between_nodes: int = 1):
        super().__init__(time_horizon, num_control_steps, buffer_control_steps)
        self.num_int_step_between_nodes = num_int_step_between_nodes


    def discretize(self, dynamics_fn: DynamicsFn) -> DynamicsFn:
        """
        Discretizes the dynamics function using the Euler method
        :param dynamics_fn: ode function to discretize
        :return:
        """

        def discretized_dynamics_fn(params: PyTree, x: chex.Array, u: chex.Array) -> chex.Array:
            _dt = self.dt / self.num_int_step_between_nodes

            def f(_x, _):
                k1 = _dt * dynamics_fn(params, _x, u)
                k2 = _dt * dynamics_fn(params, _x + k1 / 2, u)
                k3 = _dt * dynamics_fn(params, _x + k2 / 2, u)
                k4 = _dt * dynamics_fn(params, _x + k3, u)
                _x_next = _x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                return _x_next, None

            x_next = scan(f, x, xs=None, length=self.num_int_step_between_nodes)[0]
            return x_next

        return discretized_dynamics_fn
