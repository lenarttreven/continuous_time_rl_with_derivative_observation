from typing import Callable

import chex
from jax.lax import scan
from jaxtyping import PyTree

from cucrl.environment_interactor.discretization.equidistant_discretization import EquidistantDiscretization

DynamicsFn = Callable[[PyTree, chex.Array, chex.Array], chex.Array]


class Euler(EquidistantDiscretization):
    def __init__(self, num_integration_steps: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_integration_steps = num_integration_steps

    def discretize(self, dynamics_fn: DynamicsFn) -> DynamicsFn:
        """
        Discretizes the dynamics function using the Euler method
        :param dynamics_fn: ode function to discretize
        :return:
        """

        def discretized_dynamics_fn(params: PyTree, x: chex.Array, u: chex.Array) -> chex.Array:
            _dt = self.dt / self.num_integration_steps

            def f(_x, ):
                x_dot = dynamics_fn(params, _x, u)
                _x_next = _x + _dt * x_dot
                return _x_next, None

            x_next = scan(f, x, xs=None, length=self.num_integration_steps)[0]
            return x_next

        return discretized_dynamics_fn
