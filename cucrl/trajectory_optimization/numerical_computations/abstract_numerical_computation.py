from abc import ABC, abstractmethod
from typing import Tuple

import jax.numpy as jnp


class NumericalComputation(ABC):
    def __init__(self, num_nodes: int, time_horizon: Tuple[float, float]):
        self.num_nodes = num_nodes
        self.time_horizon = time_horizon

    @abstractmethod
    def numerical_integral(self, integrand: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def numerical_derivative(self, states: jnp.ndarray) -> jnp.ndarray:
        pass
