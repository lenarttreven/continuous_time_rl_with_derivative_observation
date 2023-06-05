from abc import ABC, abstractmethod

from jax import random, numpy as jnp

from cucrl.utils.classes import DynamicsModel, CollectorCarry


class AbstractMeasurementsCollector(ABC):
    @abstractmethod
    def update(
        self, dynamics_model: DynamicsModel, key: random.PRNGKey
    ) -> CollectorCarry:
        pass

    @abstractmethod
    def apply(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        tracking_data,
        dynamics_model,
        traj_idx,
        events,
    ):
        pass
