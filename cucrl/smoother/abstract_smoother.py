from abc import ABC, abstractmethod
from typing import Any

import jax
from flax.core import FrozenDict

from cucrl.main.data_stats import DataStats
from cucrl.utils.classes import SmootherApply, SmootherPosterior, SampledData

pytree = Any


class AbstractSmoother(ABC):
    def __init__(self, state_dim: int, numerical_correction: float = 1e-3):
        self.state_dim = state_dim
        self.numerical_correction = numerical_correction

    @abstractmethod
    def apply(self, parameters: pytree, stats: FrozenDict, observation_times: jax.Array, matching_times: jax.Array,
              ic_for_observation_times: jax.Array, ic_for_matching_times: jax.Array, observations: jax.Array,
              key: jax.Array, data_stats: DataStats, num_train_points: int) -> SmootherApply:
        pass

    @abstractmethod
    def initialize_parameters(self, key):
        pass

    @abstractmethod
    def posterior(self, parameters: pytree, stats: FrozenDict, evaluation_times: jax.Array,
                  ic_for_evaluation_times: jax.Array, observation_times, ic_for_observation_times, observations,
                  data_stats: DataStats) -> SmootherPosterior:
        pass

    @abstractmethod
    def sample_vector_field_data(self, params, stats, observation_times, observations, ic_for_observation_times,
                                 data_stats: DataStats, key: jax.Array) -> SampledData:
        pass
