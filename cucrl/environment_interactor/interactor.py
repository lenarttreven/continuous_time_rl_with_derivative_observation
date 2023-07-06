from abc import ABC, abstractmethod
from typing import Any, Tuple

import chex

from cucrl.utils.classes import DynamicsModel, MeasurementSelection, IntegrationCarry

pytree = Any

# Interaction is the output of the interactor and represents: [u, proposed_ts, events]
u_type = chex.Array
Interaction = Tuple[u_type, MeasurementSelection, IntegrationCarry]


class Interactor(ABC):
    @abstractmethod
    def update(self, dynamics_model: DynamicsModel, key: chex.PRNGKey):
        pass

    @abstractmethod
    def interact(self, x: chex.Array, t: chex.Array, traj_idx: chex.Array, events: IntegrationCarry) -> Interaction:
        pass
