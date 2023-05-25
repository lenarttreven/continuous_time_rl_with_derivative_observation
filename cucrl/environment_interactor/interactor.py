from abc import ABC, abstractmethod
from typing import Any, Tuple

import chex
from jax import random
from jax.tree_util import register_pytree_node_class

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.environment_interactor.measurements_collector.measurements_collector import MeasurementsCollector
from cucrl.environment_interactor.policy.mpc_tracking import MPCTracking
from cucrl.main.config import InteractionConfig, Scaling
from cucrl.offline_planner.abstract_offline_planner import AbstractOfflinePlanner
from cucrl.utils.classes import DynamicsModel, IntegrationCarry, MeasurementSelection
from cucrl.utils.helper_functions import AngleLayerDynamics

pytree = Any

# Interaction is the output of the interactor and represents: [u, proposed_ts, events]
u_type = chex.Array
Interaction = Tuple[u_type, MeasurementSelection, IntegrationCarry]


class Interactor(ABC):
    @abstractmethod
    def update(self, dynamics_model: DynamicsModel, key: random.PRNGKey):
        pass

    @abstractmethod
    def interact(self, x: chex.Array, t: chex.Array, traj_idx, events) -> Interaction:
        pass


@register_pytree_node_class
class MPCInteractor(Interactor):
    def __init__(self, x_dim, u_dim, dynamics: AbstractDynamics, x0s, normalizer, angle_layer: AngleLayerDynamics,
                 interaction_config: InteractionConfig, offline_planner: AbstractOfflinePlanner, scaling: Scaling):
        self.policy = MPCTracking(x_dim, u_dim, dynamics, x0s, normalizer, angle_layer,
                                  interaction_config, offline_planner, scaling)

        self.measurements_collector = MeasurementsCollector(x_dim, u_dim, dynamics, interaction_config,
                                                            offline_planner, scaling, len(x0s))

        self.dynamics_model = DynamicsModel()
        self.tracking_data = None
        self.offline_planning_data = None
        self.integration_carry = None

    def tree_flatten(self):
        children = (self.dynamics_model, self.tracking_data, self.key,
                    self.integration_carry, self.offline_planning_data)
        aux_data = {'state_dim': self.state_dim, 'control_dim': self.control_dim, 'dynamics': self.dynamics,
                    'initial_conditions': self.initial_conditions, 'normalizer': self.normalizer,
                    'angle_layer': self.angle_layer, 'interaction_config': self.interaction_config,
                    'offline_planner': self.offline_planner, 'scaling': self.scaling}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        dynamics_model, tracking_data, key, integration_carry, offline_planning_data = children
        new_class = cls(x_dim=aux_data['state_dim'], u_dim=aux_data['control_dim'],
                        dynamics=aux_data['dynamics'], x0s=aux_data['initial_conditions'],
                        normalizer=aux_data['normalizer'], angle_layer=aux_data['angle_layer'],
                        interaction_config=aux_data['interaction_config'], offline_planner=aux_data['offline_planner'],
                        scaling=aux_data['scaling'])
        new_class.dynamics_model = dynamics_model
        new_class.tracking_data = tracking_data
        new_class.key = key
        new_class.integration_carry = integration_carry
        new_class.offline_planning_data = offline_planning_data
        return new_class

    def update(self, dynamics_model: DynamicsModel, key: random.PRNGKey):
        key, subkey = random.split(key)
        mpc_carry, offline_planning_data, tracking_data = self.policy.update(dynamics_model, key)
        collector_carry = self.measurements_collector.update(dynamics_model, subkey)
        self.integration_carry = IntegrationCarry(mpc_carry=mpc_carry, collector_carry=collector_carry)
        self.offline_planning_data = offline_planning_data
        self.tracking_data = tracking_data
        self.dynamics_model = dynamics_model

    def interact(self, x: chex.Array, t: chex.Array, traj_idx, events) -> Interaction:
        u, new_events = self.policy.apply(x, t, self.tracking_data, self.dynamics_model, traj_idx, events)
        measurement_selection, new_events = self.measurements_collector.apply(x, t, self.tracking_data,
                                                                              self.dynamics_model, traj_idx, new_events)
        return u, measurement_selection, new_events
