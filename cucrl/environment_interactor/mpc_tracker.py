from typing import Any

import chex
import jax.numpy as jnp
from jax.lax import dynamic_update_slice_in_dim

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.environment_interactor.discretization.equidistant_discretization import EquidistantDiscretization
from cucrl.main.config import Scaling, InteractionConfig
from cucrl.online_tracker.online_tracker_ilqr import ILQROnlineTracking
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.utils.classes import TrackingData
from cucrl.utils.classes import TruePolicy, MPCParameters, DynamicsModel
from cucrl.utils.helper_functions import AngleNormalizer

pytree = Any


class MPCTracker:
    def __init__(self, x_dim: int, u_dim: int, scaling: Scaling, dynamics: AbstractDynamics,
                 simulator_costs: SimulatorCostsAndConstraints, interaction_config: InteractionConfig,
                 control_discretization: EquidistantDiscretization):
        self.control_discretization = control_discretization
        self.interaction_config = interaction_config
        self.angle_normalizer = AngleNormalizer(state_dim=x_dim, control_dim=u_dim,
                                                angles_dim=interaction_config.angles_dim,
                                                state_scaling=scaling.state_scaling)

        self.mpc_tracker = ILQROnlineTracking(x_dim=x_dim, u_dim=u_dim, dynamics=dynamics,
                                              simulator_costs=simulator_costs, interaction_config=interaction_config)

    def update_mpc(self, x_k: chex.Array, t_k: chex.Array, tracking_data: TrackingData, mpc_params: MPCParameters,
                   dynamics_model: DynamicsModel) -> TruePolicy:
        chex.assert_type(t_k, int)
        assert x_k.shape == (self.mpc_tracker.x_dim,)
        # Transform angles to [-\pi, \pi]
        x_k = self.angle_normalizer.transform_x(x_k)
        # Solve trajectory opt
        oc_trajectory = self.mpc_tracker.track_online(dynamics_model, x_k, mpc_params, tracking_data, t_k)
        # Update true policy
        us = dynamic_update_slice_in_dim(mpc_params.true_policy.us, oc_trajectory.us, t_k, axis=0)
        new_true_policy = TruePolicy(us=us)
        return new_true_policy

    def true_policy_place_holder(self):
        num_indices = self.control_discretization.continuous_times.size
        return TruePolicy(us=jnp.zeros(shape=(num_indices, self.mpc_tracker.u_dim)))
