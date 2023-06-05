from typing import Any

import jax.numpy as jnp

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.main.config import Scaling, InteractionConfig
from cucrl.online_tracker.online_tracker_ilqr import ILQROnlineTracking
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.utils.classes import TruePolicy, MPCParameters, DynamicsModel
from cucrl.utils.helper_functions import AngleNormalizer

pytree = Any


class MPCTracker:
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        scaling: Scaling,
        dynamics: AbstractDynamics,
        simulator_costs: SimulatorCostsAndConstraints,
        interaction_config: InteractionConfig,
    ):
        self.interaction_config = interaction_config
        self.angle_normalizer = AngleNormalizer(
            state_dim=x_dim,
            control_dim=u_dim,
            angles_dim=interaction_config.angles_dim,
            state_scaling=scaling.state_scaling,
        )

        self.mpc_tracker = ILQROnlineTracking(
            x_dim=x_dim,
            u_dim=u_dim,
            dynamics=dynamics,
            simulator_costs=simulator_costs,
            interaction_config=interaction_config,
        )

    def update_mpc(
        self,
        x_k,
        t_k_idx,
        tracking_data,
        mpc_params: MPCParameters,
        dynamics_model: DynamicsModel,
    ) -> TruePolicy:
        # Transform angles to [-\pi, \pi]
        x_k = self.angle_normalizer.transform_x(x_k)
        # Solve trajectory opt
        oc_trajectory = self.mpc_tracker.track_online(
            dynamics_model, x_k, mpc_params, tracking_data, t_k_idx
        )
        # Update true policy
        new_ts_idx = t_k_idx + jnp.arange(oc_trajectory.us.shape[0])
        new_true_policy = TruePolicy(ts_idx=new_ts_idx, us=oc_trajectory.us)
        return new_true_policy

    def true_policy_place_holder(self):
        num_indices = self.mpc_tracker.between_control_indices.size
        return TruePolicy(
            ts_idx=jnp.zeros(num_indices, dtype=int),
            us=jnp.zeros(shape=(num_indices, self.mpc_tracker.u_dim)),
        )
