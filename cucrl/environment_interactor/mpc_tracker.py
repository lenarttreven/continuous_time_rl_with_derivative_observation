from typing import Any, List

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.main.config import Scaling, OnlineTrackingConfig
from cucrl.online_tracker.online_tracker_ilqr import ILQROnlineTracking
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.utils.classes import TruePolicy, MPCParameters, DynamicsModel
from cucrl.utils.helper_functions import AngleNormalizer

pytree = Any


class MPCTracker:
    def __init__(self, state_dim: int, control_dim: int, angles_dim: List[int], scaling: Scaling,
                 dynamics: AbstractDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 online_tracking_config: OnlineTrackingConfig):
        self.angle_normalizer = AngleNormalizer(state_dim=state_dim, control_dim=control_dim, angles_dim=angles_dim,
                                                state_scaling=scaling.state_scaling)

        self.mpc_tracker = ILQROnlineTracking(x_dim=state_dim, u_dim=control_dim,
                                              num_nodes=online_tracking_config.num_nodes,
                                              time_horizon=(0, online_tracking_config.time_horizon), dynamics=dynamics,
                                              simulator_costs=simulator_costs,
                                              dynamics_tracking=online_tracking_config.dynamics_tracking)

    def update_mpc(self, cur_x, cur_t, tracking_data, mpc_params: MPCParameters,
                   dynamics_model: DynamicsModel) -> TruePolicy:
        # Transform angles to [-\pi, \pi]
        cur_x = self.angle_normalizer.transform_x(cur_x)
        # Solve trajectory opt
        oc_trajectory = self.mpc_tracker.track_online(dynamics_model, cur_x, mpc_params, tracking_data, cur_t)
        # Update true policy
        new_true_policy = TruePolicy(ts=(cur_t + oc_trajectory.ts).reshape(-1, 1), us=oc_trajectory.us)
        return new_true_policy
