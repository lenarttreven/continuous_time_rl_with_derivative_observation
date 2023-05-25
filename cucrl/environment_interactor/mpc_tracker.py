from typing import Any

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.main.config import Scaling, InteractionConfig
from cucrl.online_tracker.online_tracker_ilqr import ILQROnlineTracking
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.utils.classes import TruePolicy, MPCParameters, DynamicsModel
from cucrl.utils.helper_functions import AngleNormalizer

pytree = Any


class MPCTracker:
    def __init__(self, x_dim: int, u_dim: int, scaling: Scaling, dynamics: AbstractDynamics,
                 simulator_costs: SimulatorCostsAndConstraints, interaction_config: InteractionConfig):
        self.angle_normalizer = AngleNormalizer(state_dim=x_dim, control_dim=u_dim,
                                                angles_dim=interaction_config.angles_dim,
                                                state_scaling=scaling.state_scaling)

        self.mpc_tracker = ILQROnlineTracking(x_dim=x_dim, u_dim=u_dim, dynamics=dynamics,
                                              simulator_costs=simulator_costs, interaction_config=interaction_config)

    def update_mpc(self, cur_x, cur_t, tracking_data, mpc_params: MPCParameters,
                   dynamics_model: DynamicsModel) -> TruePolicy:
        # Transform angles to [-\pi, \pi]
        cur_x = self.angle_normalizer.transform_x(cur_x)
        # Solve trajectory opt
        oc_trajectory = self.mpc_tracker.track_online(dynamics_model, cur_x, mpc_params, tracking_data, cur_t)
        # Update true policy
        new_true_policy = TruePolicy(ts=(cur_t + oc_trajectory.ts).reshape(-1, 1), us=oc_trajectory.us)
        return new_true_policy
