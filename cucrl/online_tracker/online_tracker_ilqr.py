from functools import partial
from typing import Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import cond
from trajax.optimizers import ILQR, ILQRHyperparams

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.online_tracker.abstract_online_tracker import AbstractOnlineTracker
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.utils.classes import OCSolution, TrackingData, MPCParameters, DynamicsModel
from cucrl.utils.representatives import ExplorationStrategy, NumericalComputation, DynamicsTracking


class DynamicsILQR(NamedTuple):
    dynamics_model: DynamicsModel
    mpc_params: MPCParameters
    us_track: TrackingData


class CostILQR(NamedTuple):
    xs_track: jax.Array


class ILQROnlineTracking(AbstractOnlineTracker):
    def __init__(self, x_dim: int, u_dim: int, num_nodes: int, time_horizon: Tuple[float, float],
                 dynamics: AbstractDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 exploration_strategy=ExplorationStrategy.MEAN,
                 dynamics_tracking: DynamicsTracking = DynamicsTracking.MEAN):
        super().__init__(x_dim=x_dim, u_dim=u_dim, num_nodes=num_nodes, time_horizon=time_horizon,
                         dynamics=dynamics, simulator_costs=simulator_costs, numerical_method=NumericalComputation.LGL,
                         minimize_method='IPOPT', exploration_strategy=exploration_strategy)
        self.dynamics_tracking = dynamics_tracking
        self.num_total_params = (self.x_dim + self.u_dim) * self.num_nodes
        self.h = (self.time_horizon[1] - self.time_horizon[0]) / (self.num_nodes - 1)
        self.time = jnp.linspace(self.time_horizon[0], self.time_horizon[1], self.num_nodes)
        self.ilqr = ILQR(self.total_cost, self.discrete_dynamics)
        self.ilqr_hyperparams = ILQRHyperparams(maxiter=100)

    @partial(jit, static_argnums=0)
    def ode(self, x, u, dynamics_ilqr: DynamicsILQR) -> jax.Array:
        if self.dynamics_tracking == DynamicsTracking.MEAN:
            x_dot_mean = self.dynamics.mean_eval_one(dynamics_ilqr.dynamics_model, x, u)
            return x_dot_mean
        else:
            x_dot_mean, x_dot_std = self.dynamics.mean_and_std_eval_one(dynamics_ilqr.dynamics_model, x, u)
            return x_dot_mean + x_dot_std * dynamics_ilqr.mpc_params.dynamics_id.eta

    def discrete_dynamics(self, x, u, k, dynamics_ilqr: DynamicsILQR):
        return x + self.h * self.ode(x, u + dynamics_ilqr.us_track[k], dynamics_ilqr)

    def running_cost(self, x, u, k, cost_ilqr: CostILQR):
        return self.h * self.simulator_costs.tracking_running_cost(x - cost_ilqr.xs_track[k], u)

    def terminal_cost(self, x, u, k, cost_ilqr: CostILQR):
        return self.h * self.simulator_costs.tracking_terminal_cost(x - cost_ilqr.xs_track[-1], jnp.zeros(
            shape=(self.simulator_costs.control_dim,)))

    def total_cost(self, x, u, k, cost_ilqr: CostILQR):
        return cond(k == self.num_nodes - 1, self.terminal_cost, self.running_cost, x, u, k, cost_ilqr)

    def prepare_tracking_data(self, tracking_data: TrackingData, t_start):
        ts_eval = self.time + t_start
        return vmap(tracking_data)(ts_eval.reshape(-1, 1))

    def track_online(self, dynamics_model: DynamicsModel, initial_conditions: jax.Array, mpc_params: MPCParameters,
                     tracking_data: TrackingData, t_start) -> OCSolution:
        xs_track, us_track = self.prepare_tracking_data(tracking_data, t_start)
        dynamics_ilqr = DynamicsILQR(dynamics_model=dynamics_model, mpc_params=mpc_params, us_track=us_track)
        cost_ilqr = CostILQR(xs_track=xs_track)
        u_guess = jnp.zeros((self.num_nodes - 1, self.u_dim))
        out = self.ilqr.solve(cost_ilqr, dynamics_ilqr, initial_conditions, u_guess, self.ilqr_hyperparams)
        ts = jnp.linspace(0, self.time_horizon[1], self.num_nodes)
        jax.debug.print("{x}", x=out[2])
        # xs, delta_us = out['optimal_trajectory']
        xs, delta_us = out[0], out[1]
        xs_o, us_o = vmap(tracking_data)((t_start + self.time).reshape(-1, 1))
        delta_us = jnp.concatenate([delta_us, delta_us[-1].reshape(1, self.u_dim)])
        return OCSolution(ts, xs, us_o + delta_us, out[2], mpc_params.dynamics_id)
