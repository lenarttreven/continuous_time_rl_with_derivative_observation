from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import cond
from trajax.optimizers import ILQR, ILQRHyperparams

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.main.config import InteractionConfig
from cucrl.online_tracker.abstract_online_tracker import AbstractOnlineTracker
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.utils.classes import OCSolution, TrackingData, MPCParameters, DynamicsModel, DynamicsIdentifier
from cucrl.utils.representatives import DynamicsTracking


class DynamicsILQR(NamedTuple):
    dynamics_model: DynamicsModel
    mpc_params: MPCParameters
    us_track: TrackingData
    ts_track: chex.Array


class CostILQR(NamedTuple):
    xs_track: chex.Array
    ts_track: chex.Array
    dynamics_model: DynamicsModel


class ILQROnlineTracking(AbstractOnlineTracker):
    def __init__(self, x_dim: int, u_dim: int, dynamics: AbstractDynamics,
                 simulator_costs: SimulatorCostsAndConstraints, interaction_config: InteractionConfig, ):
        super().__init__(x_dim=x_dim, u_dim=u_dim, time_horizon=interaction_config.time_horizon, dynamics=dynamics,
                         simulator_costs=simulator_costs)
        self.interaction_config = interaction_config
        # Setup time
        ts_nodes = jnp.linspace(*interaction_config.time_horizon, interaction_config.policy.num_nodes + 1)
        policy_config = interaction_config.policy
        self.num_nodes = jnp.sum(ts_nodes <= policy_config.online_tracking.time_horizon)
        self.ts = ts_nodes[:self.num_nodes]
        self.all_ts = ts_nodes
        self.between_control_ts = jnp.linspace(self.all_ts[0], self.all_ts[1], policy_config.num_int_step_between_nodes)

        policy_config = interaction_config.policy
        total_time = interaction_config.time_horizon[1] - interaction_config.time_horizon[0]
        total_int_steps = policy_config.num_nodes * policy_config.num_int_step_between_nodes
        self.dt = total_time / total_int_steps

        self.between_control_indices = jnp.arange(self.num_nodes)

        # Setup optimizer
        self.ilqr = ILQR(self.cost_fn, self.dynamics_fn)
        self.ilqr_hyperparams = ILQRHyperparams(maxiter=100)
        self.u_init = jnp.zeros((self.num_nodes - 1, self.u_dim))

    def example_dynamics_id(self) -> DynamicsIdentifier:
        return DynamicsIdentifier(eta=jnp.ones(shape=(self.num_nodes, self.x_dim)),
                                  idx=jnp.ones(shape=(), dtype=jnp.int32),
                                  key=jnp.ones(shape=(2,), dtype=jnp.int32))

    def example_oc_solution(self) -> OCSolution:
        return OCSolution(ts=self.ts, xs=jnp.ones(shape=(self.num_nodes, self.x_dim)),
                          us=jnp.ones(shape=(self.num_nodes, self.u_dim)), opt_value=jnp.ones(shape=()),
                          dynamics_id=self.example_dynamics_id())

    def ode(self, x: chex.Array, u: chex.Array, dynamics_model: DynamicsModel) -> chex.Array:
        if self.interaction_config.policy.online_tracking.dynamics_tracking == DynamicsTracking.MEAN:
            x_dot_mean = self.dynamics.mean_eval_one(dynamics_model, x, u)
            return x_dot_mean
        else:
            raise NotImplementedError(f"Unknown dynamics tracking: "
                                      f"{self.interaction_config.policy.online_tracking.dynamics_tracking}")

    def dynamics_fn(self, x_k: chex.Array, u_k: chex.Array, k: chex.Array, dynamics_ilqr: DynamicsILQR):
        # t will go over array [0, ..., num_nodes - 1]
        assert x_k.shape == (self.x_dim,) and u_k.shape == (self.u_dim,) and k.shape == ()
        chex.assert_type(k, int)
        # dynamics_ilqr.us_track is a (num_nodes - 1, u_dim) array

        init_time = dynamics_ilqr.ts_track[k]
        cur_ts = init_time + self.between_control_ts

        def _next_step(x: chex.Array, t: chex.Array) -> Tuple[chex.Array, chex.Array]:
            x_dot = self.ode(x, u_k + dynamics_ilqr.us_track[k], dynamics_ilqr.dynamics_model)
            x_next = x + self.dt * x_dot
            return x_next, x_next

        x_k_next, _ = jax.lax.scan(_next_step, x_k, cur_ts)
        return x_k_next

    def cost_fn(self, x_k, u_k, k, cost_ilqr: CostILQR):
        assert x_k.shape == (self.x_dim,) and u_k.shape == (self.u_dim,) and k.shape == ()
        chex.assert_type(k, int)

        def running_cost(x, u, t):
            init_time = cost_ilqr.ts_track[k]
            cur_ts = init_time + self.between_control_ts

            def _next_step(_x: chex.Array, _t: chex.Array) -> Tuple[chex.Array, chex.Array]:
                x_dot = self.ode(_x, u, cost_ilqr.dynamics_model)
                x_next = _x + self.dt * x_dot
                return x_next, self.dt * self.simulator_costs.running_cost(_x, u)

            x_last, cs = jax.lax.scan(_next_step, x, cur_ts)
            assert cs.shape == (self.interaction_config.policy.num_int_step_between_nodes,)
            return jnp.sum(cs)

        def terminal_cost(x, u, t):
            return self.simulator_costs.terminal_cost(x, u)

        return cond(k == self.num_nodes - 1, terminal_cost, running_cost, x_k, u_k, k)

    def prepare_tracking_data(self, tracking_data: TrackingData, t_start):
        t_indices = t_start + self.between_control_indices
        return vmap(tracking_data)(t_indices)

    def track_online(self, dynamics_model: DynamicsModel, initial_conditions: chex.Array, mpc_params: MPCParameters,
                     tracking_data: TrackingData, t_start_idx: chex.Array) -> OCSolution:
        assert t_start_idx.shape == ()
        chex.assert_type(t_start_idx, int)
        # xs_track, us_track are of shape (num_nodes, x_dim) and (num_nodes - 1, u_dim) respectively
        xs_track, us_track, ts_track = self.prepare_tracking_data(tracking_data, t_start_idx)

        dynamics_ilqr = DynamicsILQR(dynamics_model=dynamics_model, mpc_params=mpc_params, us_track=us_track,
                                     ts_track=ts_track, )
        cost_ilqr = CostILQR(xs_track=xs_track, ts_track=ts_track, dynamics_model=dynamics_model)

        results = self.ilqr.solve(cost_ilqr, dynamics_ilqr, initial_conditions, self.u_init, self.ilqr_hyperparams)

        jax.debug.print("Objective value: {x}", x=results.obj)
        xs, delta_us = results.xs, results.us

        xs_o, us_o = xs_track, us_track
        delta_us = jnp.concatenate([delta_us, delta_us[-1].reshape(1, self.u_dim)])
        return OCSolution(ts_track, xs, us_o + delta_us, results.obj, mpc_params.dynamics_id)
