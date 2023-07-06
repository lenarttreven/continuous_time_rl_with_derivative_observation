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


class TrackingParams(NamedTuple):
    dynamics_model: DynamicsModel
    mpc_params: MPCParameters
    tracking_data: TrackingData
    t_idx_start: chex.Array


class ILQROnlineTracking(AbstractOnlineTracker):
    def __init__(self, x_dim: int, u_dim: int, dynamics: AbstractDynamics,
                 simulator_costs: SimulatorCostsAndConstraints, interaction_config: InteractionConfig):
        super().__init__(x_dim=x_dim, u_dim=u_dim, time_horizon=interaction_config.time_horizon, dynamics=dynamics,
                         simulator_costs=simulator_costs)
        self.interaction_config = interaction_config

        policy_config = interaction_config.policy
        total_time = interaction_config.time_horizon.length()
        total_int_steps = policy_config.num_control_steps * policy_config.num_int_step_between_nodes
        self.dt = total_time / total_int_steps
        self.large_dt = total_time / policy_config.num_control_steps

        self.control_steps = self.interaction_config.policy.online_tracking.control_steps
        self.ts_indices = jnp.arange(self.control_steps)

        self.between_step_indices = jnp.arange(policy_config.num_int_step_between_nodes)

        # Setup optimizer
        self.ilqr = ILQR(self.cost_fn, self.dynamics_fn)
        self.ilqr_hyperparams = ILQRHyperparams(maxiter=100)
        self.u_init = jnp.zeros((self.control_steps, self.u_dim))

    def example_dynamics_id(self) -> DynamicsIdentifier:
        return DynamicsIdentifier(eta=jnp.ones(shape=(self.control_steps, self.x_dim)),
                                  idx=jnp.ones(shape=(), dtype=jnp.int32),
                                  key=jnp.ones(shape=(2,), dtype=jnp.int32))

    def example_oc_solution(self) -> OCSolution:
        return OCSolution(ts=self.ts_indices.astype(float), xs=jnp.ones(shape=(self.control_steps, self.x_dim)),
                          us=jnp.ones(shape=(self.control_steps, self.u_dim)), opt_value=jnp.ones(shape=()),
                          dynamics_id=self.example_dynamics_id())

    def ode(self, x: chex.Array, u: chex.Array, dynamics_model: DynamicsModel) -> chex.Array:
        if self.interaction_config.policy.online_tracking.dynamics_tracking == DynamicsTracking.MEAN:
            x_dot_mean = self.dynamics.mean_eval_one(dynamics_model, x, u)
            return x_dot_mean
        else:
            raise NotImplementedError(f'Unknown dynamics tracking: '
                                      f'{self.interaction_config.policy.online_tracking.dynamics_tracking}')

    def dynamics_fn(self, x_k: chex.Array, delta_u_k: chex.Array, k: chex.Array, tracking_params: TrackingParams):
        # t will go over array [0, ..., num_nodes - 1]
        assert x_k.shape == (self.x_dim,) and delta_u_k.shape == (self.u_dim,) and k.shape == ()
        chex.assert_type(k, int)

        indices = k + self.between_step_indices
        tracking_k = k + tracking_params.t_idx_start

        def _next_step(x: chex.Array, _: chex.Array) -> Tuple[chex.Array, chex.Array]:
            tracking_u_k = tracking_params.tracking_data(tracking_k)[1]
            x_dot = self.ode(x, delta_u_k + tracking_u_k, tracking_params.dynamics_model)
            x_next = x + self.dt * x_dot
            return x_next, x_next

        x_k_next, _ = jax.lax.scan(_next_step, x_k, indices)
        return x_k_next

    def cost_fn(self, x_k, delta_u_k, k, tracking_params: TrackingParams):
        assert x_k.shape == (self.x_dim,) and delta_u_k.shape == (self.u_dim,) and k.shape == ()
        chex.assert_type(k, int)
        tracking_k = k + tracking_params.t_idx_start

        def running_cost(_x_k, _delta_u_k, _t_k):
            x_tracking = tracking_params.tracking_data(_t_k)[0]
            x_error = _x_k - x_tracking
            return self.large_dt * self.simulator_costs.tracking_running_cost(x_error, _delta_u_k)

        def terminal_cost(_x_k, _delta_u_k, _t_k):
            x_tracking = tracking_params.tracking_data(_t_k)[0]
            x_error = _x_k - x_tracking
            return self.simulator_costs.tracking_terminal_cost(x_error, _delta_u_k)

        return cond(k == self.control_steps, terminal_cost, running_cost, x_k, delta_u_k, tracking_k)

    def prepare_tracking_data(self, tracking_data: TrackingData, t_start):
        t_indices = t_start + self.ts_indices
        return vmap(tracking_data)(t_indices)

    def track_online(self, dynamics_model: DynamicsModel, initial_conditions: chex.Array, mpc_params: MPCParameters,
                     tracking_data: TrackingData, t_start_idx: chex.Array) -> OCSolution:
        assert t_start_idx.shape == ()
        chex.assert_type(t_start_idx, int)

        tracking_params = TrackingParams(dynamics_model=dynamics_model,
                                         mpc_params=mpc_params,
                                         tracking_data=tracking_data,
                                         t_idx_start=t_start_idx)

        results = self.ilqr.solve(tracking_params, tracking_params, initial_conditions, self.u_init,
                                  self.ilqr_hyperparams)

        jax.debug.print('Objective value: {x}', x=results.obj)
        xs, delta_us = results.xs, results.us
        ts_indices = t_start_idx + jnp.arange(self.control_steps + 1)
        us, ts = vmap(tracking_data)(ts_indices)[1:]
        delta_us = jnp.concatenate([delta_us, delta_us[-1:]], axis=0)

        return OCSolution(ts, xs, us + delta_us, results.obj, mpc_params.dynamics_id)
