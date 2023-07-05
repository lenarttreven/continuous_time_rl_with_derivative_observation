from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
from jax import vmap
from jax.lax import cond
from trajax.optimizers import ILQRHyperparams, CEMHyperparams, ILQR_with_CEM_warmstart, ILQR

from cucrl.main.config import PolicyConfig, TimeHorizon
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.simulator.simulator_dynamics import SimulatorDynamics
from cucrl.utils.classes import OCSolution, DynamicsIdentifier
from cucrl.utils.classes import TrackingData
from cucrl.utils.representatives import MinimizationMethod


class TrackingParams(NamedTuple):
    tracking_data: TrackingData
    t_idx_start: chex.Array


class OnlineTracking:
    def __init__(self, x_dim: int, u_dim: int, dynamics: SimulatorDynamics,
                 simulator_costs: SimulatorCostsAndConstraints, policy_config: PolicyConfig,
                 time_horizon: TimeHorizon):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.dynamics = dynamics
        self.costs = simulator_costs

        policy_config = policy_config

        # Setup time
        total_time = time_horizon.length()
        total_int_steps = policy_config.num_control_steps * policy_config.num_int_step_between_nodes
        self.dt = total_time / total_int_steps

        self.control_steps = policy_config.online_tracking.control_steps
        self.between_step_indices = jnp.arange(policy_config.num_int_step_between_nodes)

        # Setup optimizer
        self.minimize_method = policy_config.offline_planning.minimization_method
        if self.minimize_method == MinimizationMethod.ILQR_WITH_CEM:
            self.cem_params = CEMHyperparams(max_iter=10, sampling_smoothing=0.0, num_samples=200,
                                             evolution_smoothing=0.0,
                                             elite_portion=0.1)
            self.ilqr_params = ILQRHyperparams(maxiter=100)
            self.control_low = -10.0 * jnp.ones(shape=(self.u_dim,))
            self.control_high = 10.0 * jnp.ones(shape=(self.u_dim,))
            self.optimizer = ILQR_with_CEM_warmstart(self.cost_fn, self.dynamics_fn)

        elif self.minimize_method == MinimizationMethod.ILQR:
            self.ilqr_params = ILQRHyperparams(maxiter=100)
            self.optimizer = ILQR(self.cost_fn, self.dynamics_fn)

        self.u_init = jnp.zeros((self.control_steps, self.u_dim))

    def example_dynamics_id(self) -> DynamicsIdentifier:
        return DynamicsIdentifier(eta=jnp.ones(shape=(self.control_steps, self.x_dim)),
                                  idx=jnp.ones(shape=(), dtype=jnp.int32),
                                  key=jnp.ones(shape=(2,), dtype=jnp.int32))

    def ode(self, x: chex.Array, u: chex.Array) -> chex.Array:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        return self.dynamics.dynamics(x, u, jnp.zeros(shape=(1,)))

    def dynamics_fn(self, x_k: chex.Array, delta_u_k: chex.Array, k: chex.Array, tracking_params: TrackingParams):
        assert x_k.shape == (self.x_dim,) and delta_u_k.shape == (self.u_dim,) and k.shape == ()
        chex.assert_type(k, int)
        # Prepare indices for the in between steps
        indices = k + self.between_step_indices
        tracking_k = k + tracking_params.t_idx_start

        def _next_step(x: chex.Array, _: chex.Array) -> Tuple[chex.Array, chex.Array]:
            tracking_u_k = tracking_params.tracking_data(tracking_k)[1]
            x_dot = self.ode(x, delta_u_k + tracking_u_k)
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
            return self.costs.tracking_running_cost(x_error, _delta_u_k)

        def terminal_cost(_x_k, _delta_u_k, _t_k):
            x_tracking = tracking_params.tracking_data(_t_k)[0]
            x_error = _x_k - x_tracking
            return self.costs.tracking_terminal_cost(x_error, _delta_u_k)

        return cond(k == self.control_steps, terminal_cost, running_cost, x_k, delta_u_k, tracking_k)

    def track_online(self, initial_conditions: chex.Array, tracking_data: TrackingData,
                     t_start_idx: chex.Array, key: chex.PRNGKey) -> OCSolution:
        assert t_start_idx.shape == ()
        chex.assert_type(t_start_idx, int)

        tracking_params = TrackingParams(tracking_data, t_start_idx)
        match self.minimize_method:
            case MinimizationMethod.ILQR_WITH_CEM:
                results = self.optimizer.solve(tracking_params, tracking_params, initial_conditions, self.u_init,
                                               control_low=self.control_low, control_high=self.control_high,
                                               ilqr_hyperparams=self.ilqr_params, cem_hyperparams=self.cem_params,
                                               random_key=key)
            case MinimizationMethod.ILQR:
                results = self.optimizer.solve(tracking_params, tracking_params, initial_conditions,
                                               self.u_init, self.ilqr_hyperparams)
            case _:
                raise NotImplementedError(f'Minimization method {self.minimize_method} not implemented')

        # jax.debug.print('Objective value: {x}', x=results.obj)
        xs, delta_us = results.xs, results.us
        ts_indices = t_start_idx + jnp.arange(self.control_steps + 1)
        us, ts = vmap(tracking_data)(ts_indices)[1:]
        delta_us = jnp.concatenate([delta_us, delta_us[-1:]], axis=0)
        return OCSolution(ts, xs, us + delta_us, results.obj, self.example_dynamics_id())
