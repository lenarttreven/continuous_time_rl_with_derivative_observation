from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax.lax import cond
from trajax.optimizers import ILQRHyperparams, CEMHyperparams, ILQR_with_CEM_warmstart, ILQR

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.main.config import PolicyConfig
from cucrl.offline_planner.abstract_offline_planner import AbstractOfflinePlanner
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.utils.classes import OCSolution, DynamicsModel, DynamicsIdentifier
from cucrl.utils.representatives import MinimizationMethod


class EtaTimeOfflinePlanner(AbstractOfflinePlanner):
    def __init__(self, x_dim: int, u_dim: int, time_horizon: Tuple[float, float], dynamics: AbstractDynamics,
                 simulator_costs: SimulatorCostsAndConstraints,
                 policy_config: PolicyConfig = PolicyConfig()):
        super().__init__(x_dim=x_dim, u_dim=u_dim, time_horizon=time_horizon, dynamics=dynamics,
                         simulator_costs=simulator_costs)
        # Number of parameters for eta + number of parameters for control
        self.policy_config = policy_config
        # Setup nodes
        self.num_control_nodes = policy_config.num_nodes
        self.num_nodes = self.num_control_nodes + 1
        # Setup time
        total_time = time_horizon[1] - time_horizon[0]
        total_int_steps = policy_config.num_nodes * policy_config.num_int_step_between_nodes
        self.dt = total_time / total_int_steps

        self.ts = jnp.linspace(*self.time_horizon, self.num_nodes)
        self.between_control_ts = jnp.linspace(self.ts[0], self.ts[1], policy_config.num_int_step_between_nodes)
        self.num_total_params = (self.x_dim + self.u_dim) * self.num_control_nodes

        # Setup optimizer, i.e., either ILQR or ILQR with CEM
        self.minimize_method = policy_config.offline_planning.minimization_method
        if self.minimize_method == MinimizationMethod.ILQR_WITH_CEM:
            self.cem_params = CEMHyperparams(max_iter=10, sampling_smoothing=0.0, num_samples=200,
                                             evolution_smoothing=0.0,
                                             elite_portion=0.1)
            self.ilqr_params = ILQRHyperparams(maxiter=100)
            self.control_low = -10.0 * jnp.ones(shape=(self.u_dim + self.x_dim,))
            self.control_high = 10.0 * jnp.ones(shape=(self.u_dim + self.x_dim,))
            self.optimizer = ILQR_with_CEM_warmstart(self.cost_fn, self.dynamics_fn)

        elif self.minimize_method == MinimizationMethod.ILQR:
            self.ilqr_params = ILQRHyperparams(maxiter=100)
            self.optimizer = ILQR(self.cost_fn, self.dynamics_fn)

    def example_dynamics_id(self) -> DynamicsIdentifier:
        return DynamicsIdentifier(eta=jnp.ones(shape=(self.num_nodes, self.x_dim)),
                                  idx=jnp.ones(shape=(), dtype=jnp.int32),
                                  key=jnp.ones(shape=(2,), dtype=jnp.int32))

    def example_oc_solution(self) -> OCSolution:
        return OCSolution(ts=self.ts, xs=jnp.ones(shape=(self.num_nodes, self.x_dim)),
                          us=jnp.ones(shape=(self.num_nodes, self.u_dim)), opt_value=jnp.ones(shape=()),
                          dynamics_id=self.example_dynamics_id())

    def ode(self, x: chex.Array, u: chex.Array, eta: chex.Array, dynamics_model: DynamicsModel):
        assert x.shape == eta.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x_dot_mean, x_dot_std = self.dynamics.mean_and_std_eval_one(dynamics_model, x, u)
        return x_dot_mean + jnp.tanh(eta) * x_dot_std

    def dynamics_fn(self, x, u, t, dynamics_model: DynamicsModel):
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,) and t.shape == () and t.dtype == jnp.int32
        cur_ts = self.ts[t] + self.between_control_ts[:-1]

        def _next_step(_x: chex.Array, _t: chex.Array) -> Tuple[chex.Array, chex.Array]:
            x_dot = self.ode(_x, u[:self.u_dim], u[self.u_dim:], dynamics_model)
            x_next = _x + self.dt * x_dot
            return x_next, x_next

        x_last, _ = jax.lax.scan(_next_step, x, cur_ts)
        return x_last

    def cost_fn(self, x, u, t, dynamics_model: DynamicsModel):
        assert x.shape == (self.x_dim,) and u.shape == (self.x_dim + self.u_dim,)

        def running_cost(_x, _u, _t):
            cur_ts = self.ts[_t] + self.between_control_ts[:-1]

            def _next_step(_x_: chex.Array, _t_: chex.Array) -> Tuple[chex.Array, chex.Array]:
                x_dot = self.ode(_x_, _u[:self.u_dim], _u[self.u_dim:], dynamics_model)
                x_next = _x_ + self.dt * x_dot
                return x_next, self.dt * self.simulator_costs.running_cost(_x_, _u[:self.u_dim])

            x_last, cs = jax.lax.scan(_next_step, _x, cur_ts)
            assert cs.shape == (self.policy_config.num_int_step_between_nodes,)
            return jnp.sum(cs)

        def terminal_cost(_x, _u, _t):
            return self.simulator_costs.terminal_cost(_x, _u[:self.u_dim])

        return cond(t == self.num_control_nodes, terminal_cost, running_cost, x, u, t)

    def plan_offline(self, dynamics_model: DynamicsModel, key: chex.PRNGKey, x0: chex.Array) -> OCSolution:
        initial_actions = jnp.zeros(shape=(self.num_control_nodes, self.u_dim + self.x_dim))
        match self.minimize_method:
            case MinimizationMethod.ILQR_WITH_CEM:
                results = self.optimizer.solve(dynamics_model, dynamics_model, x0, initial_actions,
                                               control_low=self.control_low, control_high=self.control_high,
                                               ilqr_hyperparams=self.ilqr_params, cem_hyperparams=self.cem_params,
                                               random_key=key)
            case MinimizationMethod.ILQR:
                results = self.optimizer.solve(dynamics_model, dynamics_model, x0, initial_actions, self.ilqr_params)
            case _:
                raise NotImplementedError(f"Minimization method {self.minimize_method} not implemented")

        us = results.us[:, :self.u_dim]
        us = jnp.concatenate([us, us[-1][None, :]])
        # eps = results.us[:, self.u_dim:]
        # eps = jnp.tanh(eps)
        # eps = jnp.concatenate([eps, eps[-1][None, :]])
        return OCSolution(ts=self.ts, xs=results.xs, us=us, opt_value=results.obj,
                          dynamics_id=self.example_dynamics_id())
