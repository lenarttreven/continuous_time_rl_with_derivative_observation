from typing import Tuple

import jax
import jax.numpy as jnp
from jax.lax import cond
from trajax.optimizers import ILQRHyperparams, CEMHyperparams, ILQR_with_CEM_warmstart

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.offline_planner.abstract_offline_planner import AbstractOfflinePlanner
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.utils.classes import OCSolution, OfflinePlanningParams, DynamicsModel, DynamicsIdentifier
from cucrl.utils.representatives import ExplorationStrategy, NumericalComputation, Norm


class EtaTimeOfflinePlanner(AbstractOfflinePlanner):
    def __init__(self, state_dim: int, control_dim: int, num_nodes: int, time_horizon: Tuple[float, float],
                 dynamics: AbstractDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 numerical_method=NumericalComputation.LGL, minimize_method='IPOPT',
                 exploration_norm: Norm = Norm.L_INF, exploration_strategy=ExplorationStrategy.OPTIMISTIC_ETA_TIME):
        super().__init__(state_dim=state_dim, control_dim=control_dim, num_nodes=num_nodes, time_horizon=time_horizon,
                         dynamics=dynamics, simulator_costs=simulator_costs, numerical_method=numerical_method,
                         minimize_method=minimize_method, exploration_strategy=exploration_strategy)
        # Number of parameters for eta + number of parameters for control
        self.num_control_nodes = self.num_nodes - 1
        self.dt = (self.time_horizon[1] - self.time_horizon[0]) / self.num_control_nodes
        self.ts = jnp.linspace(self.time_horizon[0], self.time_horizon[1], self.num_control_nodes + 1)
        self.num_total_params = self.state_dim * self.num_control_nodes + self.control_dim * self.num_control_nodes
        self.exploration_norm = exploration_norm

        self.cem_params = CEMHyperparams(max_iter=10, sampling_smoothing=0.0, num_samples=200, evolution_smoothing=0.0,
                                         elite_portion=0.1)
        self.ilqr_params = ILQRHyperparams(maxiter=100)
        self.optimizer = ILQR_with_CEM_warmstart(self.cost_fn, self.dynamics_fn)

        self.control_low = -10.0 * jnp.ones(shape=(self.control_dim + self.state_dim,))
        self.control_high = 10.0 * jnp.ones(shape=(self.control_dim + self.state_dim,))

    def ode(self, x: jax.Array, u: jax.Array, eta: jax.Array, dynamics_model: DynamicsModel):
        assert x.shape == eta.shape == (self.state_dim,) and u.shape == (self.control_dim,)
        x_dot_mean, x_dot_std = self.dynamics.mean_and_std_eval_one(dynamics_model, x, u)
        return x_dot_mean + jnp.tanh(eta) * x_dot_std

    def dynamics_fn(self, x, u, t, dynamics_model: DynamicsModel):
        x_dot = self.ode(x, u[:self.control_dim], u[self.control_dim:], dynamics_model)
        return x + x_dot * self.dt

    def cost_fn(self, x, u, t, cost_params: None):
        assert x.shape == (self.state_dim,) and u.shape == (self.state_dim + self.control_dim,)

        def running_cost(x, u, t):
            return self.dt * self.simulator_costs.running_cost(x, u[:self.control_dim])

        def terminal_cost(x, u, t):
            return self.simulator_costs.terminal_cost(x, u[:self.control_dim])

        return cond(t == self.num_control_nodes, terminal_cost, running_cost, x, u, t)

    def plan_offline(self, dynamics_model: DynamicsModel, initial_parameters: OfflinePlanningParams,
                     x0: jax.Array) -> OCSolution:
        initial_actions = jnp.zeros(shape=(self.num_control_nodes, self.control_dim + self.state_dim))
        out = self.optimizer.solve(None, dynamics_model, x0, initial_actions, control_low=self.control_low,
                                   control_high=self.control_high, ilqr_hyperparams=self.ilqr_params,
                                   cem_hyperparams=self.cem_params, random_key=initial_parameters.key)
        us = out.us[:, :self.control_dim]
        us = jnp.concatenate([us, us[-1][None, :]])
        eps = out.us[:, self.control_dim:]
        eps = jnp.tanh(eps)
        eps = jnp.concatenate([eps, eps[-1][None, :]])
        # Todo: in case we want to use eps to see what kind of function we learned use eps, for now we return 0
        dynamics_id = DynamicsIdentifier(key=self.example_dynamics_id.key, idx=self.example_dynamics_id.idx,
                                         eta=self.example_dynamics_id.eta)
        return OCSolution(ts=self.ts, xs=out.xs, us=us, opt_value=out.obj, dynamics_id=dynamics_id)
