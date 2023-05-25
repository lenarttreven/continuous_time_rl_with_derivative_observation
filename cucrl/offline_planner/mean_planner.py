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


class MeanOfflinePlanner(AbstractOfflinePlanner):
    def __init__(self, x_dim: int, u_dim: int, num_nodes: int, time_horizon: Tuple[float, float],
                 dynamics: AbstractDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 numerical_method=NumericalComputation.LGL, minimize_method='IPOPT',
                 exploration_norm: Norm = Norm.L_INF, exploration_strategy=ExplorationStrategy.OPTIMISTIC_ETA_TIME):
        super().__init__(x_dim=x_dim, u_dim=u_dim, num_nodes=num_nodes, time_horizon=time_horizon,
                         dynamics=dynamics, simulator_costs=simulator_costs, numerical_method=numerical_method,
                         minimize_method=minimize_method, exploration_strategy=exploration_strategy)
        # Number of parameters for eta + number of parameters for control
        self.num_control_nodes = self.num_nodes - 1
        self.dt = (self.time_horizon[1] - self.time_horizon[0]) / self.num_control_nodes
        self.ts = jnp.linspace(self.time_horizon[0], self.time_horizon[1], self.num_control_nodes + 1)
        self.num_total_params = self.u_dim * self.num_control_nodes
        self.exploration_norm = exploration_norm

        self.cem_params = CEMHyperparams(max_iter=10, sampling_smoothing=0.0, num_samples=200, evolution_smoothing=0.0,
                                         elite_portion=0.1)
        self.ilqr_params = ILQRHyperparams(maxiter=100)
        self.optimizer = ILQR_with_CEM_warmstart(self.cost_fn, self.dynamics_fn)

        self.control_low = -10.0 * jnp.ones(shape=(self.u_dim,))
        self.control_high = 10.0 * jnp.ones(shape=(self.u_dim,))

    def ode(self, x: jax.Array, u: jax.Array, dynamics_model: DynamicsModel):
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x_dot_mean = self.dynamics.mean_eval_one(dynamics_model, x, u)
        return x_dot_mean

    def dynamics_fn(self, x, u, t, dynamics_model: DynamicsModel):
        x_dot = self.ode(x, u, dynamics_model)
        return x + x_dot * self.dt

    def cost_fn(self, x, u, t, cost_params: None):
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)

        def running_cost(x, u, t):
            return self.dt * self.simulator_costs.running_cost(x, u)

        def terminal_cost(x, u, t):
            return self.simulator_costs.terminal_cost(x, u)

        return cond(t == self.num_control_nodes, terminal_cost, running_cost, x, u, t)

    def plan_offline(self, dynamics_model: DynamicsModel, initial_parameters: OfflinePlanningParams,
                     x0: jax.Array) -> OCSolution:
        initial_actions = jnp.zeros(shape=(self.num_control_nodes, self.u_dim))
        out = self.optimizer.solve(None, dynamics_model, x0, initial_actions, control_low=self.control_low,
                                   control_high=self.control_high, ilqr_hyperparams=self.ilqr_params,
                                   cem_hyperparams=self.cem_params, random_key=initial_parameters.key)
        us = out.us
        us = jnp.concatenate([us, us[-1][None, :]])
        dynamics_id = DynamicsIdentifier(key=self.example_dynamics_id.key, idx=self.example_dynamics_id.idx,
                                         eta=self.example_dynamics_id.eta)
        return OCSolution(ts=self.ts, xs=out.xs, us=us, opt_value=out.obj, dynamics_id=dynamics_id)
