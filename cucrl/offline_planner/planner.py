from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import random, vmap, jit
from jax.tree_util import tree_map

from cucrl.environment_interactor.discretization.runge_kutta import RungeKutta
from cucrl.main.config import InteractionConfig
from cucrl.offline_planner.abstract_offline_planner import AbstractOfflinePlanner
from cucrl.utils.classes import DynamicsModel, OfflinePlanningData, OCSolution

pytree = Any

SolveFun = Callable[[jax.Array, jax.Array, DynamicsModel], OCSolution]


class Planner:
    def __init__(self, offline_planner: AbstractOfflinePlanner, interaction_config: InteractionConfig):
        self.offline_planner = offline_planner
        self.interaction_config = interaction_config

        self.plan = jit(self.prepare_plan(self.solve))
        self.initialize = jit(self.prepare_plan(self.solve_init))
        # Here we compute the final time
        control_discretization = RungeKutta(time_horizon=interaction_config.time_horizon,
                                            num_control_steps=interaction_config.policy.num_control_steps,
                                            buffer_control_steps=interaction_config.policy.online_tracking.control_steps + 1)

        self.final_time = control_discretization.continuous_times[-1]

    def solve(self, key, x0, dynamics_model: DynamicsModel) -> OCSolution:
        return self.offline_planner.plan_offline(dynamics_model, key, x0)

    def solve_init(self, *_) -> OCSolution:
        return self.offline_planner.example_oc_solution()

    def prepare_plan(self, solve_fun: SolveFun):
        def _plan(dynamics_model: DynamicsModel, initial_conditions: jax.Array,
                  key: random.PRNGKey) -> OfflinePlanningData:
            # We need to solve optimal control problem for every trajectory
            num_traj = initial_conditions.shape[0]
            traj_keys = random.split(key, num_traj)

            def plan_one_traj(traj_key, x0, dyn_model: DynamicsModel):
                keys = random.split(traj_key, self.interaction_config.policy.offline_planning.num_independent_runs)
                solver_results = vmap(solve_fun, in_axes=(0, None, None))(keys, x0, dyn_model)
                best_index = jnp.argmin(solver_results.opt_value)
                solver_result = tree_map(lambda x: x[best_index], solver_results)
                return OfflinePlanningData(ts=solver_result.ts, xs=solver_result.xs, us=solver_result.us,
                                           x0s=jnp.repeat(x0.reshape(1, -1), solver_result.ts.size, axis=0),
                                           final_t=self.final_time,
                                           target_x=self.offline_planner.simulator_costs.state_target,
                                           target_u=self.offline_planner.simulator_costs.action_target,
                                           dynamics_ids=solver_result.dynamics_id)

            return vmap(plan_one_traj, in_axes=(0, 0, None))(traj_keys, initial_conditions, dynamics_model)

        return _plan


def get_planner(offline_planner: AbstractOfflinePlanner, control_config: InteractionConfig):
    return Planner(offline_planner=offline_planner, interaction_config=control_config)
