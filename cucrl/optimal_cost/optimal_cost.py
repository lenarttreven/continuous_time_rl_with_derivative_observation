from typing import Tuple

import jax
import jax.numpy as jnp
from cyipopt import minimize_ipopt
from jax import jit, vmap, grad, jacrev, random

from cucrl.simulator.simulator_costs import Pendulum as PendulumCost
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.simulator.simulator_dynamics import Pendulum
from cucrl.simulator.simulator_dynamics import SimulatorDynamics
from cucrl.trajectory_optimization.numerical_computations.numerical_computation import get_numerical_computation
from cucrl.utils.representatives import NumericalComputation


class OptimalCost:
    def __init__(self, simulator_dynamics: SimulatorDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 time_horizon: Tuple[float, float], num_nodes: int = 50):
        self.simulator_dynamics = simulator_dynamics
        self.simulator_costs = simulator_costs
        self.num_nodes = num_nodes
        self.state_dim = self.simulator_dynamics.state_dim
        self.control_dim = self.simulator_dynamics.control_dim

        self.numerical_computation = get_numerical_computation(numerical_computation=NumericalComputation.LGL,
                                                               num_nodes=self.num_nodes, time_horizon=time_horizon)

        self.time = self.numerical_computation.time
        self.numerical_derivative = self.numerical_computation.numerical_derivative
        self.numerical_integral = self.numerical_computation.numerical_integral
        self.num_traj_params = (self.state_dim + self.control_dim) * self.num_nodes

        objective = self.prepare_objective()
        jac_objective = jit(grad(objective))
        equality_dynamics = self.prepare_equality_constraints_dynamics()
        jac_equality_dynamics = jit(jacrev(equality_dynamics))

        self.objective = objective
        self.jac_objective = jac_objective
        self.equality_dynamics = equality_dynamics
        self.jac_equality_dynamics = jac_equality_dynamics

        inequality_simulator = self.prepare_inequality_constraints_simulator()
        jac_inequality_simulator = jit(jacrev(inequality_simulator))
        self.inequality_simulator = inequality_simulator
        self.jac_inequality_simulator = jac_inequality_simulator

    def get_states_and_controls(self, parameters):
        reshaped_parameters = parameters.reshape(self.num_nodes, self.state_dim + self.control_dim)
        states = reshaped_parameters[:, :self.state_dim]
        actions = reshaped_parameters[:, self.state_dim:]
        return states, actions

    def running_cost(self, parameters):
        states, actions = self.get_states_and_controls(parameters)
        return vmap(self.simulator_costs.running_cost)(states, actions)

    def terminal_cost(self, parameters):
        states, actions = self.get_states_and_controls(parameters)
        return self.simulator_costs.terminal_cost(states[-1], actions[-1])

    def prepare_objective(self):
        @jit
        def objective(parameters):
            not_integrated_cost = self.terminal_cost(parameters)
            integrand = self.running_cost(parameters)
            integrated_cost = self.numerical_integral(integrand)
            return not_integrated_cost + integrated_cost

        return objective

    def prepare_inequality_constraints_simulator(self):
        @jit
        def inequality(parameters):
            states, action = self.get_states_and_controls(parameters)
            return vmap(self.simulator_costs.inequality)(states, action).reshape(-1)

        return inequality

    def prepare_equality_constraints_dynamics(self):
        @jit
        def equality(parameters, initial_conditions):
            states, action = self.get_states_and_controls(parameters)
            initial_condition_constraint = states[0, :] - initial_conditions
            real_der = vmap(self.simulator_dynamics.dynamics, in_axes=(0, 0, None))(states, action,
                                                                                    jnp.array([0.0]).reshape(1, ))
            numerical_der = self.numerical_derivative(states)
            derivative_constraint = numerical_der - real_der
            return jnp.concatenate([initial_condition_constraint.reshape(-1), derivative_constraint.reshape(-1)])

        return equality

    def solve(self, initial_conditions: jax.Array):
        params = random.normal(random.PRNGKey(0), (self.num_traj_params,))
        cons = [
            {'type': 'eq', 'fun': self.equality_dynamics, 'jac': self.jac_equality_dynamics,
             'args': (initial_conditions,)},
            {'type': 'ineq', 'fun': self.inequality_simulator, 'jac': self.jac_inequality_simulator},
        ]
        out_res = minimize_ipopt(self.objective, jac=self.jac_objective, x0=params, constraints=cons,
                                 options={'disp': 4})
        return out_res.fun


if __name__ == '__main__':
    from jax.config import config

    config.update("jax_enable_x64", True)
    system = Pendulum()
    costs = PendulumCost()
    optimal_cost = OptimalCost(simulator_dynamics=system, simulator_costs=costs, time_horizon=(0.0, 10.0),
                               num_nodes=100)
    out = optimal_cost.solve(jnp.array([0.5 * jnp.pi, 0]))
    print(out)
