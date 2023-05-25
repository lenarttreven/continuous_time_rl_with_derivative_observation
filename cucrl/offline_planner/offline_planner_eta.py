from typing import Tuple

import cyipopt
import jax
import jax.numpy as jnp
from jax import jit, grad, jacrev, vmap, random, pure_callback
from termcolor import colored

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.offline_planner.abstract_offline_planner import AbstractOfflinePlanner
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.trajectory_optimization.ipopt_solvers.eta import EtaSolver
from cucrl.utils.classes import OCSolution, OfflinePlanningParams, DynamicsModel, DynamicsIdentifier
from cucrl.utils.representatives import ExplorationStrategy, NumericalComputation, Norm


class EtaOfflinePlanner(AbstractOfflinePlanner):
    def __init__(self, x_dim: int, u_dim: int, num_nodes: int, time_horizon: Tuple[float, float],
                 dynamics: AbstractDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 numerical_method=NumericalComputation.LGL, minimize_method='IPOPT',
                 exploration_norm: Norm = Norm.L_INF, exploration_strategy=ExplorationStrategy.OPTIMISTIC_ETA):
        super().__init__(x_dim=x_dim, u_dim=u_dim, num_nodes=num_nodes, time_horizon=time_horizon,
                         dynamics=dynamics, simulator_costs=simulator_costs, numerical_method=numerical_method,
                         minimize_method=minimize_method, exploration_strategy=exploration_strategy)
        self.num_total_params = self.x_dim + (self.x_dim + self.u_dim) * self.num_nodes
        if exploration_strategy != ExplorationStrategy.OPTIMISTIC_ETA:
            raise NotImplementedError(
                'For TrajectoryOptimizationEta only ExplorationStrategy.OPTIMISTIC_ETA strategy is implemented')

        self.exploration_norm = exploration_norm
        self.system_dynamics = self.prepare_dynamics()
        objective = self.prepare_objective()
        jac_objective = jit(grad(objective))
        equality_dynamics = self.prepare_equality_constraints_dynamics()
        inequality_eta = self.prepare_inequality_constraints_eta()
        jac_inequality_eta = jit(jacrev(inequality_eta))
        jac_equality_dynamics = jit(jacrev(equality_dynamics))

        self.objective = objective
        self.jac_objective = jac_objective
        self.equality_dynamics = equality_dynamics
        self.inequality_eta = inequality_eta
        self.jac_equality_dynamics = jac_equality_dynamics
        self.jac_inequality_eta = jac_inequality_eta

        self.key = random.PRNGKey(0)

        inequality_simulator = self.prepare_inequality_constraints_simulator()
        jac_inequality_simulator = jit(jacrev(inequality_simulator))
        self.inequality_simulator = inequality_simulator
        self.jac_inequality_simulator = jac_inequality_simulator

        self.solver = EtaSolver(self.objective, self.jac_objective,
                                self.equality_dynamics, self.jac_equality_dynamics,
                                self.inequality_simulator, self.jac_inequality_simulator,
                                self.inequality_eta, self.jac_inequality_eta)

        if self.exploration_strategy != ExplorationStrategy.OPTIMISTIC_ETA:
            raise TypeError('This algorithm works only with ExplorationStrategy.OPTIMISTIC_ETA')

    def get_states_and_controls(self, parameters):
        reshaped_parameters = parameters[self.x_dim:].reshape(self.num_nodes, self.x_dim + self.u_dim)
        states = reshaped_parameters[:, :self.x_dim]
        actions = reshaped_parameters[:, self.x_dim:]
        return states, actions

    @staticmethod
    def get_parameters(states, actions):
        return jnp.concatenate([states, actions], axis=1).reshape(-1)

    def prepare_dynamics(self):
        @jit
        def system_dynamics(dynamics_model: DynamicsModel, states: jax.Array,
                            actions: jax.Array) -> Tuple[jax.Array, jax.Array]:
            x_dot_mean, x_dot_std = self.dynamics.mean_and_std_eval_one(dynamics_model, states, actions)
            return x_dot_mean, x_dot_std

        return system_dynamics

    def prepare_objective(self):
        @jit
        def objective(parameters):
            not_integrated_cost = self.terminal_cost(parameters)
            integrand = self.running_cost(parameters)
            integrated_cost = self.numerical_integral(integrand)
            return not_integrated_cost + integrated_cost

        return objective

    def prepare_equality_constraints_dynamics(self):
        @jit
        def equality(parameters, initial_conditions, dynamics_model: DynamicsModel):
            states, action = self.get_states_and_controls(parameters)
            eta = parameters[:self.x_dim]
            initial_condition_constraint = states[0, :] - initial_conditions
            real_der_mean, real_der_std = vmap(self.system_dynamics, in_axes=(None, 0, 0))(dynamics_model, states,
                                                                                           action)
            real_der = real_der_mean + eta * real_der_std
            numerical_der = self.numerical_derivative(states)
            derivative_constraint = numerical_der - real_der
            return jnp.concatenate([initial_condition_constraint.reshape(-1), derivative_constraint.reshape(-1)])

        return equality

    def prepare_inequality_constraints_simulator(self):
        @jit
        def inequality(parameters):
            states, action = self.get_states_and_controls(parameters)
            return vmap(self.simulator_costs.inequality)(states, action).reshape(-1)

        return inequality

    def prepare_inequality_constraints_eta(self):
        @jit
        def inequality(parameters, beta: jax.Array):
            assert beta.shape == (self.x_dim,)
            eta = parameters[:self.x_dim]
            return jnp.concatenate([eta + beta, beta - eta])

        return inequality

    def running_cost(self, parameters):
        states, actions = self.get_states_and_controls(parameters)
        return vmap(self.simulator_costs.running_cost)(states, actions)

    def terminal_cost(self, parameters):
        states, actions = self.get_states_and_controls(parameters)
        return self.simulator_costs.terminal_cost(states[-1], actions[-1])

    def _solve(self, dynamics_model, initial_conditions, params_for_opt):
        self.solver.update_args(dynamics_model, initial_conditions, params_for_opt)
        nlp = cyipopt.Problem(
            n=self.solver.params_for_opt.size,
            m=self.solver.cl.size,
            problem_obj=self.solver,
            cl=self.solver.cl,
            cu=self.solver.cu
        )

        nlp.add_option('print_level', 3)
        nlp.add_option('timing_statistics', 'yes')
        # nlp.add_option('nlp_scaling_method', 'gradient')
        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('bound_mult_init_method', 'mu-based')
        # nlp.add_option('linear_solver', 'ma57')
        nlp.add_option('linear_solver', 'mumps')
        nlp.add_option('tol', 1.0e-8)
        nlp.add_option('max_iter', 500)

        x, info = nlp.solve(params_for_opt)
        print(info['status_msg'])
        print(info['obj_val'])
        xs, us = self.get_states_and_controls(x)
        eta = x[:self.x_dim]
        print(colored('Optimization variable eta {}'.format(eta), 'cyan'))
        dynamics_id = DynamicsIdentifier(key=jnp.ones(shape=(2,), dtype=jnp.uint32),
                                         idx=jnp.ones(shape=(), dtype=jnp.uint32), eta=eta)
        return OCSolution(self.time, xs, us, jnp.array(info['obj_val']), dynamics_id)

    def plan_offline(self, dynamics_model: DynamicsModel, initial_parameters: OfflinePlanningParams,
                     initial_conditions: jax.Array):
        eta = random.normal(key=initial_parameters.key, shape=(self.x_dim,))
        params_for_opt = jnp.concatenate([eta, initial_parameters.xs_and_us_params])
        return pure_callback(self._solve, self.example_OCSolution, dynamics_model, initial_conditions, params_for_opt)
