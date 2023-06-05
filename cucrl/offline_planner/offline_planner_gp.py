from typing import Tuple

import cyipopt
import jax.debug
import jax.numpy as jnp
from jax import jit, grad, jacrev, vmap, random
from termcolor import colored

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.trajectory_optimization.abstract_trajectory_optimization import AbstractTrajectoryOptimization
from cucrl.trajectory_optimization.ipopt_solvers.eta import EtaSolver
from cucrl.utils.classes import OCSolution, OfflinePlanningParams
from cucrl.utils.helper_functions import RandomFunctionQuantile, GPFunc
from cucrl.utils.representatives import ExplorationStrategy, NumericalComputation, Norm


class GPOfflinePlanner(AbstractTrajectoryOptimization):
    def __init__(self, x_dim: int, u_dim: int, num_nodes: int, time_horizon: Tuple[float, float],
                 dynamics: AbstractDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 numerical_method=NumericalComputation.LGL, minimize_method='IPOPT', alpha=0.95,
                 exploration_norm: Norm = Norm.L_INF, exploration_strategy=ExplorationStrategy.OPTIMISTIC_ETA):
        super().__init__(x_dim=x_dim, u_dim=u_dim, num_nodes=num_nodes, time_horizon=time_horizon,
                         dynamics=dynamics, simulator_costs=simulator_costs, numerical_method=numerical_method,
                         minimize_method=minimize_method, exploration_strategy=exploration_strategy)

        self.eta_place_holder = jnp.ones(shape=(self.num_nodes, self.x_dim))
        self.num_traj_params = (self.x_dim + self.u_dim) * self.num_nodes
        self.example_OCSolution = OCSolution(times=self.time,
                                             xs=jnp.ones(shape=(self.num_nodes, self.x_dim)),
                                             us=jnp.ones(shape=(self.num_nodes, self.u_dim)),
                                             params=jnp.ones(shape=(self.num_traj_params,)),
                                             opt_value=jnp.ones(shape=()),
                                             eta=jnp.ones(shape=(self.num_nodes, self.x_dim)),
                                             dynamics_idx=jnp.ones(shape=(), dtype=jnp.uint32),
                                             dynamics_key=jnp.ones(shape=(2,), dtype=jnp.uint32)
                                             )

        self.num_total_params = (self.x_dim + self.x_dim + self.u_dim) * self.num_nodes
        self.alpha = alpha

        self.rfq, self.gp_func = self.prepare_rfq()

        if exploration_strategy != ExplorationStrategy.OPTIMISTIC_ETA:
            raise NotImplementedError(
                'For TrajectoryOptimizationEta only ExplorationStrategy.OPTIMISTIC_ETA strategy is implemented')
        # First self.state_dim parameters are responsible for the interpolation between mean and std
        self.exploration_norm = exploration_norm
        self.system_dynamics = self.prepare_dynamics()

        objective = self.prepare_objective_eta()
        jac_objective = jit(grad(objective))
        equality_dynamics = self.prepare_equality_constraints_eta()
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

    def prepare_rfq(self):
        h = 1.0

        def kernel(x, y):
            return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h))

        def mean(x):
            return jnp.zeros(shape=(self.x_dim,))

        rfq = RandomFunctionQuantile(dim_in=self.x_dim + self.u_dim, dim_out=self.x_dim,
                                     mu=mean, k=kernel, quantile=self.alpha, outer_dim_norm=Norm.L_INF)
        gp_fun = GPFunc(dim_in=self.x_dim + self.u_dim, dim_out=self.x_dim, mean=mean, kernel=kernel)
        return rfq, gp_fun

    def get_eta(self, parameters):
        reshaped_eta = parameters[:self.x_dim * self.num_nodes]
        return reshaped_eta.reshape(self.num_nodes, self.x_dim)

    def get_params_without_eta(self, parameters):
        return parameters[self.x_dim * self.num_nodes:]

    def get_states_and_controls(self, parameters):
        reshaped_parameters = parameters[self.x_dim * self.num_nodes:].reshape(self.num_nodes,
                                                                               self.x_dim + self.u_dim)
        states = reshaped_parameters[:, :self.x_dim]
        actions = reshaped_parameters[:, self.x_dim:]
        return states, actions

    @staticmethod
    def get_parameters(states, actions):
        return jnp.concatenate([states, actions], axis=1).reshape(-1)

    def prepare_dynamics(self):
        @jit
        def system_dynamics(params, model_states, states, actions, data_stats, key) -> Tuple[
            jnp.ndarray, jnp.ndarray]:
            x_dot_mean, x_dot_std = self.dynamics.mean_and_std_eval_one(params, model_states, states, actions,
                                                                        data_stats)
            return x_dot_mean, x_dot_std

        return system_dynamics

    def prepare_objective_eta(self):
        @jit
        def objective(parameters):
            not_integrated_cost = self.terminal_cost(parameters)
            integrand = self.running_cost(parameters)
            integrated_cost = self.numerical_integral(integrand)
            return not_integrated_cost + integrated_cost

        return objective

    def prepare_equality_constraints_eta(self):
        @jit
        def equality(parameters, initial_conditions, dynamics_params, model_states, data_stats, key):
            states, action = self.get_states_and_controls(parameters)
            eta = self.get_eta(parameters)
            initial_condition_constraint = states[0, :] - initial_conditions
            real_der_mean, real_der_std = vmap(self.system_dynamics, in_axes=(None, None, 0, 0, None, None))(
                dynamics_params, model_states, states, action, data_stats, key)
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
        if self.exploration_norm == Norm.L_INF:
            @jit
            def inequality(parameters):
                eta = self.get_eta(parameters)
                xs, us = self.get_states_and_controls(parameters)
                return self.rfq.inequality(jnp.concatenate([xs, us], axis=1), eta)

            return inequality

        else:
            raise NotImplementedError("This exploration norm hasn't been implemented yet.")

    def running_cost(self, parameters):
        states, actions = self.get_states_and_controls(parameters)
        return vmap(self.simulator_costs.running_cost)(states, actions)

    def terminal_cost(self, parameters):
        states, actions = self.get_states_and_controls(parameters)
        return self.simulator_costs.terminal_cost(states[-1], actions[-1])

    def _solve(self, dynamics_parameters, model_states, dynamics_key, initial_conditions, params_for_opt, data_stats):
        jax.debug.breakpoint()
        self.solver.update_args(dynamics_parameters, model_states, dynamics_key, initial_conditions,
                                params_for_opt, data_stats,
                                params_for_opt[:self.x_dim * self.num_nodes])
        nlp = cyipopt.Problem(
            n=self.solver.params_for_opt.size,
            m=self.solver.cl.size,
            problem_obj=self.solver,
            cl=self.solver.cl,
            cu=self.solver.cu
        )

        nlp.add_option('print_level', 0)
        nlp.add_option('timing_statistics', 'yes')
        # nlp.add_option('nlp_scaling_method', 'gradient')
        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('bound_mult_init_method', 'mu-based')
        nlp.add_option('linear_solver', 'ma57')
        nlp.add_option('tol', 1.0e-8)
        nlp.add_option('max_iter', 500)

        x, info = nlp.solve(params_for_opt)
        print(info['status_msg'])
        print(info['obj_val'])
        states, actions = self.get_states_and_controls(x)
        eta = self.get_eta(x)
        print(colored('Optimization variable eta {}'.format(eta), 'cyan'))
        # Update gp_func
        self.gp_func.prepare_function_vec(xs=jnp.concatenate([states, actions], axis=1), ys=eta)
        return OCSolution(self.time, states, actions, self.get_params_without_eta(x), jnp.array(info['obj_val']),
                          eta=eta, dynamics_key=jnp.ones(shape=(2,), dtype=jnp.uint32),
                          dynamics_idx=jnp.ones(shape=(), dtype=jnp.uint32))

    def solve(self, dynamics_parameters, model_states, dynamics_key, initial_conditions,
              initial_parameters: OfflinePlanningParams, data_stats):
        if initial_parameters.use_eta:
            raise NotImplementedError('This type of exploration is not supported')
        else:
            eta = random.normal(key=initial_parameters.eta_key, shape=(self.num_nodes * self.x_dim,))
            params_for_opt = jnp.concatenate([eta, initial_parameters.xs_and_us_params])
            return self._solve(dynamics_parameters, model_states, dynamics_key, initial_conditions,
                               params_for_opt, data_stats)
