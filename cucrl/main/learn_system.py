import copy
import os
import pickle
import time
from copy import deepcopy
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import wandb
from jax import jit, value_and_grad, random, vmap
from jax.tree_util import tree_leaves
from termcolor import colored

from cucrl.dynamics_with_control.dynamics_models import get_dynamics
from cucrl.environment_interactor.get_interactor import get_interactor
from cucrl.main.config import RunConfig, DynamicsConfig, OptimizersConfig, DataGeneratorConfig
from cucrl.main.data_stats import DataLoader
from cucrl.main.data_stats import DataStats, Normalizer, DataLearn, DynamicsData, Stats
from cucrl.main.handlers import DataGenerator, DynamicsDataManager
from cucrl.main.handlers import Keys, DataRepr, VisualisationData
from cucrl.objectives.objectives import Objectives
from cucrl.offline_planner.offline_planner import get_offline_planner
from cucrl.optimal_cost.best_possible_discrete_time import BestPossibleDiscreteAlgorithm
from cucrl.optimal_cost.dynamics_wrapper import TrueDynamicsWrapper
from cucrl.plotter.plotter import Plotter
from cucrl.schedules.learning_rate import get_learning_rate
from cucrl.simulator.simulator_costs import get_simulator_costs
from cucrl.utils.classes import MeasurementSelection
from cucrl.utils.classes import PlotData, PlotOpenLoop, OfflinePlanningData, DynamicsModel, NumberTrainPoints
from cucrl.utils.helper_functions import AngleLayerDynamics, BetaExploration
from cucrl.utils.representatives import Optimizer
from cucrl.utils.splines import MultivariateSpline

Schedule = Callable[[int], float]
pytree = Any


class LearnSystem:
    def __init__(self, config: RunConfig):
        # Prepare dimensions
        self.config = config
        self._prepare_dimensions(config.data_generator)

        self.angle_layer = AngleLayerDynamics(state_dim=self.x_dim, control_dim=self.u_dim,
                                              angles_dim=config.interaction.angles_dim,
                                              state_scaling=self.config.data_generator.simulator.scaling.state_scaling)
        self.normalizer = Normalizer(state_dim=self.x_dim, action_dim=self.u_dim,
                                     tracking_c=None, angle_layer=self.angle_layer)
        init_stats_state = Stats(jnp.zeros(shape=(self.x_dim,)), jnp.ones(shape=(self.x_dim,)))
        init_stats_control = Stats(jnp.zeros(shape=(self.u_dim,)), jnp.ones(shape=(self.u_dim,)))
        init_stats_time = Stats(jnp.zeros(shape=(1,)), jnp.ones(shape=(1,)))
        angles_shape = self.angle_layer.angle_layer(jnp.ones(shape=(self.x_dim,))).shape
        init_stats_after_angle_layer = Stats(jnp.zeros(shape=angles_shape), jnp.ones(shape=angles_shape))

        self.data_stats = DataStats(ts_stats=init_stats_time, xs_stats=init_stats_state, us_stats=init_stats_control,
                                    xs_dot_noise_stats=init_stats_state,
                                    xs_after_angle_layer=init_stats_after_angle_layer)
        # Prepare pool for vector field data
        self.vector_field_data = DynamicsDataManager(self.x_dim, self.u_dim)
        self.simulator_costs = get_simulator_costs(config.data_generator.simulator.simulator_type,
                                                   scaling=self.config.data_generator.simulator.scaling)

        # Prepare randomness
        self.current_rng = jax.random.PRNGKey(config.seed)
        self.current_rng, self.episode_key = random.split(self.current_rng)
        # Prepare data_loader
        self.data_batcher = DataLoader(batch_size=config.optimizers.batch_size,
                                       no_batching=config.optimizers.no_batching)
        # Prepare black box data
        self.offline_planning_data = None
        # Prepare dynamics
        self.dynamics_config = config.dynamics
        self._prepare_dynamics(self.dynamics_config)
        # Prepare control
        self.interaction = config.interaction
        # Prepare trajectory optimization
        self._prepare_offline_planner()
        # Prepare feedback controller
        self._prepare_controller(self.interaction, config.data_generator.data_collection.initial_conditions)
        # Prepare data generator
        self.data_generator = DataGenerator(data_generation=config.data_generator, interactor=self.policy)
        # Initialize parameters
        self._initialize_parameters()
        # Prepare plotter
        self.plotter = Plotter(state_dim=self.data_generator.state_dim, action_dim=self.data_generator.control_dim)
        # Prepare objectives
        self._prepare_objectives()
        # Prepare optimizer
        self.optimizer_config = config.optimizers
        self._prepare_optimizer(self.optimizer_config)
        # Prepare tracking
        self.track_wandb = config.logging.track_wandb
        self.track_just_loss = config.logging.track_just_loss
        self.visualization = config.logging.visualization
        # Prepare beta exploration
        self.beta_exploration = BetaExploration(delta=0.1, state_dim=self.x_dim, rkhs_bound=1,
                                                type=self.config.interaction.policy.offline_planning.beta_exploration)

        best_possible_discrete = BestPossibleDiscreteAlgorithm(
            simulator_dynamics=self.data_generator.simulator.simulator_dynamics,
            simulator_costs=self.simulator_costs,
            time_horizon=self.config.data_generator.simulator.time_horizon,
            num_nodes=self.config.comparator.num_discrete_points)
        self.best_possible_discrete_cost = [best_possible_discrete.get_optimal_cost(ic) for ic in
                                            self.data_generator.initial_conditions]

        self.optimal_cost = self.compute_optimal_cost(self.interaction, self.data_generator.initial_conditions)
        self.dynamics_model = None

    def _prepare_dimensions(self, data_generation: DataGeneratorConfig):
        self.data_generation_key = data_generation.data_collection.data_generation_key
        self.u_dim = data_generation.control_dim
        self.x_dim = data_generation.state_dim
        self.num_trajectories = len(data_generation.data_collection.initial_conditions)
        self.time_horizon = data_generation.simulator.time_horizon
        self.noise_stds = data_generation.data_collection.noise

    def _prepare_dynamics(self, dynamics: DynamicsConfig):
        time_dynamics = time.time()
        self.dynamics = get_dynamics(dynamics_model=dynamics.type, state_dim=self.x_dim,
                                     action_dim=self.u_dim, normalizer=self.normalizer,
                                     dynamics_config=dynamics,
                                     angle_layer=self.angle_layer,
                                     measurement_collection_config=self.config.interaction.measurement_collector)
        print("Time for dynamics preparation: ", time.time() - time_dynamics)

    def _prepare_controller(self, control_options, initial_condition):
        self.policy = get_interactor(self.x_dim, self.u_dim, self.dynamics, initial_condition,
                                     self.normalizer,
                                     self.angle_layer, control_options, self.offline_planner,
                                     self.config.data_generator.simulator.scaling)

    def compute_optimal_cost(self, control_options, initial_condition):
        offline_planer_config = self.interaction.policy.offline_planning
        planner_class = get_offline_planner(offline_planer_config.exploration_strategy)
        true_dynamics_wrapper = TrueDynamicsWrapper(
            simulator_dynamics=self.data_generator.simulator.simulator_dynamics, simulator_costs=self.simulator_costs,
            measurement_collection_config=self.config.interaction.measurement_collector)
        offline_planner = planner_class(x_dim=self.x_dim, u_dim=self.u_dim, time_horizon=self.time_horizon,
                                        dynamics=true_dynamics_wrapper, simulator_costs=self.simulator_costs,
                                        policy_config=self.interaction.policy)

        true_policy = get_interactor(self.x_dim, self.u_dim, true_dynamics_wrapper, initial_condition,
                                     self.normalizer,self.angle_layer, control_options, offline_planner,
                                     self.config.data_generator.simulator.scaling)

        true_data_gen = DataGenerator(data_generation=self.config.data_generator, interactor=true_policy)

        dynamics_model = DynamicsModel(params=self.parameters['dynamics'], model_stats=self.stats['dynamics'],
                                       data_stats=self.data_stats, episode=1,
                                       beta=self.beta_exploration(num_episodes=1),
                                       history=DynamicsData(
                                           ts=jnp.ones(shape=(1, 1)),
                                           xs=jnp.ones(shape=(1, self.x_dim)),
                                           us=jnp.ones(shape=(1, self.u_dim)),
                                           xs_dot_std=jnp.ones(shape=(1, self.x_dim)),
                                           xs_dot=jnp.ones(shape=(1, self.x_dim))),
                                       calibration_alpha=jnp.ones(shape=(self.x_dim,)))
        key = jax.random.PRNGKey(0)
        true_data_gen.simulator.interactor.update(dynamics_model=dynamics_model, key=key)
        trajs = true_data_gen.generate_trajectories(key)
        costs = vmap(self.compute_cost)(trajs[0].visualization_data.ts, trajs[0].visualization_data.xs,
                                        trajs[0].visualization_data.us)
        print("Optimal costs: ", costs)
        return costs

    def _initialize_parameters(self):
        time_parameters = time.time()
        # Get new random keys
        self.current_rng, *keys = jax.random.split(self.current_rng, 3)

        # Initialize parameters
        params_dynamics, stats_dynamics = self.dynamics.initialize_parameters(keys[0])

        self.parameters = {"dynamics": params_dynamics}
        self.stats = {'dynamics': stats_dynamics}

        self.old_parameters = copy.deepcopy(self.parameters)
        self.old_stats = copy.deepcopy(self.stats)

        # Count number of parameters of the model
        self.num_dynamics_parameters = 0

        for leave in tree_leaves(params_dynamics):
            self.num_dynamics_parameters += leave.size

        self.num_parameters = self.num_dynamics_parameters
        print("Time to initialize parameters", time.time() - time_parameters)

    def _reset_parameters_dynamics(self, key):
        params_dynamics, states_dynamics = self.dynamics.initialize_parameters(key)
        self.parameters['dynamics'] = params_dynamics
        self.stats['dynamics'] = states_dynamics

    def _prepare_objectives(self):
        time_objective_builder = time.time()
        # Build objectives - learn_dynamics and optimize_policy
        objectives = Objectives(dynamics=self.dynamics)

        self.dynamics_training = objectives.dynamics_training
        self.values_and_grad_dynamics_training = jit(value_and_grad(self.dynamics_training, 0, has_aux=True))
        print("Time to prepare objective builder", time.time() - time_objective_builder)

    def _prepare_offline_planner(self):
        policy_config = self.interaction.policy
        planner_class = get_offline_planner(policy_config.offline_planning.exploration_strategy)
        self.offline_planner = planner_class(x_dim=self.x_dim, u_dim=self.u_dim, time_horizon=self.time_horizon,
                                             dynamics=self.dynamics, simulator_costs=self.simulator_costs,
                                             policy_config=policy_config)

    def _prepare_optimizer(self, optimizer: OptimizersConfig):
        # Prepare learning rate
        self.learning_rate_dynamics_training = get_learning_rate(
            optimizer.dynamics_training.learning_rate.type,
            optimizer.dynamics_training.learning_rate.kwargs)
        # Prepare optimizer for learning dynamics
        if optimizer.dynamics_training.type == Optimizer.ADAM:
            self.optimizer_dynamics_training = optax.adamw
        elif optimizer.dynamics_training.type == Optimizer.SGD:
            self.optimizer_dynamics_training = optax.sgd

    def generate_data(self) -> Tuple[DataRepr, MeasurementSelection]:
        # Generate noisy trajectories following the dynamics with the given policy
        self.data_generation_key, key = random.split(self.data_generation_key)
        return self.data_generator.generate_trajectories(key)

    def prepare_dynamics_training(self):
        if self.track_wandb:
            wandb.define_metric("x_axis/step")
            # set all other train/ metrics to use this step
            wandb.define_metric("Dynamics training/*", step_metric="x_axis/step", summary="last")
            wandb.define_metric('Dynamics training/iter_time', step_metric="x_axis/step", summary='mean')
        tx_dynamics_training = self.optimizer_dynamics_training(
            self.learning_rate_dynamics_training,
            weight_decay=self.config.optimizers.dynamics_training.wd)

        def prepare_parameters_and_stats(key):
            # Reset smoother and dynamics parameters
            key, subkey = jax.random.split(key)
            self._reset_parameters_dynamics(subkey)
            # Split parameters to trainable parameters and policy parameters
            params_train = {'dynamics': self.parameters['dynamics']}
            stats_train = {'dynamics': self.stats['dynamics']}
            # Watch trainable parameters with the optimizer
            opt_state = tx_dynamics_training.init(params_train)
            return params_train, stats_train, opt_state

        # Define optimization step
        @jit
        def do_step(step, parameters, stats, opt_state, data, data_stats, keys, num_train_points):
            (current_loss, new_stats), params_grad = self.values_and_grad_dynamics_training(
                parameters, stats, data, data_stats, keys, num_train_points)

            updates, opt_state = tx_dynamics_training.update(params_grad, opt_state, parameters)
            parameters_train_new = optax.apply_updates(parameters, updates)
            return current_loss, opt_state, parameters_train_new, new_stats

        # Run optimization for predefine number of steps
        def run_optimization(number_of_steps, params, stats, opt_state, data, data_stats, episode, keys):
            self.optimize(do_step=do_step, num_steps=number_of_steps, params=params, stats=stats,
                          opt_state=opt_state, data=data, data_stats=data_stats,
                          log_name='Dynamics training/Full objective', episode=episode, keys=keys)

        return prepare_parameters_and_stats, run_optimization

    def optimize(self, do_step, num_steps, params, stats, opt_state, data: DataLearn, data_stats, log_name: str,
                 keys: Keys, episode: int = 0):
        # Set the timer
        current_time, iter_time = time.time(), 0
        initial_time = current_time
        self.current_rng, key = random.split(self.current_rng)

        num_train_points_dynamics = data.dynamics_data.xs.shape[0]
        num_train_points = NumberTrainPoints(dynamics=num_train_points_dynamics, matching=0,
                                             smoother=0)
        data_loader = self.data_batcher.prepare_loader(dataset=data, key=self.current_rng)

        # for step in range(num_steps):
        for step, data_batch in enumerate(data_loader):
            # Print times of first 10 steps to see the time which we need for training
            if step >= num_steps:
                break
            cur_data = DataLearn(dynamics_data=data_batch)
            new_step_key_gen, step_key = jax.random.split(keys.step_key)
            this_step_keys = Keys(step_key=step_key, episode_key=keys.episode_key)
            keys = Keys(step_key=new_step_key_gen, episode_key=keys.episode_key)
            if step < 5:
                next_time = time.time()
                print("Time for step {}:".format(step), next_time - current_time)
                current_time = next_time

            # Make an optimization step
            loss, opt_state, new_parameters, new_model_states = do_step(step, params, stats, opt_state, cur_data,
                                                                        data_stats, this_step_keys, num_train_points)
            # Track optimization progress with Weights & Biases
            if self.track_wandb:
                variables_dict = dict()
                variables_dict[log_name] = float(loss)
                variables_dict['x_axis/step'] = step + episode * num_steps
                wandb.log(variables_dict)
            params = new_parameters
            stats = new_model_states

        # Print total training time
        time_spent_for_training = time.time() - initial_time
        print("Time spent for training:", time_spent_for_training, "seconds")
        # Update the trained parameters and stats
        for dict_keys, value in copy.deepcopy(self.parameters).items():
            self.old_parameters[dict_keys] = value
        for dict_keys, value in copy.deepcopy(self.stats).items():
            self.old_stats[dict_keys] = value
        for dict_keys, value in stats.items():
            self.stats[dict_keys] = value
        for dict_keys, value in params.items():
            self.parameters[dict_keys] = value
        # Save final parameters
        if self.track_wandb:
            directory = os.path.join(wandb.run.dir, 'models')
            if not os.path.exists(directory):
                os.makedirs(directory)
            model_path = os.path.join('models', 'final_parameters.pkl')
            with open(os.path.join(wandb.run.dir, model_path), 'wb') as handle:
                pickle.dump(self.parameters, handle)
            wandb.save(os.path.join(wandb.run.dir, model_path), wandb.run.dir)

    def compute_cost(self, times, states, controls):
        rc_to_integrate = vmap(self.simulator_costs.running_cost)(states, controls)
        running_cost = jnp.trapz(x=times.reshape(-1), y=rc_to_integrate)
        terminal_cost = self.simulator_costs.terminal_cost(states[-1], controls[-1])
        return running_cost + terminal_cost

    def cost_logging(self, episode: int, true_data: VisualisationData, predicted_data: None | OfflinePlanningData):
        true_traj_costs = vmap(self.compute_cost)(true_data.ts, true_data.xs, true_data.us)
        wandb.log({'episode': episode} | {'Cost/True trajectory {}'.format(traj_id): float(true_traj_costs[traj_id]) for
                                          traj_id in range(self.num_trajectories)})
        wandb.log(
            {'episode': episode} | {
                'Cost/Optimal Trajectory {}'.format(traj_id): float(self.optimal_cost[traj_id])
                for traj_id in range(self.num_trajectories)})

        wandb.log(
            {'episode': episode} | {
                'Cost/Best possible discrete time {}'.format(traj_id): float(self.best_possible_discrete_cost[traj_id])
                for traj_id in range(self.num_trajectories)})

        if predicted_data is not None:
            predicted_traj_costs = vmap(self.compute_cost)(predicted_data.ts, predicted_data.xs, predicted_data.us)
            wandb.log(
                {'episode': episode} | {
                    'Cost/Predicted trajectory {}'.format(traj_id): float(predicted_traj_costs[traj_id])
                    for traj_id in range(self.num_trajectories)})

    @staticmethod
    def data_for_learning(dynamics_data: DynamicsData) -> DataLearn:
        return DataLearn(dynamics_data=dynamics_data)

    def log_measurement_selection(self, episode: int, measurement_selection: MeasurementSelection):
        directory = os.path.join(wandb.run.dir, 'mss')
        if not os.path.exists(directory):
            os.makedirs(directory)
        data_path = os.path.join('mss', 'measurement_selection_{}.pkl'.format(episode))
        with open(os.path.join(wandb.run.dir, data_path), 'wb') as handle:
            pickle.dump(measurement_selection, handle)
        wandb.save(os.path.join(wandb.run.dir, data_path), wandb.run.dir)

        fig_measurement_selection, fig_measurement_selection_space, fig_phase = self.plotter.plot_measurement_selection(
            measurement_selection)
        fig_measurement_selection.tight_layout()
        fig_measurement_selection_space.tight_layout()
        episode_prefix = 'Time Selection based on Hallucination' + '/'
        episode_suffix = 'Episode ' + str(episode)
        episode_prefix_space = 'Time Selection based on Hallucination Space Difference Plot' + '/'
        if fig_phase is None:
            wandb.log({episode_prefix + episode_suffix: wandb.Image(fig_measurement_selection),
                       episode_prefix_space + episode_suffix: wandb.Image(fig_measurement_selection_space)})
        else:
            fig_phase.tight_layout()
            episode_prefix_phase = 'Time Selection based on Hallucination Phase' + '/'
            wandb.log({episode_prefix + episode_suffix: wandb.Image(fig_measurement_selection),
                       episode_prefix_space + episode_suffix: wandb.Image(fig_measurement_selection_space),
                       episode_prefix_phase + episode_suffix: wandb.Image(fig_phase)})

        plt.close('all')

    def run_episodes(self, num_episodes: int, num_iter_training: int):
        prepare_params_dynamics_training, run_optimization_dynamics_training = self.prepare_dynamics_training()

        for episode in range(num_episodes):
            print(colored('Episode {}'.format(episode), 'blue'))
            self.current_rng, self.episode_key = random.split(self.current_rng)
            self.current_rng, key = random.split(self.current_rng)
            # Generate new rollouts
            if episode == 0:
                dynamics_model = DynamicsModel(params=self.parameters['dynamics'], model_stats=self.stats['dynamics'],
                                               data_stats=self.data_stats, episode=episode,
                                               beta=self.beta_exploration(num_episodes=num_episodes),
                                               history=DynamicsData(
                                                   ts=jnp.ones(shape=(1, 1)),
                                                   xs=jnp.ones(shape=(1, self.x_dim)),
                                                   us=jnp.ones(shape=(1, self.u_dim)),
                                                   xs_dot_std=jnp.ones(shape=(1, self.x_dim)),
                                                   xs_dot=jnp.ones(shape=(1, self.x_dim))),
                                               calibration_alpha=jnp.ones(shape=(self.x_dim,)))
                self.data_generator.simulator.interactor.update(dynamics_model=dynamics_model, key=key)
            data, measurement_selection = self.generate_data()
            # Add data to permanent pool
            xs_dot_noise = jnp.concatenate(data.observation_data.xs_dot_noise)
            xs_dot_std = self.config.data_generator.data_collection.noise * jnp.ones_like(xs_dot_noise)

            current_dynamics_data = DynamicsData(ts=jnp.concatenate(data.observation_data.ts),
                                                 xs=jnp.concatenate(data.observation_data.xs),
                                                 us=jnp.concatenate(data.observation_data.us),
                                                 xs_dot=xs_dot_noise, xs_dot_std=xs_dot_std)
            # Need to add dynamics_data to the permanent and test pool
            self.vector_field_data.add_data_to_permanent_pool(current_dynamics_data)

            current_data = self.data_for_learning(self.vector_field_data.permanent_pool)
            data_stats = self.normalizer.compute_stats(current_data)
            self.data_stats = data_stats
            if episode >= 1:
                self.log_measurement_selection(episode, measurement_selection)
            self.cost_logging(episode, data.visualization_data, predicted_data=self.offline_planning_data)
            # Pretraining
            print(colored('Training', 'green'))
            self.current_rng, step_key = jax.random.split(self.current_rng)
            keys = Keys(episode_key=self.episode_key, step_key=step_key)
            # Train dynamics
            self.current_rng, train_key = jax.random.split(self.current_rng)
            params_train, stats_train, opt_state = prepare_params_dynamics_training(train_key)

            run_optimization_dynamics_training(number_of_steps=num_iter_training, params=params_train,
                                               stats=stats_train, opt_state=opt_state, data=current_data,
                                               data_stats=data_stats, episode=episode, keys=keys)

            self.dynamics_model = DynamicsModel(params=self.parameters['dynamics'], model_stats=self.stats['dynamics'],
                                                data_stats=self.data_stats, episode=episode + 1,
                                                calibration_alpha=jnp.ones(shape=(self.x_dim,)),
                                                history=self.vector_field_data.permanent_pool)

            # Compute calibration
            calibration_dynamics_alpha = self.dynamics.calculate_calibration_alpha(
                dynamics_model=self.dynamics_model, xs=self.vector_field_data.test_pool.xs,
                us=self.vector_field_data.test_pool.us, xs_dot=self.vector_field_data.test_pool.xs_dot,
                xs_dot_std=self.vector_field_data.test_pool.xs_dot_std)

            # Here we need to add uncertainties for the beta exploration
            stds = vmap(self.dynamics.mean_and_std_eval_one, in_axes=(None, 0, 0))(self.dynamics_model,
                                                                                   current_dynamics_data.xs,
                                                                                   current_dynamics_data.us)[1]

            self.beta_exploration.update_info_gain(stds=stds, taus=xs_dot_std)

            if self.visualization and self.track_wandb:
                self.visualize_data(data=data, episode=episode)

            # Trajectory optimization
            print(colored('Trajectory optimization', 'blue'))
            start_time_trajectory_optimization = time.time()
            self.current_rng, key = random.split(self.current_rng)

            dynamics_model = DynamicsModel(params=self.parameters['dynamics'], model_stats=self.stats['dynamics'],
                                           data_stats=self.data_stats, episode=episode + 1,
                                           beta=self.beta_exploration(num_episodes=num_episodes),
                                           history=deepcopy(self.vector_field_data.permanent_pool),
                                           calibration_alpha=calibration_dynamics_alpha)

            self.dynamics_model = dynamics_model
            self.data_generator.simulator.interactor.update(dynamics_model=dynamics_model, key=key)
            print('Time spent for trajectory optimization: {} seconds'.format(
                time.time() - start_time_trajectory_optimization))
            self.offline_planning_data = self.data_generator.simulator.interactor.offline_planning_data
            if self.visualization and self.track_wandb:
                self.visualize_controller(offline_planning_data=self.offline_planning_data, episode=episode)
            # We copy parameters of the last state since we need them for the next iteration
            for key, value in copy.deepcopy(self.parameters).items():
                self.old_parameters[key] = value
            for key, value in copy.deepcopy(self.stats).items():
                self.old_stats[key] = value

    def visualize_controller(self, offline_planning_data: OfflinePlanningData, episode: int):
        vis_times = self.data_generator.visualization_times_whole_horizon

        def evaluate_prediction(offline_plan: OfflinePlanningData, ts):
            spline_states = MultivariateSpline(offline_plan.ts.reshape(-1), offline_plan.xs)
            spline_controls = MultivariateSpline(offline_plan.ts.reshape(-1), offline_plan.us)
            return spline_states(ts), spline_controls(ts)

        state_prediction, control_prediction = vmap(evaluate_prediction, in_axes=(0, None))(offline_planning_data,
                                                                                            vis_times)

        plot_data = PlotOpenLoop(
            times=offline_planning_data.ts,
            states=offline_planning_data.xs,
            controls=offline_planning_data.us,
            visualization_times=jnp.repeat(vis_times[jnp.newaxis, ...], repeats=self.num_trajectories, axis=0),
            controls_prediction=control_prediction,
            state_prediction=state_prediction
        )

        # Save & Plot data
        directory = os.path.join(wandb.run.dir, 'data')
        if not os.path.exists(directory):
            os.makedirs(directory)
        data_path = os.path.join('data', 'control_learning_data_episode_{}.pkl'.format(episode))
        with open(os.path.join(wandb.run.dir, data_path), 'wb') as handle:
            pickle.dump(plot_data, handle)
        wandb.save(os.path.join(wandb.run.dir, data_path), wandb.run.dir)

        fig_control_learning = self.plotter.plot_open_loop_learning(plot_data)
        fig_control_learning.tight_layout()

        episode_prefix = 'Episode ' + str(episode) + '/'
        wandb.log({episode_prefix + 'TO: Control learning': wandb.Image(fig_control_learning), })
        plt.close('all')

    def visualize_data(self, data: DataRepr, episode: int, stage: str = ''):
        # Compute dynamics terms
        dynamics_der_means_vis, dynamics_der_stds_vis = vmap(self.dynamics.mean_and_std_eval_batch,
                                                             in_axes=(None, 0, 0))(self.dynamics_model,
                                                                                   data.visualization_data.xs,
                                                                                   data.visualization_data.us)

        # Hallucinated states
        pred_states, pred_control = None, None
        if self.offline_planning_data is not None:
            def evaluate_prediction(offline_plan: OfflinePlanningData, vis_data: VisualisationData):
                spline_states = MultivariateSpline(offline_plan.ts.reshape(-1), offline_plan.xs)
                spline_controls = MultivariateSpline(offline_plan.ts.reshape(-1), offline_plan.us)
                return spline_states(vis_data.ts), spline_controls(vis_data.ts)

            pred_states, pred_control = vmap(evaluate_prediction, in_axes=(0, 0))(self.offline_planning_data,
                                                                                  data.visualization_data)

        plot_data = PlotData(
            dynamics_der_means=dynamics_der_means_vis,
            dynamics_der_vars=dynamics_der_stds_vis ** 2,
            actual_actions=data.visualization_data.us,
            visualization_times=data.visualization_data.ts,
            observation_times=data.observation_data.ts,
            observations=data.observation_data.xs_dot_noise,
            gt_states_vis=data.visualization_data.xs,
            gt_der_vis=data.visualization_data.xs_dot_true,
            prediction_states=pred_states,
            predicted_actions=pred_control,
        )

        # Save & Plot data
        directory = os.path.join(wandb.run.dir, 'data')
        if not os.path.exists(directory):
            os.makedirs(directory)
        prefix = stage if stage == "" else stage + "_"
        data_path = os.path.join('data', prefix + 'data_episode_{}.pkl'.format(episode))
        data_vis_path = os.path.join('data', prefix + 'plot_data_episode_{}.pkl'.format(episode))
        with open(os.path.join(wandb.run.dir, data_path), 'wb') as handle:
            pickle.dump(data, handle)
        with open(os.path.join(wandb.run.dir, data_vis_path), 'wb') as handle:
            pickle.dump(plot_data, handle)

        wandb.save(os.path.join(wandb.run.dir, data_path), wandb.run.dir)
        wandb.save(os.path.join(wandb.run.dir, data_vis_path), wandb.run.dir)

        fig_dynamics_der, fig_state = self.plotter.plot(plot_data)
        fig_dynamics_der.tight_layout()
        fig_state.tight_layout()

        episode_prefix = 'Episode ' + str(episode) + stage + '/'
        if stage == 'pretraining':
            episode_prefix = 'Episode ' + str(episode) + ' ' + stage + '/'

        wandb.log({episode_prefix + 'DGM: Dynamics derivatives': wandb.Image(fig_dynamics_der),
                   episode_prefix + 'DGM: State': wandb.Image(fig_state)})
        plt.close('all')
