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
from cucrl.main.config import RunConfig, SmootherConfig, DynamicsConfig, OptimizersConfig, DataGenerationConfig
from cucrl.main.data_stats import DataLoader
from cucrl.main.data_stats import DataStats, Normalizer, DataLearn, DynamicsData, Stats, SmoothingData, MatchingData
from cucrl.main.handlers import DataGenerator, DynamicsDataManager
from cucrl.main.handlers import Keys, DataRepr, VisualisationData
from cucrl.objectives.objectives import Objectives
from cucrl.offline_planner.offline_planner import get_offline_planner
from cucrl.optimal_cost.best_possible_discrete_time import BestPossibleDiscreteAlgorithm
from cucrl.optimal_cost.optimal_cost import OptimalCost
from cucrl.plotter.plotter import Plotter
from cucrl.schedules.betas import get_betas
from cucrl.schedules.learning_rate import get_learning_rate
from cucrl.simulator.simulator_costs import get_simulator_costs
from cucrl.smoother.smoother import SmootherFactory
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
        self._prepare_dimensions(config.data_generation)

        self.angle_layer = AngleLayerDynamics(state_dim=self.state_dim, control_dim=self.control_dim,
                                              angles_dim=config.interaction.angles_dim,
                                              state_scaling=self.config.data_generation.scaling.state_scaling)
        self.normalizer = Normalizer(state_dim=self.state_dim, action_dim=self.control_dim,
                                     tracking_c=None, angle_layer=self.angle_layer)
        init_stats_state = Stats(jnp.zeros(shape=(self.state_dim,)), jnp.ones(shape=(self.state_dim,)))
        init_stats_control = Stats(jnp.zeros(shape=(self.control_dim,)), jnp.ones(shape=(self.control_dim,)))
        init_stats_time = Stats(jnp.zeros(shape=(1,)), jnp.ones(shape=(1,)))
        angles_shape = self.angle_layer.angle_layer(jnp.ones(shape=(self.state_dim,))).shape
        init_stats_after_angle_layer = Stats(jnp.zeros(shape=angles_shape), jnp.ones(shape=angles_shape))

        self.data_stats = DataStats(ic_stats=init_stats_state, ts_stats=init_stats_time, ys_stats=init_stats_state,
                                    xs_stats=init_stats_state, us_stats=init_stats_control,
                                    dot_xs_stats=init_stats_state, xs_after_angle_layer=init_stats_after_angle_layer)
        # Prepare pool for vector field data
        self.vector_field_data = DynamicsDataManager(self.state_dim, self.control_dim)
        self.simulator_costs = get_simulator_costs(config.data_generation.simulator_type,
                                                   scaling=self.config.data_generation.scaling)

        # Prepare randomness
        self.current_rng = jax.random.PRNGKey(config.seed)
        self.current_rng, self.episode_key = random.split(self.current_rng)
        # Prepare data_loader
        self.data_batcher = DataLoader(batch_size=config.optimizers.batch_size,
                                       no_batching=config.optimizers.no_batching)
        # Prepare black box data
        self.offline_planning_data = None
        # Prepare smoother model.
        self._prepare_smoother(config.smoother)
        # Prepare dynamics
        self.dynamics_config = config.dynamics
        self._prepare_dynamics(self.dynamics_config)
        # Prepare control
        self.interaction = config.interaction
        # Prepare trajectory optimization
        self._prepare_offline_planner()
        # Prepare feedback controller
        self._prepare_controller(self.interaction, config.data_generation.initial_conditions)
        # Prepare data generator
        self.data_generator = DataGenerator(data_generation=config.data_generation, interactor=self.policy)
        # Prepare betas
        self._prepare_betas(config.betas)
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
        self.beta_exploration = BetaExploration(delta=0.1, state_dim=self.state_dim, rkhs_bound=5,
                                                type=self.config.interaction.policy.offline_planning.beta_exploration)
        optimal_cost = OptimalCost(simulator_dynamics=self.data_generator.simulator.simulator_dynamics,
                                   simulator_costs=self.simulator_costs,
                                   time_horizon=self.config.data_generation.time_horizon)
        self.optimal_cost = [optimal_cost.solve(ic) for ic in self.data_generator.initial_conditions]

        best_possible_discrete = BestPossibleDiscreteAlgorithm(
            simulator_dynamics=self.data_generator.simulator.simulator_dynamics,
            simulator_costs=self.simulator_costs,
            time_horizon=self.config.data_generation.time_horizon,
            num_nodes=self.config.comparator.num_discrete_points)
        self.best_possible_discrete_cost = [best_possible_discrete.get_optimal_cost(ic) for ic in
                                            self.data_generator.initial_conditions]

        self.dynamics_model = None

    def _prepare_dimensions(self, data_generation: DataGenerationConfig):
        self.data_generation_key = data_generation.data_generation_key
        self.control_dim = data_generation.control_dim
        self.state_dim = data_generation.state_dim
        self.num_trajectories = len(data_generation.initial_conditions)
        self.num_obs_per_episode = sum(map(len, data_generation.initial_conditions))
        self.time_horizon = data_generation.time_horizon
        self.noise_stds = data_generation.noise

    def _prepare_betas(self, betas):
        self.betas_joint_training = get_betas(betas.type, betas.kwargs)

    def _prepare_smoother(self, smoother: SmootherConfig):
        time_smoother = time.time()
        self.smoother = SmootherFactory().make_smoother(smoother_type=smoother.type, smoother_config=smoother,
                                                        state_dim=self.state_dim, noise_stds=self.noise_stds,
                                                        normalizer=self.normalizer)
        print("Time for smoother preparation: ", time.time() - time_smoother)

    def _prepare_dynamics(self, dynamics: DynamicsConfig):
        time_dynamics = time.time()
        self.dynamics = get_dynamics(dynamics_model=dynamics.type, state_dim=self.state_dim,
                                     action_dim=self.control_dim, normalizer=self.normalizer,
                                     dynamics_config=dynamics,
                                     angle_layer=self.angle_layer,
                                     measurement_collection_config=self.config.interaction.measurement_collector)
        print("Time for dynamics preparation: ", time.time() - time_dynamics)

    def _prepare_controller(self, control_options, initial_condition):
        self.policy = get_interactor(self.state_dim, self.control_dim, self.dynamics, initial_condition,
                                     self.normalizer,
                                     self.angle_layer, control_options, self.offline_planner,
                                     self.config.data_generation.scaling)

    def _initialize_parameters(self):
        time_parameters = time.time()
        # Get new random keys
        self.current_rng, *keys = jax.random.split(self.current_rng, 3)

        # Initialize parameters
        params_dynamics, stats_dynamics = self.dynamics.initialize_parameters(keys[0])
        params_smoother, stats_smoother = self.smoother.initialize_parameters(keys[1])

        self.parameters = {"smoother": params_smoother, "dynamics": params_dynamics}
        self.stats = {'smoother': stats_smoother, 'dynamics': stats_dynamics}

        self.old_parameters = copy.deepcopy(self.parameters)
        self.old_stats = copy.deepcopy(self.stats)

        # Count number of parameters of the model
        self.num_dynamics_parameters = 0
        self.num_smoother_parameters = 0

        for leave in tree_leaves(params_dynamics):
            self.num_dynamics_parameters += leave.size
        for leave in tree_leaves(params_smoother):
            self.num_smoother_parameters += leave.size

        self.num_parameters = self.num_smoother_parameters + self.num_dynamics_parameters
        print("Time to initialize parameters", time.time() - time_parameters)

    def _reset_parameters_smoother(self, key):
        params_smoother, states_smoother = self.smoother.initialize_parameters(key)
        self.parameters['smoother'] = params_smoother
        self.stats['smoother'] = states_smoother

    def _reset_parameters_dynamics(self, key):
        params_dynamics, states_dynamics = self.dynamics.initialize_parameters(key)
        self.parameters['dynamics'] = params_dynamics
        self.stats['dynamics'] = states_dynamics

    def _prepare_objectives(self):
        time_objective_builder = time.time()
        # Build objectives - learn_dynamics and optimize_policy
        objectives = Objectives(smoother=self.smoother, dynamics=self.dynamics)

        self.pretraining_smoother = objectives.pretraining_smoother
        self.pretraining_dynamics = objectives.pretraining_dynamics
        self.joint_training = objectives.joint_training

        self.values_and_grad_pretraining_smoother = jit(value_and_grad(self.pretraining_smoother, 0, has_aux=True))
        self.values_and_grad_pretraining_dynamics = jit(value_and_grad(self.pretraining_dynamics, 0, has_aux=True))
        self.values_and_grad_joint_training = jit(value_and_grad(self.joint_training, 0, has_aux=True))

        print("Time to prepare objective builder", time.time() - time_objective_builder)

    def _prepare_offline_planner(self):
        offline_planer_config = self.interaction.policy.offline_planning
        planner_class = get_offline_planner(offline_planer_config.exploration_strategy)
        self.offline_planner = planner_class(
            state_dim=self.state_dim, control_dim=self.control_dim, num_nodes=offline_planer_config.num_nodes,
            numerical_method=offline_planer_config.numerical_method, time_horizon=self.time_horizon,
            dynamics=self.dynamics, simulator_costs=self.simulator_costs,
            exploration_strategy=offline_planer_config.exploration_strategy,
            exploration_norm=offline_planer_config.exploration_norm)

    def _prepare_optimizer(self, optimizer: OptimizersConfig):
        # Prepare learning rate
        self.learning_rate_pretraining_smoother = get_learning_rate(
            optimizer.pretraining_smoother.learning_rate.type,
            optimizer.pretraining_smoother.learning_rate.kwargs)
        self.learning_rate_pretraining_dynamics = get_learning_rate(
            optimizer.pretraining_dynamics.learning_rate.type,
            optimizer.pretraining_dynamics.learning_rate.kwargs)
        self.learning_rate_joint_training = get_learning_rate(
            optimizer.joint_training.learning_rate.type,
            optimizer.joint_training.learning_rate.kwargs)
        # Prepare optimizer for learning dynamics
        if optimizer.pretraining_smoother.type == Optimizer.ADAM:
            self.optimizer_pretraining_smoother = optax.adamw
        elif optimizer.pretraining_smoother.type == Optimizer.SGD:
            self.optimizer_pretraining_smoother = optax.sgd
        if optimizer.pretraining_dynamics.type == Optimizer.ADAM:
            self.optimizer_pretraining_dynamics = optax.adamw
        elif optimizer.pretraining_dynamics.type == Optimizer.SGD:
            self.optimizer_pretraining_dynamics = optax.sgd

        if optimizer.joint_training.type == Optimizer.ADAM:
            self.optimizer_joint_training = optax.adamw
        elif optimizer.joint_training.type == Optimizer.SGD:
            self.optimizer_joint_training = optax.sgd

    def generate_data(self) -> Tuple[DataRepr, MeasurementSelection]:
        # Generate noisy trajectories following the dynamics with the given policy
        self.data_generation_key, key = random.split(self.data_generation_key)
        return self.data_generator.generate_trajectories(key)

    def prepare_pretraining_smoother(self):
        if self.track_wandb:
            wandb.define_metric("x_axis/step")
            # set all other train/ metrics to use this step
            wandb.define_metric("Smoother pretraining/*", step_metric="x_axis/step", summary="last")
            wandb.define_metric('Smoother pretraining/iter_time', step_metric="x_axis/step", summary='mean')
        tx_smoother_pretraining = self.optimizer_pretraining_smoother(
            self.learning_rate_pretraining_smoother,
            weight_decay=self.config.optimizers.pretraining_smoother.wd)

        def prepare_parameters_and_stats(key):
            # Reset smoother and dynamics parameters
            key, subkey = jax.random.split(key)
            self._reset_parameters_smoother(subkey)
            # Split parameters to trainable parameters and policy parameters
            params_train = {'smoother': self.parameters['smoother']}
            stats_train = {'smoother': self.stats['smoother']}
            # Watch trainable parameters with the optimizer
            opt_state = tx_smoother_pretraining.init(params_train)
            return params_train, stats_train, opt_state

        # Define optimization step
        @jit
        def do_step(step, parameters, stats, opt_state, data, data_stats, keys, num_train_points):
            (current_loss, new_stats), params_grad = self.values_and_grad_pretraining_smoother(
                parameters, stats, data, data_stats, keys, num_train_points)

            updates, opt_state = tx_smoother_pretraining.update(params_grad, opt_state, parameters)
            parameters_train_new = optax.apply_updates(parameters, updates)
            return current_loss, opt_state, parameters_train_new, new_stats

        # Run optimization for predefine number of steps
        def run_optimization(number_of_steps, params, stats, opt_state, data, data_stats, episode, keys):
            self.optimize(do_step=do_step, num_steps=number_of_steps, params=params, stats=stats,
                          opt_state=opt_state, data=data, data_stats=data_stats,
                          log_name='Smoother pretraining/Full objective', episode=episode, keys=keys)

        return prepare_parameters_and_stats, run_optimization

    def prepare_pretraining_dynamics(self):
        if self.track_wandb:
            wandb.define_metric("x_axis/step")
            # set all other train/ metrics to use this step
            wandb.define_metric("Dynamics pretraining/*", step_metric="x_axis/step", summary="last")
            wandb.define_metric('Dynamics pretraining/iter_time', step_metric="x_axis/step", summary='mean')
        tx_dynamics_pretraining = self.optimizer_pretraining_dynamics(
            self.learning_rate_pretraining_dynamics,
            weight_decay=self.config.optimizers.pretraining_dynamics.wd)

        def prepare_parameters_and_stats(key):
            # Reset smoother and dynamics parameters
            key, subkey = jax.random.split(key)
            self._reset_parameters_dynamics(subkey)
            # Split parameters to trainable parameters and policy parameters
            params_train = {'dynamics': self.parameters['dynamics']}
            stats_train = {'dynamics': self.stats['dynamics']}
            # Watch trainable parameters with the optimizer
            opt_state = tx_dynamics_pretraining.init(params_train)
            return params_train, stats_train, opt_state

        # Define optimization step
        @jit
        def do_step(step, parameters, stats, opt_state, data, data_stats, keys, num_train_points):
            (current_loss, new_stats), params_grad = self.values_and_grad_pretraining_dynamics(
                parameters, stats, data, data_stats, keys, num_train_points)

            updates, opt_state = tx_dynamics_pretraining.update(params_grad, opt_state, parameters)
            parameters_train_new = optax.apply_updates(parameters, updates)
            return current_loss, opt_state, parameters_train_new, new_stats

        # Run optimization for predefine number of steps
        def run_optimization(number_of_steps, params, stats, opt_state, data, data_stats, episode, keys):
            self.optimize(do_step=do_step, num_steps=number_of_steps, params=params, stats=stats,
                          opt_state=opt_state, data=data, data_stats=data_stats,
                          log_name='Dynamics pretraining/Full objective', episode=episode, keys=keys)

        return prepare_parameters_and_stats, run_optimization

    def prepare_joint_training(self):
        if self.track_wandb:
            wandb.define_metric("x_axis/step")
            # set all other train/ metrics to use this step
            wandb.define_metric("Learning dynamics/*", step_metric="x_axis/step", summary="last")
            wandb.define_metric('Learning dynamics/iter_time', step_metric="x_axis/step", summary='mean')
        tx_learn_dynamics = self.optimizer_joint_training(self.learning_rate_joint_training,
                                                          weight_decay=self.config.optimizers.joint_training.wd)

        def prepare_parameters_and_stats(_):
            # Prepare trainable parameters
            params_train = self.parameters
            stats_train = self.stats
            # Watch trainable parameters with the optimizer
            opt_state = tx_learn_dynamics.init(params_train)
            return params_train, stats_train, opt_state

        # Define optimization step
        @jit
        def do_step(step, parameters, stats, opt_state, data, data_stats, keys, num_train_points):
            (loss, new_stats), params_grad = self.values_and_grad_joint_training(parameters, stats, data, data_stats,
                                                                                 self.betas_joint_training(step), keys,
                                                                                 num_train_points)
            updates, opt_state = tx_learn_dynamics.update(params_grad, opt_state, parameters)
            parameters_train_new = optax.apply_updates(parameters, updates)
            return loss, opt_state, parameters_train_new, new_stats

        # Run optimization for predefine number of steps
        def run_optimization(number_of_steps, params, stats, opt_state, data, data_stats, episode, keys):
            self.optimize(do_step=do_step, num_steps=number_of_steps, params=params,
                          stats=stats, opt_state=opt_state, data=data, data_stats=data_stats,
                          log_name='Learning dynamics/Full objective', episode=episode, keys=keys)

        return prepare_parameters_and_stats, run_optimization

    def optimize(self, do_step, num_steps, params, stats, opt_state, data: DataLearn, data_stats, log_name: str,
                 keys: Keys, episode: int = 0):
        # Set the timer
        current_time, iter_time = time.time(), 0
        initial_time = current_time
        self.current_rng, key = random.split(self.current_rng)
        if data.dynamics_data is not None:
            num_train_points_dynamics = data.dynamics_data.xs.shape[0]
            num_train_points_matching = data.matching_data.ts.shape[0]
            num_train_points_smoothing = data.smoothing_data.ys.shape[0]
        else:
            num_train_points_dynamics = 0
            num_train_points_matching = data.matching_data.ts.shape[0]
            num_train_points_smoothing = data.smoothing_data.ys.shape[0]

        num_train_points = NumberTrainPoints(dynamics=num_train_points_dynamics, matching=num_train_points_matching,
                                             smoother=num_train_points_smoothing)
        data_loader = self.data_batcher.prepare_loader(dataset=data, key=self.current_rng,
                                                       no_dynamics_data=data.dynamics_data is None)

        # for step in range(num_steps):
        for step, data_batch in enumerate(data_loader):
            # Print times of first 10 steps to see the time which we need for training
            if step >= num_steps:
                break
            cur_data = DataLearn(*data_batch)
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
    def data_for_learning(data: DataRepr, vf_data: DynamicsData | None = None) -> DataLearn:
        return DataLearn(
            smoothing_data=SmoothingData(
                ts=jnp.concatenate(data.observation_data.ts),
                x0s=jnp.concatenate(data.observation_data.x0s),
                ys=jnp.concatenate(data.observation_data.ys),
                us=jnp.concatenate(data.observation_data.us),
            ),
            matching_data=MatchingData(
                ts=jnp.concatenate(data.matching_data.ts),
                x0s=jnp.concatenate(data.matching_data.x0s),
                us=jnp.concatenate(data.matching_data.us),
            ),
            dynamics_data=vf_data
        )

    def log_measurement_selection(self, episode: int, measurement_selection: MeasurementSelection):
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

    def run_episodes(self, num_episodes: int, num_iter_pretraining: int, num_iter_joint_training: int):
        prepare_params_pretraining_smoother, run_optimization_pretraining_smoother = self.prepare_pretraining_smoother()
        prepare_params_pretraining_dynamics, run_optimization_pretraining_dynamics = self.prepare_pretraining_dynamics()
        prepare_params_joint_training, run_optimization_joint_training = self.prepare_joint_training()
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
                                                   xs=jnp.ones(shape=(1, self.state_dim)),
                                                   us=jnp.ones(shape=(1, self.control_dim)),
                                                   xs_dot_std=jnp.ones(shape=(1, self.state_dim)),
                                                   xs_dot=jnp.ones(shape=(1, self.state_dim))),
                                               calibration_alpha=jnp.ones(shape=(self.state_dim,)))
                self.data_generator.simulator.interactor.update(dynamics_model=dynamics_model, key=key)
            data, measurement_selection = self.generate_data()
            if episode >= 1:
                self.log_measurement_selection(episode, measurement_selection)
            self.cost_logging(episode, data.visualization_data, predicted_data=self.offline_planning_data)
            # Pretraining
            print(colored('Pretraining', 'green'))

            self.current_rng, *key_pretraining = jax.random.split(self.current_rng, 3)
            self.current_rng, step_key = jax.random.split(self.current_rng)
            keys = Keys(episode_key=self.episode_key, step_key=step_key)
            # Pretrain smoother
            params_train, stats_train, opt_state = prepare_params_pretraining_smoother(key_pretraining[0])
            current_data = self.data_for_learning(data)
            data_stats = self.normalizer.compute_stats(current_data)
            self.data_stats = data_stats

            self.current_rng, key = random.split(self.current_rng)

            run_optimization_pretraining_smoother(number_of_steps=num_iter_pretraining, params=params_train,
                                                  stats=stats_train, opt_state=opt_state, data=current_data,
                                                  data_stats=data_stats, episode=episode, keys=keys)

            self.current_rng, subkey_for_vf_data = random.split(self.current_rng)
            sampled_data = self.smoother.sample_vector_field_data(self.parameters['smoother'], self.stats['smoother'],
                                                                  current_data.smoothing_data.ts,
                                                                  current_data.smoothing_data.ys,
                                                                  current_data.smoothing_data.x0s, self.data_stats,
                                                                  subkey_for_vf_data)

            self.vector_field_data.add_data_to_training_pool(x=sampled_data.xs, u=current_data.smoothing_data.us,
                                                             x_dot=sampled_data.xs_dot,
                                                             std_x_dot=sampled_data.std_xs_dot)
            vector_field_data = DynamicsData(**deepcopy(self.vector_field_data.training_pool))

            # Pretrain dynamics
            params_train, stats_train, opt_state = prepare_params_pretraining_dynamics(key_pretraining[0])
            current_data = self.data_for_learning(data, vector_field_data)
            data_stats = self.normalizer.compute_stats(current_data)
            self.data_stats = data_stats

            self.current_rng, key = random.split(self.current_rng)
            run_optimization_pretraining_dynamics(number_of_steps=num_iter_pretraining, params=params_train,
                                                  stats=stats_train, opt_state=opt_state, data=current_data,
                                                  data_stats=data_stats, episode=episode, keys=keys)

            self.dynamics_model = DynamicsModel(params=self.parameters['dynamics'], model_stats=self.stats['dynamics'],
                                                data_stats=self.data_stats, episode=episode + 1,
                                                calibration_alpha=jnp.ones(shape=(self.state_dim,)),
                                                history=vector_field_data)

            if self.visualization and self.track_wandb:
                self.visualize_data(data=data, data_stats=self.data_stats, episode=episode, stage='pretraining')

            # Learn dynamics
            print(colored('Learning system', 'green'))
            self.current_rng, key_params_joint_training = jax.random.split(self.current_rng)
            params_train, stats_train, opt_state = prepare_params_joint_training(key_params_joint_training)

            self.current_rng, step_key = jax.random.split(self.current_rng)
            keys = Keys(episode_key=self.episode_key, step_key=step_key)
            run_optimization_joint_training(number_of_steps=num_iter_joint_training, params=params_train,
                                            stats=stats_train, opt_state=opt_state, data=current_data,
                                            data_stats=data_stats, episode=episode, keys=keys)

            sampled_data = self.smoother.sample_vector_field_data(self.parameters['smoother'], self.stats['smoother'],
                                                                  current_data.smoothing_data.ts,
                                                                  current_data.smoothing_data.ys,
                                                                  current_data.smoothing_data.x0s, self.data_stats,
                                                                  subkey_for_vf_data)

            self.current_rng, subkey_test_data = random.split(self.current_rng)

            sampled_data_test = self.smoother.sample_vector_field_data(self.parameters['smoother'],
                                                                       self.stats['smoother'],
                                                                       current_data.smoothing_data.ts,
                                                                       current_data.smoothing_data.ys,
                                                                       current_data.smoothing_data.x0s, self.data_stats,
                                                                       subkey_test_data)

            self.vector_field_data.add_data_to_test_pool(x=sampled_data_test.xs, u=current_data.smoothing_data.us,
                                                         x_dot=sampled_data_test.xs_dot,
                                                         std_x_dot=sampled_data_test.std_xs_dot)

            self.vector_field_data.add_data_to_permanent_pool(x=sampled_data.xs, u=current_data.smoothing_data.us,
                                                              x_dot=sampled_data.xs_dot,
                                                              std_x_dot=sampled_data.std_xs_dot)

            # Compute calibration
            dynamics_model = DynamicsModel(params=self.parameters['dynamics'], model_stats=self.stats['dynamics'],
                                           data_stats=self.data_stats, episode=episode + 1)

            calibration_dynamics_alpha = self.dynamics.calculate_calibration_alpha(
                dynamics_model=dynamics_model, xs=self.vector_field_data.test_pool['xs'],
                us=self.vector_field_data.test_pool['us'], xs_dot=self.vector_field_data.test_pool['xs_dot'],
                xs_dot_std=self.vector_field_data.test_pool['xs_dot_std'])

            # Here we need to add uncertainties for the beta exploration
            stds = vmap(self.dynamics.mean_and_std_eval_one, in_axes=(None, 0, 0))(self.dynamics_model, sampled_data.xs,
                                                                                   current_data.smoothing_data.us)[1]

            self.beta_exploration.update_info_gain(stds=stds, taus=sampled_data.std_xs_dot)

            self.vector_field_data.set_training_pool_to_permanent_pool()
            if self.visualization and self.track_wandb:
                self.visualize_data(data=data, data_stats=self.data_stats, episode=episode)
            # Trajectory optimization
            print(colored('Trajectory optimization', 'blue'))
            start_time_trajectory_optimization = time.time()
            self.current_rng, key = random.split(self.current_rng)

            dynamics_model = DynamicsModel(params=self.parameters['dynamics'], model_stats=self.stats['dynamics'],
                                           data_stats=self.data_stats, episode=episode + 1,
                                           beta=self.beta_exploration(num_episodes=num_episodes),
                                           history=DynamicsData(**deepcopy(self.vector_field_data.permanent_pool)),
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

    def visualize_data(self, data: DataRepr, data_stats: DataStats, episode: int,
                       stage: str = ''):
        smoother_posterior = vmap(self.smoother.posterior, in_axes=(None, None, 0, 0, None, None, None, None))(
            self.parameters["smoother"], self.stats['smoother'], data.visualization_data.ts,
            data.visualization_data.x0s, jnp.concatenate(data.observation_data.ts),
            jnp.concatenate(data.observation_data.x0s), jnp.concatenate(data.observation_data.ys), data_stats)

        state_std_vis = jnp.sqrt(smoother_posterior.xs_var)
        der_std_vis = jnp.sqrt(smoother_posterior.xs_dot_var)

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
            smoother_state_means=smoother_posterior.xs_mean,
            smoother_state_vars=state_std_vis ** 2,
            smoother_der_means=smoother_posterior.xs_dot_mean,
            smoother_der_vars=der_std_vis ** 2,
            dynamics_der_means=dynamics_der_means_vis,
            dynamics_der_vars=dynamics_der_stds_vis ** 2,
            actual_actions=data.visualization_data.us,
            visualization_times=data.visualization_data.ts,
            observation_times=data.observation_data.ts,
            observations=data.observation_data.ys,
            gt_states_vis=data.visualization_data.xs,
            gt_der_vis=data.visualization_data.xs_dot,
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

        fig_smoother_states, fig_smoother_der, fig_dynamics_der = self.plotter.plot(plot_data)
        fig_smoother_states.tight_layout()
        fig_smoother_der.tight_layout()
        fig_dynamics_der.tight_layout()

        episode_prefix = 'Episode ' + str(episode) + stage + '/'
        if stage == 'pretraining':
            episode_prefix = 'Episode ' + str(episode) + ' ' + stage + '/'

        wandb.log({episode_prefix + 'DGM: Smoother states': wandb.Image(fig_smoother_states),
                   episode_prefix + 'DGM: Smoother derivatives': wandb.Image(fig_smoother_der),
                   episode_prefix + 'DGM: Dynamics derivatives': wandb.Image(fig_dynamics_der)})

        plt.close('all')
