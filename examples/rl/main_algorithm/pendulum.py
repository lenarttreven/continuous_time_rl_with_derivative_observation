import argparse
import os

import jax.numpy as jnp
import jax.random
from jax.config import config

import wandb
from cucrl.main.config import LearningRate, OptimizerConfig, OptimizersConfig, OfflinePlanningConfig
from cucrl.main.config import LoggingConfig, Scaling, TerminationConfig, OnlineTrackingConfig, BatchSize
from cucrl.main.config import MeasurementCollectionConfig, TimeHorizonConfig, PolicyConfig, ComparatorConfig
from cucrl.main.config import RunConfig, DataGeneratorConfig, DynamicsConfig, InteractionConfig, SimulatorConfig
from cucrl.main.config import TimeHorizon, DataCollection
from cucrl.main.learn_system import LearnSystem
from cucrl.schedules.learning_rate import LearningRateType
from cucrl.utils.helper_functions import namedtuple_to_dict
from cucrl.utils.representatives import ExplorationStrategy, DynamicsTracking, BNNTypes
from cucrl.utils.representatives import Optimizer, Dynamics, SimulatorType, BetaType
from cucrl.utils.representatives import TimeHorizonType, BatchStrategy

config.update('jax_enable_x64', True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_seed', type=int, default=0)
    args = parser.parse_args()

    data_generation_seed = args.data_seed

    seed = 0
    num_matching_points = 50
    num_visualization_points = 1000

    initial_conditions = [jnp.array([jnp.pi, 0])]
    time_horizon = TimeHorizon(t_min=0, t_max=10)
    noise_scalar = 0.01
    stds_for_simulation = jnp.array([noise_scalar, noise_scalar], dtype=jnp.float64)
    simulator_parameters = {'system_params': jnp.array([5.0, 9.81], jnp.float64)}

    track_wandb = True
    track_just_loss = True
    debug = False
    visualization = True
    numerical_correction = 0

    state_dim = 2
    action_dim = 1
    num_trajectories = len(initial_conditions)


    def initial_control(x, t):
        return 0.1 * jnp.sin(t).reshape(1, )


    run_config = RunConfig(
        seed=seed,
        data_generator=DataGeneratorConfig(
            control_dim=action_dim,
            state_dim=state_dim,
            simulator=SimulatorConfig(
                scaling=Scaling(state_scaling=jnp.diag(jnp.array([1.0, 2.0])),
                                control_scaling=jnp.eye(action_dim),
                                time_scaling=jnp.ones(shape=(1,))),
                simulator_type=SimulatorType.PENDULUM,
                simulator_params=simulator_parameters,
                num_control_nodes=100,
                num_int_step_between_nodes=10,
                time_horizon=time_horizon,
                termination_config=TerminationConfig(episode_budget_running_cost=1500.0,
                                                     limited_budget=False,
                                                     max_state=100 * jnp.ones(shape=(state_dim,))),

            ),
            data_collection=DataCollection(
                data_generation_key=jax.random.PRNGKey(data_generation_seed),
                initial_conditions=initial_conditions,
                num_matching_points=num_matching_points,
                num_visualization_points=num_visualization_points,
                noise=stds_for_simulation,
            ),
        ),
        dynamics=DynamicsConfig(
            type=Dynamics.GP,
            features=[64, 64, 64],
            num_particles=10,
            bandwidth_prior=3.0,
            bandwidth_svgd=0.2,
            bnn_type=BNNTypes.DETERMINISTIC_ENSEMBLE
        ),
        interaction=InteractionConfig(
            time_horizon=time_horizon,
            policy=PolicyConfig(
                num_control_steps=100,
                num_int_step_between_nodes=10,
                online_tracking=OnlineTrackingConfig(
                    mpc_update_period=1,
                    control_steps=20,
                    dynamics_tracking=DynamicsTracking.MEAN
                ),
                offline_planning=OfflinePlanningConfig(
                    num_independent_runs=4,
                    exploration_strategy=ExplorationStrategy.OPTIMISTIC_ETA_TIME,
                    beta_exploration=BetaType.GP
                ),
                initial_control=initial_control,
            ),
            angles_dim=[0, ],
            measurement_collector=MeasurementCollectionConfig(
                batch_size_per_time_horizon=10,
                batch_strategy=BatchStrategy.MAX_DETERMINANT_GREEDY,
                noise_std=0.0,
                time_horizon=TimeHorizonConfig(type=TimeHorizonType.FIXED, init_horizon=10.0),
                num_interpolated_values=1000,
            )
        ),
        optimizers=OptimizersConfig(
            no_batching=True,
            batch_size=BatchSize(dynamics=64),
            dynamics_training=OptimizerConfig(type=Optimizer.ADAM, wd=0.0,
                                              learning_rate=LearningRate(type=LearningRateType.PIECEWISE_CONSTANT,
                                                                         kwargs={'boundaries': [10 ** 4],
                                                                                 'values': [0.1, 0.01]}, )
                                              ),
        ),
        logging=LoggingConfig(track_wandb=track_wandb, track_just_loss=track_just_loss, visualization=visualization),
        comparator=ComparatorConfig(num_discrete_points=10)
    )

    if track_wandb:
        home_folder = os.getcwd()
        home_folder = '/'.join(home_folder.split('/')[:4])
        group_name = 'Testing'
        if home_folder == '/cluster/home/trevenl':
            wandb.init(
                dir='/cluster/scratch/trevenl',
                project='Pendulum',
                group=group_name,
                config=namedtuple_to_dict(run_config),
            )
        else:
            wandb.init(
                project='Pendulum',
                group=group_name,
                config=namedtuple_to_dict(run_config),
            )
        config = wandb.config

    model = LearnSystem(run_config)
    model.run_episodes(num_episodes=20, num_iter_training=8000)
    wandb.finish()
