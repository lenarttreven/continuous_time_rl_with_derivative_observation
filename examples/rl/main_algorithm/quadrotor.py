import argparse
import os

import jax.numpy as jnp
import jax.random
from jax import random, jit
from jax.config import config

import wandb
from cucrl.main.config import (
    LearningRate,
    OptimizerConfig,
    OptimizersConfig,
    OfflinePlanningConfig,
)
from cucrl.main.config import (
    LoggingConfig,
    Scaling,
    TerminationConfig,
    BetasConfig,
    OnlineTrackingConfig,
    BatchSize,
)
from cucrl.main.config import (
    MeasurementCollectionConfig,
    TimeHorizonConfig,
    PolicyConfig,
    ComparatorConfig,
)
from cucrl.main.config import (
    RunConfig,
    DataGenerationConfig,
    DynamicsConfig,
    InteractionConfig,
)
from cucrl.main.learn_system import LearnSystem
from cucrl.schedules.betas import BetasType
from cucrl.schedules.learning_rate import LearningRateType
from cucrl.utils.euler_angles import move_frame
from cucrl.utils.helper_functions import namedtuple_to_dict, sample_func
from cucrl.utils.representatives import ExplorationStrategy, DynamicsTracking, BNNTypes
from cucrl.utils.representatives import (
    Optimizer,
    Dynamics,
    SimulatorType,
    NumericalComputation,
    Norm,
    BetaType,
)
from cucrl.utils.representatives import (
    TimeHorizonType,
    BatchStrategy,
    MinimizationMethod,
)

config.update("jax_enable_x64", True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_seed", type=int, default=0)
    args = parser.parse_args()

    data_generation_seed = args.data_seed

    seed = 0
    state_dim = 12
    action_dim = 4

    num_matching_points = 50
    num_visualization_points = 1000
    num_observation_points = 10

    my_initial_conditions = [
        jnp.array(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=jnp.float64,
        )
    ]

    time_horizon = (0.0, 15.0)

    beta = 1
    num_trajectories = len(my_initial_conditions)

    noise_scalar = 0.001
    my_stds_for_simulation = jnp.array(
        [noise_scalar for _ in range(state_dim)], dtype=jnp.float64
    )
    my_simulator_parameters = {"system_params": None}

    track_wandb = True
    track_just_loss = True
    debug = False
    visualization = True
    numerical_correction = 0

    variance = 1.0
    key = random.PRNGKey(0)

    @jit
    def kernel(x: jnp.ndarray, y: jnp.ndarray):
        assert x.shape == y.shape == ()
        return jnp.exp(-((x - y) ** 2) / (2 * variance))

    key, subkey = random.split(key)
    random_func = sample_func(
        kernel_func=kernel,
        key=subkey,
        n_dim=3,
        t_min=0,
        t_max=15,
        num_samples=100,
        max_value=0.1,
        decay_factor=0.0,
    )

    perturbation_func = sample_func(
        kernel_func=kernel,
        key=subkey,
        n_dim=1,
        t_min=0,
        t_max=15,
        num_samples=100,
        max_value=0.1,
        decay_factor=0.0,
    )

    def initial_control(state, t):
        t = t.reshape()
        x, y, z, xdot, ydot, zdot, phi, theta, psi, p, q, r = state
        kp = 5 * jnp.array([1, 1, 1], jnp.float64)
        kd = 2 * jnp.sqrt(kp)
        angles = jnp.array([zdot, phi, theta])
        omega = jnp.array([p, q, r])
        angles_dot = jnp.linalg.inv(move_frame(angles)) @ omega
        u = -kp * angles - kd * angles_dot
        # u += 0.2 * random.normal(key=key, shape=(3,))
        u += 1.0 * random_func(t)
        return jnp.concatenate([jnp.array([0.1458]) + 0.1 * perturbation_func(t), u])

    state_scaling = jnp.diag(
        jnp.array([1, 1, 1, 1, 1, 1, 10, 10, 1, 10, 10, 1], dtype=jnp.float64)
    )

    run_config = RunConfig(
        seed=seed,
        data_generation=DataGenerationConfig(
            scaling=Scaling(
                state_scaling=state_scaling,
                control_scaling=jnp.eye(action_dim),
                time_scaling=jnp.ones(shape=(1,)),
            ),
            data_generation_key=jax.random.PRNGKey(data_generation_seed),
            simulator_step_size=0.001,
            simulator_type=SimulatorType.QUADROTOR_EULER,
            simulator_params=my_simulator_parameters,
            noise=my_stds_for_simulation,
            initial_conditions=my_initial_conditions,
            time_horizon=time_horizon,
            num_matching_points=num_matching_points,
            num_visualization_points=num_visualization_points,
            control_dim=action_dim,
            state_dim=state_dim,
            termination_config=TerminationConfig(
                episode_budget_running_cost=1500.0,
                limited_budget=False,
                max_state=100 * jnp.ones(shape=(state_dim,)),
            ),
        ),
        dynamics=DynamicsConfig(
            type=Dynamics.GP,
            features=[64, 64, 64],
            num_particles=10,
            bandwidth_prior=3.0,
            bandwidth_svgd=0.2,
            bnn_type=BNNTypes.DETERMINISTIC_ENSEMBLE,
        ),
        interaction=InteractionConfig(
            time_horizon=time_horizon,
            policy=PolicyConfig(
                online_tracking=OnlineTrackingConfig(
                    mpc_dt=0.02,
                    time_horizon=3.0,
                    num_nodes=50,
                    dynamics_tracking=DynamicsTracking.MEAN,
                ),
                offline_planning=OfflinePlanningConfig(
                    num_independent_runs=4,
                    exploration_strategy=ExplorationStrategy.OPTIMISTIC_ETA_TIME,
                    exploration_norm=Norm.L_INF,
                    numerical_method=NumericalComputation.LGL,
                    num_nodes=100,
                    beta_exploration=BetaType.GP,
                    minimization_method=MinimizationMethod.ILQR,
                ),
                initial_control=initial_control,
            ),
            angles_dim=[
                6,
                7,
                8,
            ],
            measurement_collector=MeasurementCollectionConfig(
                batch_size_per_time_horizon=10,
                batch_strategy=BatchStrategy.MAX_DETERMINANT_GREEDY,
                noise_std=0.0,
                time_horizon=TimeHorizonConfig(
                    type=TimeHorizonType.FIXED, init_horizon=time_horizon[1]
                ),
                num_hallucination_nodes=100,
                num_interpolated_values=1000,
            ),
        ),
        betas=BetasConfig(
            type=BetasType.CONSTANT, kwargs={"value": beta, "num_dim": state_dim}
        ),
        optimizers=OptimizersConfig(
            no_batching=False,
            batch_size=BatchSize(dynamics=64),
            dynamics_training=OptimizerConfig(
                type=Optimizer.ADAM,
                wd=0.1,
                learning_rate=LearningRate(
                    type=LearningRateType.PIECEWISE_CONSTANT,
                    kwargs={"boundaries": [10**4], "values": [0.1, 0.01]},
                ),
            ),
        ),
        logging=LoggingConfig(
            track_wandb=track_wandb,
            track_just_loss=track_just_loss,
            visualization=visualization,
        ),
        comparator=ComparatorConfig(num_discrete_points=num_observation_points),
    )

    if track_wandb:
        home_folder = os.getcwd()
        home_folder = "/".join(home_folder.split("/")[:4])
        group_name = "Testing"
        if home_folder == "/cluster/home/trevenl":
            wandb.init(
                dir="/cluster/scratch/trevenl",
                project="QuadrotorEuler",
                group=group_name,
                config=namedtuple_to_dict(run_config),
            )
        else:
            wandb.init(
                project="QuadrotorEuler",
                group=group_name,
                config=namedtuple_to_dict(run_config),
            )
        config = wandb.config

    model = LearnSystem(run_config)
    model.run_episodes(num_episodes=20, num_iter_training=8000)
    wandb.finish()
