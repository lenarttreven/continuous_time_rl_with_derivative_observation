from typing import NamedTuple, Dict, List, Callable, Tuple

import jax
from jax import random, numpy as jnp

from cucrl.schedules.learning_rate import LearningRateType
from cucrl.utils.representatives import BetaType, BNNTypes, BatchStrategy
from cucrl.utils.representatives import Optimizer, Dynamics, SimulatorType, TimeHorizonType
from cucrl.utils.representatives import SmootherType, ExplorationStrategy, DynamicsTracking, MinimizationMethod


class Scaling(NamedTuple):
    state_scaling: jnp.ndarray
    control_scaling: jnp.ndarray
    time_scaling: jnp.ndarray


class BatchSize(NamedTuple):
    dynamics: int


class TerminationConfig(NamedTuple):
    episode_budget_running_cost: float = 2000.0
    limited_budget: bool = True
    max_state: jnp.ndarray | None = None


class SimulatorConfig(NamedTuple):
    scaling: Scaling
    simulator_type: SimulatorType
    simulator_params: Dict
    num_nodes: int
    num_int_step_between_nodes: int
    time_horizon: Tuple[float, float]
    termination_config: TerminationConfig = TerminationConfig()


class DataCollection(NamedTuple):
    data_generation_key: random.PRNGKeyArray
    noise: jnp.ndarray
    initial_conditions: List[jnp.ndarray]
    num_matching_points: int
    num_visualization_points: int


class DataGeneratorConfig(NamedTuple):
    control_dim: int
    state_dim: int
    simulator: SimulatorConfig
    data_collection: DataCollection


class LearningRate(NamedTuple):
    type: LearningRateType
    kwargs: Dict


class SmootherConfig(NamedTuple):
    type: SmootherType
    features: List[int]
    state_dim: int
    num_particles: int
    bandwidth_prior: float
    bandwidth_svgd: float


class DynamicsConfig(NamedTuple):
    type: Dynamics
    features: List[int]
    num_particles: int
    bandwidth_prior: float
    bandwidth_svgd: float
    bnn_type: BNNTypes = BNNTypes.DETERMINISTIC_ENSEMBLE


class SystemAssumptions(NamedTuple):
    l_f: jax.Array = jnp.array(1.0)
    l_pi: jax.Array = jnp.array(1.0)
    l_sigma: jax.Array = jnp.array(1.0)
    hallucination_error: jax.Array = jnp.array(10.0)


class TimeHorizonConfig(NamedTuple):
    type: TimeHorizonType = TimeHorizonType.FIXED
    init_horizon: float = 10.0


class MeasurementCollectionConfig(NamedTuple):
    batch_size_per_time_horizon: int = 3
    noise_std: float = 0.1
    time_horizon: TimeHorizonConfig = TimeHorizonConfig()
    num_interpolated_values: int = 100
    batch_strategy: BatchStrategy = BatchStrategy.MAX_KERNEL_DISTANCE_GREEDY


class OfflinePlanningConfig(NamedTuple):
    num_independent_runs: int = 3
    exploration_strategy: ExplorationStrategy = ExplorationStrategy.MEAN
    beta_exploration: BetaType = BetaType.GP
    minimization_method: MinimizationMethod = MinimizationMethod.ILQR_WITH_CEM


class OnlineTrackingConfig(NamedTuple):
    mpc_update_period: int = 1
    time_horizon: float = 2.0
    dynamics_tracking: DynamicsTracking = DynamicsTracking.MEAN


class PolicyConfig(NamedTuple):
    offline_planning: OfflinePlanningConfig = OfflinePlanningConfig()
    online_tracking: OnlineTrackingConfig = OnlineTrackingConfig()
    num_nodes: int = 100
    num_int_step_between_nodes: int = 10
    initial_control: float | jnp.ndarray | Callable = 0.0


class InteractionConfig(NamedTuple):
    time_horizon: Tuple[float, float] = (0, 10)
    angles_dim: List[int] = []
    policy: PolicyConfig = PolicyConfig()
    system_assumptions: SystemAssumptions = SystemAssumptions()
    measurement_collector: MeasurementCollectionConfig = MeasurementCollectionConfig()



class OptimizerConfig(NamedTuple):
    type: Optimizer
    learning_rate: LearningRate
    wd: float = 0.0


class OptimizersConfig(NamedTuple):
    batch_size: BatchSize
    dynamics_training: OptimizerConfig
    no_batching: bool = True


class LoggingConfig(NamedTuple):
    track_wandb: bool
    track_just_loss: bool
    visualization: bool


class ComparatorConfig(NamedTuple):
    num_discrete_points: int = 10


class RunConfig(NamedTuple):
    seed: int
    data_generator: DataGeneratorConfig
    dynamics: DynamicsConfig
    interaction: InteractionConfig
    optimizers: OptimizersConfig
    logging: LoggingConfig
    comparator: ComparatorConfig
