from typing import Any

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.main.data_stats import DataStats, DataLearn
from cucrl.utils.classes import NumberTrainPoints

pytree = Any


class Objectives:
    def __init__(self, dynamics: AbstractDynamics):
        self.dynamics = dynamics

    def dynamics_training(self, parameters, stats, data: DataLearn, data_stats: DataStats, keys,
                          num_train_points: NumberTrainPoints):
        dynamics_pretraining_loss, updated_states_dynamics = self.dynamics.loss(
            parameters["dynamics"], stats['dynamics'], data.dynamics_data.xs, data.dynamics_data.us,
            data.dynamics_data.xs_dot, data.dynamics_data.xs_dot_std, data_stats, num_train_points.dynamics,
            keys.step_key)
        objective = dynamics_pretraining_loss
        stats['dynamics'] = updated_states_dynamics
        return objective, stats
