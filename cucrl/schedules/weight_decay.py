from enum import Enum, auto
from typing import Callable

from jax.example_libraries.optimizers import constant, piecewise_constant

from cucrl.schedules.betas import polynomial_decay, indicator

Schedule = Callable[[int], float]


class WeightDecayType(Enum):
    PIECEWISE_CONSTANT = auto()
    CONSTANT = auto()
    POLYNOMIAL_DECAY = auto()
    TRANSITION_BETWEEN_VALUES = auto()


def get_weight_decay(wd_type: WeightDecayType, kwargs: dict) -> Schedule:
    if wd_type == WeightDecayType.PIECEWISE_CONSTANT:
        weight_decay = piecewise_constant(**kwargs)
    elif wd_type == WeightDecayType.CONSTANT:
        weight_decay = constant(**kwargs)
    elif wd_type == WeightDecayType.POLYNOMIAL_DECAY:
        weight_decay = polynomial_decay(**kwargs)
    elif wd_type == WeightDecayType.TRANSITION_BETWEEN_VALUES:
        weight_decay = transition_between_values(**kwargs)
    return weight_decay


def transition_between_values(
    transition_start, step_size, decay_steps, final_step_size, power=1.0
) -> Schedule:
    initial_value = constant(step_size)
    later_value = polynomial_decay(step_size, decay_steps, final_step_size, power=power)

    def schedule(i):
        return indicator(transition_start - i) * initial_value(i) + indicator(
            i - transition_start
        ) * later_value(i - transition_start)

    return schedule
