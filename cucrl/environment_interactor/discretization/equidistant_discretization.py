from functools import partial
from typing import Tuple

import chex
import jax.numpy as jnp
from flax import struct
from jax import jit


@struct.dataclass
class TimeHorizon:
    t_min: float
    t_max: float


class EquidistantDiscretization:
    def __init__(self, time_horizon: TimeHorizon, num_control_steps: int, buffer_control_steps: int = 1):
        """

        :param time_horizon: time horizon of the simulation
        :param num_control_steps: number of time we apply control
        :param buffer_control_steps: additional steps we can use to simulate after the last control step
        """
        self.time_horizon = time_horizon
        self.num_control_steps = num_control_steps
        self.buffer_steps = buffer_control_steps
        self.discrete_times, self.continuous_times, self.dt = self._get_time_horizon(time_horizon, num_control_steps,
                                                                                     buffer_control_steps)

    @staticmethod
    def _get_time_horizon(time_horizon: TimeHorizon, num_control_steps,
                          buffer_steps: int) -> Tuple[chex.Array, ...]:
        t_min = time_horizon.t_min
        t_max = time_horizon.t_max
        dt = (t_max - t_min) / num_control_steps
        discrete_times = jnp.arange(num_control_steps + buffer_steps)
        true_times = t_min + discrete_times * dt

        assert jnp.sum(true_times <= t_max) == num_control_steps + 1
        assert discrete_times.shape == (num_control_steps + buffer_steps,)
        return discrete_times, true_times, dt

    @partial(jit, static_argnums=(0,))
    def discrete_to_continuous(self, k: chex.Array) -> chex.Array:
        chex.assert_type(k, int)
        assert k.shape == ()
        return self.continuous_times[k]

    @partial(jit, static_argnums=(0,))
    def continuous_to_discrete(self, t: chex.Array) -> chex.Array:
        chex.assert_type(t, float)
        assert t.shape == ()
        return jnp.rint((t - self.time_horizon.t_min) / self.dt).astype(int)


if __name__ == '__main__':
    model = EquidistantDiscretization(time_horizon=TimeHorizon(t_min=0.0, t_max=10.0), num_control_steps=7,
                                      buffer_control_steps=3)
    print(model.discrete_times)
    print(model.continuous_times)
    print(model.continuous_to_discrete(jnp.array(1.428)))
