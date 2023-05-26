from typing import Callable

import chex
import jax.numpy as jnp
from cyipopt import minimize_ipopt
from jax import jit, grad

from cucrl.main.config import InteractionConfig
from cucrl.utils.representatives import TimeHorizonType

Schedule = Callable[[int], chex.Array]


class TimeSampler:
    def __init__(self, interaction_config: InteractionConfig):
        self.system_assumptions = interaction_config.system_assumptions
        self.time_horizon_config = interaction_config.measurement_collector.time_horizon
        self.interaction_config = interaction_config

    @staticmethod
    @jit
    def _horizon_condition(x, beta, l_f, l_pi, l_sigma, target):
        # This is the function that we want to minimize, value of x at minimum is the hallucination horizon
        value = 2 * beta * l_sigma * jnp.sqrt(1 + l_pi) * jnp.exp(l_f * jnp.sqrt(1 + l_pi) * x) * x
        return jnp.sum((value - target) ** 2)

    def init_time_horizon(self):
        return self.time_horizon_config.init_horizon

    def time_horizon(self, beta: chex.Array):
        if self.time_horizon_config.type == TimeHorizonType.ADAPTIVE_TRUE:
            max_beta = jnp.max(beta)
            bnds = [(0, 1)]
            out = minimize_ipopt(self._horizon_condition, x0=jnp.array(0.6), jac=jit(grad(self._horizon_condition)),
                                 options={'max_iter': 1000, 'disp': 5}, bounds=bnds,
                                 args=(max_beta, self.system_assumptions.l_f, self.system_assumptions.l_pi,
                                       self.system_assumptions.l_sigma, self.system_assumptions.hallucination_error))
            horizon = out.x
            return horizon
        elif self.time_horizon_config.type == TimeHorizonType.FIXED:
            return self.time_horizon_config.init_horizon

    def time_steps(self, beta: chex.Array) -> chex.Array:
        time_horizon = self.time_horizon(beta)
        ts_nodes = jnp.linspace(*self.interaction_config.time_horizon, self.interaction_config.policy.num_nodes + 1)
        num_nodes = jnp.sum(ts_nodes <= time_horizon)
        return num_nodes
