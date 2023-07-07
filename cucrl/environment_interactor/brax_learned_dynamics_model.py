from typing import Tuple

import chex
from brax.envs.base import State, Env
from jax import numpy as jnp
from jax.lax import scan

from cucrl.dynamics_with_control.dynamics_models import AbstractDynamics
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.utils.classes import DynamicsModel
from cucrl.utils.helper_functions import AngleNormalizer


class LearnedModel(Env):

    def __init__(self,
                 integration_dt: float,
                 num_int_steps: int,
                 dynamics_model: AbstractDynamics,
                 dynamics_params: DynamicsModel,
                 sim_cost: SimulatorCostsAndConstraints,
                 angle_normalizer: AngleNormalizer,
                 backend: str = 'string'):
        # The dynamics model that we design needs to have actions bounded to [-1, 1]
        self.integration_dt = integration_dt
        self.num_int_steps = num_int_steps
        self._backend = backend
        self._dynamics_model = dynamics_model
        self._dynamics_params = dynamics_params
        self.angle_normalizer = angle_normalizer
        self.sim_cost = sim_cost

    def _ode(self, x: chex.Array, u: chex.Array) -> chex.Array:
        assert x.shape == (self._dynamics_model.x_dim,)
        assert u.shape == (self._dynamics_model.u_dim + self._dynamics_model.x_dim,)
        pure_u = u[:self._dynamics_model.u_dim]
        eta = u[self._dynamics_model.u_dim:]
        mean, std = self._dynamics_model.mean_and_std_eval_one(self._dynamics_params, x, pure_u)
        # Todo: potentially we need to add scaling of std here
        return mean + std * eta

    def reward(self, x: chex.Array, u: chex.Array) -> chex.Array:
        assert x.shape == (self._dynamics_model.x_dim,)
        assert u.shape == (self._dynamics_model.u_dim + self._dynamics_model.x_dim,)
        pure_u = u[:self._dynamics_model.u_dim]

        def _next_step(_x: chex.Array, _: chex.Array) -> Tuple[chex.Array, chex.Array]:
            x_dot = self._ode(_x, u)
            x_next = _x + self.integration_dt * x_dot
            return x_next, self.integration_dt * self.sim_cost.running_cost(_x, pure_u)

        x_last, cs = scan(_next_step, x, None, length=self.num_int_steps)
        return -jnp.sum(cs)

    def reset(self, rng: jnp.ndarray) -> State:
        initial_state = self.sim_cost.get_x0()
        initial_action = jnp.zeros(shape=(self._dynamics_model.u_dim + self._dynamics_model.x_dim,))
        reward, done = self.reward(initial_state, initial_action), jnp.array(0.0)
        pipeline_state = None
        metrics = {}
        return State(pipeline_state=pipeline_state,
                     obs=initial_state,
                     reward=reward,
                     done=done,
                     metrics=metrics)

    def step(self, state: State, action: jnp.ndarray) -> State:
        assert action.shape == (self._dynamics_model.u_dim + self._dynamics_model.x_dim,)
        x = state.obs
        u = action

        def _next_step(_x: chex.Array, _: chex.Array) -> Tuple[chex.Array, chex.Array]:
            x_dot = self._ode(_x, u)
            x_next = _x + self.integration_dt * x_dot
            return x_next, x_next

        x_last, _ = scan(_next_step, x, None, length=self.num_int_steps)
        x_last = self.angle_normalizer.transform_x(x_last)
        reward, done = self.reward(x, u), jnp.array(0.0)
        return state.replace(obs=x_last, reward=reward, done=done)

    @property
    def observation_size(self) -> int:
        return self._dynamics_model.x_dim

    @property
    def action_size(self) -> int:
        return self._dynamics_model.u_dim + self._dynamics_model.x_dim

    @property
    def backend(self) -> str:
        return self._backend
