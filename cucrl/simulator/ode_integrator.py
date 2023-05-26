from abc import abstractmethod
from functools import partial
from typing import NamedTuple, Tuple, List

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random, vmap, jit
from jax.lax import scan
from jax.tree_util import tree_map

from cucrl.environment_interactor.interactor import Interactor
from cucrl.main.config import SimulatorConfig
from cucrl.simulator.simulator_costs import get_simulator_costs
from cucrl.simulator.simulator_dynamics import get_simulator_dynamics
from cucrl.utils.classes import IntegrationCarry
from cucrl.utils.classes import Trajectory, MeasurementSelection
from cucrl.utils.splines import MultivariateSpline


class TrajectoryData(NamedTuple):
    xs: jax.Array
    us: jax.Array
    ts: jax.Array
    xs_dot: jax.Array


class IntegrationData(NamedTuple):
    xs: jax.Array
    us: jax.Array
    ts: jax.Array
    xs_dot: jax.Array
    measurement_selection: MeasurementSelection


class _IntegrationData(NamedTuple):
    xs: jnp.ndarray
    us: jnp.ndarray
    xs_dot: jnp.ndarray
    to_take_data: jax.Array
    measurement_selection: MeasurementSelection


class _IntegrationCarry(NamedTuple):
    x: jax.Array
    t: jax.Array
    terminate_condition: jax.Array
    events: IntegrationCarry
    traj_idx: jax.Array


class Integrator:
    def __init__(self, interactor: Interactor, simulator_config: SimulatorConfig):
        self.simulator_config = simulator_config
        self.scaling = simulator_config.scaling
        self.interactor = interactor
        self.limited_budget = simulator_config.termination_config.limited_budget
        self.episode_budget = simulator_config.termination_config.episode_budget_running_cost
        self.simulator_type = simulator_config.simulator_type
        self.simulator_dynamics = get_simulator_dynamics(simulator_config.simulator_type, simulator_config.scaling)
        self.simulator_costs = get_simulator_costs(simulator_config.simulator_type, simulator_config.scaling)
        if simulator_config.termination_config.max_state is None:
            self.max_state: jax.Array = 1e8 * jnp.ones(shape=(self.simulator_dynamics.state_dim,))
        else:
            self.max_state: jax.Array = simulator_config.termination_config.max_state

    @abstractmethod
    def simulate(self, ic, ts, traj_idx, events) -> IntegrationData:
        pass

    @staticmethod
    @jit
    def _evaluate(sim_ts, sim_values, eval_times):
        spline = MultivariateSpline(sim_ts.reshape(-1), sim_values, k=1)
        return spline(eval_times.reshape(-1))

    def evaluate(self, ts: jnp.ndarray, integration_data: IntegrationData) -> TrajectoryData:
        # Take only the part of ts that is smaller than the largest integration_data.ts
        ts = ts[jnp.where(ts <= jnp.max(integration_data.ts))]

        xs = self._evaluate(integration_data.ts, integration_data.xs, ts)
        us = self._evaluate(integration_data.ts, integration_data.us, ts)
        xs_dot = self._evaluate(integration_data.ts, integration_data.xs_dot, ts)
        return TrajectoryData(xs, us, ts, xs_dot)

    def simulate_trajectory(self, ic: jnp.array, time_horizon: Tuple[float, float], num_vis_ts: int, sigma: jnp.ndarray,
                            key, traj_idx, events) -> Tuple[Trajectory, Trajectory, MeasurementSelection]:
        # Here we have to cut sim data
        sim_data = self.simulate(ic, time_horizon, traj_idx, events)
        if self.limited_budget:
            sim_data = self.check_budget(sim_data)

        fixed_nodes_period = events.collector_carry.hallucination_steps_arr.shape[0]
        ts_fixed_nodes = self.ts[::fixed_nodes_period]

        ts_variable_nodes = jnp.concatenate(sim_data.measurement_selection.proposed_ts)

        ts = jnp.concatenate([ts_fixed_nodes[None, ...], ts_variable_nodes])
        ts = jnp.sort(ts, axis=0)

        vis_ts = jnp.linspace(0, jnp.max(ts), num_vis_ts).reshape(-1, 1)

        obs_data = self.evaluate(ts, sim_data)
        vis_data = self.evaluate(vis_ts, sim_data)

        key, subkey = random.split(key)
        noise = sigma * random.normal(key=subkey, shape=obs_data.xs_dot.shape, dtype=jnp.float64)
        xs_dot_noise = obs_data.xs_dot + noise

        obs_traj = Trajectory(ts=obs_data.ts.reshape(-1), us=obs_data.us, xs=obs_data.xs, xs_dot_true=obs_data.xs_dot,
                              xs_dot_noise=xs_dot_noise)
        vis_traj = Trajectory(ts=vis_data.ts.reshape(-1), us=vis_data.us, xs=vis_data.xs, xs_dot_true=vis_data.xs_dot,
                              xs_dot_noise=vis_data.xs_dot)

        return obs_traj, vis_traj, sim_data.measurement_selection

    def check_terminal_condition(self, carry: _IntegrationCarry) -> jax.Array:
        x = carry.x
        max_xs = self.max_state
        assert x.shape == max_xs.shape == (self.simulator_dynamics.state_dim,)
        return jnp.all(jnp.abs(x) - max_xs <= 0)

    def check_budget(self, integration_data: IntegrationData):
        # This function shortens trajectory if we run out of budget, e.g., if the integrated total cost is larger
        # than budget value
        ts = integration_data.ts
        us = integration_data.us
        xs = integration_data.xs

        def total_cost(ts, xs, us):
            integrand = vmap(self.simulator_costs.running_cost)(xs, us)
            integration_cost = jnp.trapz(x=ts.reshape(-1), y=integrand)
            terminal_cost = self.simulator_costs.terminal_cost(xs[-1], us[-1])
            return integration_cost + terminal_cost

        current_ts, current_xs, current_us = jnp.ndarray, jnp.ndarray, jnp.ndarray
        final_index = ts.size

        for i in range(ts.size):
            current_ts = ts[:i + 1]
            current_xs = xs[:i + 1]
            current_us = us[:i + 1]

            if total_cost(current_ts, current_xs, current_us) >= self.episode_budget:
                final_index = i + 1
                break
        return IntegrationData(xs=current_xs, us=current_us, ts=current_ts,
                               xs_dot=integration_data.xs_dot[:final_index])

    def simulate_trajectories(self, ics, time_horizon, num_vis_ts, sigmas, key,
                              events) -> Tuple[List[Trajectory], List[Trajectory], List[MeasurementSelection]]:
        trajectories_obs, trajectories_vis, measurement_selections = [], [], []
        num_trajectories = len(ics)
        key, *subkeys = random.split(key, num_trajectories + 1)
        for i in range(num_trajectories):
            traj_obs, traj_vis, meas_selection = self.simulate_trajectory(ic=ics[i], time_horizon=time_horizon,
                                                                          num_vis_ts=num_vis_ts,
                                                                          sigma=sigmas, key=subkeys[i], traj_idx=i,
                                                                          events=tree_map(lambda x: x[i], events))
            trajectories_obs.append(traj_obs)
            trajectories_vis.append(traj_vis)
            measurement_selections.append(meas_selection)
        return trajectories_obs, trajectories_vis, measurement_selections


@chex.dataclass
class BetweenControlState:
    ts: chex.Array
    xs: chex.Array
    xs_dot: chex.Array
    u: chex.Array


@chex.dataclass
class StateDerivativePair:
    xs: chex.Array
    xs_dot: chex.Array


class ForwardEuler(Integrator):
    def __init__(self, interactor, simulator_config: SimulatorConfig):
        super(ForwardEuler, self).__init__(interactor=interactor, simulator_config=simulator_config)
        T = simulator_config.time_horizon[1] - simulator_config.time_horizon[0]
        total_int_steps = simulator_config.num_nodes * simulator_config.num_int_step_between_nodes
        self.dt = T / total_int_steps
        self.ts = jnp.linspace(*simulator_config.time_horizon, simulator_config.num_nodes + 1)
        self.between_control_ts = jnp.linspace(self.ts[0], self.ts[1], simulator_config.num_int_step_between_nodes + 1)

        total_num_steps = simulator_config.num_int_step_between_nodes * simulator_config.num_nodes + 1
        self.all_ts = jnp.linspace(*simulator_config.time_horizon, total_num_steps)

        inner_ts = self.between_control_ts[:-1]
        outer_ts = self.ts[:-1]
        self.integration_ts = outer_ts[..., None] + inner_ts[None, ...]

    @partial(jit, static_argnums=0)
    def between_control_points(self, x, u, t) -> Tuple[chex.Array, BetweenControlState]:
        assert x.shape == (self.simulator_dynamics.state_dim,) and u.shape == (self.simulator_dynamics.control_dim,)
        assert t.shape == ()
        chex.assert_type(t, int)
        cur_ts = self.ts[t] + self.between_control_ts[:-1]

        def _next_step(x: chex.Array, t: chex.Array) -> Tuple[chex.Array, StateDerivativePair]:
            x_dot = self.simulator_dynamics.dynamics(x, u, t.reshape(-1))
            x_next = x + self.dt * x_dot
            return x_next, StateDerivativePair(xs=x, xs_dot=x_dot)

        x_last, state_der_pairs = jax.lax.scan(_next_step, x, cur_ts)
        return x_last, BetweenControlState(ts=cur_ts, xs=state_der_pairs.xs, xs_dot=state_der_pairs.xs_dot, u=u)

    def integration_step(self, carry: _IntegrationCarry, _):
        u, measurement_selection, new_events = self.interactor.interact(carry.x, carry.t, carry.traj_idx, carry.events)
        x_next, between_control_state = self.between_control_points(carry.x, u, carry.t)
        new_terminate_condition = jnp.array(False)
        new_carry = _IntegrationCarry(x=x_next, t=carry.t + 1, events=new_events,
                                      terminate_condition=new_terminate_condition, traj_idx=carry.traj_idx)
        new_y = _IntegrationData(xs=between_control_state.xs, us=between_control_state.u,
                                 xs_dot=between_control_state.xs_dot, to_take_data=new_terminate_condition,
                                 measurement_selection=measurement_selection)
        return new_carry, new_y

    def fast_simulate(self, ic, time_horizon, traj_idx, events) -> _IntegrationData:
        init = _IntegrationCarry(x=ic, t=jnp.array(0), terminate_condition=jnp.array(False),
                                 traj_idx=traj_idx, events=events)

        new_carry, to_return = scan(self.integration_step, init, None, length=self.simulator_config.num_nodes)
        return to_return

    def simulate(self, ic, time_horizon, traj_idx, events):
        integration_data = self.fast_simulate(ic, time_horizon, traj_idx, events)
        to_take = jnp.invert(integration_data.to_take_data)

        # take non -1 values
        def check_for_negative_one(x):
            return jnp.all(x != -1)

        hallucination_indices = vmap(check_for_negative_one)(integration_data.measurement_selection.proposed_ts)
        measurement_selection = jtu.tree_map(lambda x: x[hallucination_indices], integration_data.measurement_selection)

        # Repeat the us
        us = integration_data.us[to_take]
        us = jnp.repeat(us[:, None, :], repeats=self.simulator_config.num_int_step_between_nodes, axis=1)

        return IntegrationData(xs=jnp.concatenate(integration_data.xs[to_take]),
                               ts=jnp.concatenate(self.integration_ts[to_take]),
                               us=jnp.concatenate(us),
                               xs_dot=jnp.concatenate(integration_data.xs_dot[to_take]),
                               measurement_selection=measurement_selection)
