from abc import abstractmethod
from typing import NamedTuple, Tuple, List

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random, vmap, jit
from jax.lax import cond, bitwise_and, bitwise_not, scan
from jax.tree_util import tree_map

from cucrl.environment_interactor.interactor import Interactor
from cucrl.main.config import Scaling, TerminationConfig
from cucrl.simulator.simulator_costs import get_simulator_costs
from cucrl.simulator.simulator_dynamics import get_simulator_dynamics
from cucrl.utils.classes import IntegrationCarry
from cucrl.utils.classes import Trajectory, MeasurementSelection
from cucrl.utils.representatives import SimulatorType
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
    ts: jnp.ndarray
    xs_dot: jnp.ndarray
    to_take_data: jax.Array
    measurement_selection: MeasurementSelection


class IntegrationDataOne(NamedTuple):
    x: jnp.ndarray
    u: jnp.ndarray
    t: jnp.ndarray
    x_dot: jnp.ndarray


class _IntegrationCarry(NamedTuple):
    x: jax.Array
    t: jax.Array
    terminate_condition: jax.Array
    events: IntegrationCarry
    traj_idx: jax.Array


class Integrator:
    def __init__(self, interactor: Interactor, simulator_type: SimulatorType, scaling: Scaling,
                 termination_config: TerminationConfig):
        self.scaling = scaling
        self.interactor = interactor
        self.limited_budget = termination_config.limited_budget
        self.episode_budget = termination_config.episode_budget_running_cost
        self.simulator_type = simulator_type
        self.simulator_dynamics = get_simulator_dynamics(simulator_type, scaling)
        self.simulator_costs = get_simulator_costs(simulator_type, scaling)
        if termination_config.max_state is None:
            self.max_state: jax.Array = 1e8 * jnp.ones(shape=(self.simulator_dynamics.state_dim,))
        else:
            self.max_state: jax.Array = termination_config.max_state

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

        # Todo: add here the last point as well
        ts_fixed_nodes = jnp.arange(*time_horizon, events.collector_carry.hallucination_setup.time_horizon)
        ts_fixed_nodes = jnp.append(ts_fixed_nodes, time_horizon[1]).reshape(-1, 1)
        ts_variable_nodes = jnp.concatenate(sim_data.measurement_selection.proposed_ts)

        ts = jnp.concatenate([ts_fixed_nodes, ts_variable_nodes])
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


class ForwardEuler(Integrator):
    def __init__(self, interactor, simulator_type, scaling, termination_config, step_size=0.01):
        super(ForwardEuler, self).__init__(interactor=interactor, simulator_type=simulator_type, scaling=scaling,
                                           termination_config=termination_config)
        self.step_size = step_size

    def f(self, carry: _IntegrationCarry, _):
        def true_fun(carry: _IntegrationCarry):
            def print_f(carry):
                jax.debug.print("Current time in simulation: {x}", x=carry.t)

            def skip_print(carry):
                pass

            cond(jnp.allclose(jnp.mod(carry.t, 1), jnp.zeros_like(carry.t), atol=self.step_size), print_f, skip_print,
                 carry)

            u, measurement_selection, new_events = self.interactor.interact(carry.x, carry.t, carry.traj_idx,
                                                                            carry.events)
            x_dot = self.simulator_dynamics.dynamics(carry.x, u, carry.t)
            new_terminate_condition = jnp.array(False)
            new_carry = _IntegrationCarry(x=carry.x + self.step_size * x_dot, t=carry.t + self.step_size,
                                          events=new_events, terminate_condition=new_terminate_condition,
                                          traj_idx=carry.traj_idx)
            new_y = _IntegrationData(carry.x, u, carry.t, x_dot, new_terminate_condition, measurement_selection)
            return new_carry, new_y

        def false_fun(carry: _IntegrationCarry):
            new_carry = _IntegrationCarry(x=carry.x, t=carry.t + self.step_size, events=carry.events,
                                          terminate_condition=jnp.array(True), traj_idx=carry.traj_idx)
            new_y = _IntegrationData(jnp.zeros(shape=(self.simulator_dynamics.state_dim,)),
                                     jnp.zeros(shape=(self.simulator_dynamics.control_dim,)),
                                     carry.t, jnp.zeros(shape=(self.simulator_dynamics.state_dim,)),
                                     jnp.array(True),
                                     self.interactor.measurements_collector.default_measurement_selection())
            return new_carry, new_y

        return cond(bitwise_and(self.check_terminal_condition(carry), bitwise_not(carry.terminate_condition)), true_fun,
                    false_fun, carry)

    def fast_simulate(self, ic, time_horizon, traj_idx, events) -> _IntegrationData:
        start_t, end_t = time_horizon
        xs = jnp.arange(start_t, end_t, self.step_size)
        init = _IntegrationCarry(x=ic, t=jnp.array(start_t).reshape(1, ), terminate_condition=jnp.array(False),
                                 traj_idx=traj_idx, events=events)
        new_carry, to_return = scan(self.f, init, xs)
        return to_return

    def simulate(self, ic, time_horizon, traj_idx, events):
        integration_data = self.fast_simulate(ic, time_horizon, traj_idx, events)
        to_take = jnp.invert(integration_data.to_take_data)

        # take non -1 values
        def check_for_negative_one(x):
            return jnp.all(x != -1)

        hallucination_indices = vmap(check_for_negative_one)(integration_data.measurement_selection.proposed_ts)
        measurement_selection = jtu.tree_map(lambda x: x[hallucination_indices], integration_data.measurement_selection)

        # ts_variable_nodes = ts_variable_nodes[jnp.array(ts_variable_nodes != -1)]
        return IntegrationData(xs=integration_data.xs[to_take], ts=integration_data.ts[to_take],
                               us=integration_data.us[to_take], xs_dot=integration_data.xs_dot[to_take],
                               measurement_selection=measurement_selection)
