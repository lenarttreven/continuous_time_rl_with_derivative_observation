from typing import NamedTuple, List, Any, Tuple

import chex
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax import jit
from jax.lax import cond

from cucrl.main.data_stats import DataStats
from cucrl.main.data_stats import DynamicsData


class DynamicsIdentifier(NamedTuple):
    key: jax.Array
    idx: jax.Array
    eta: jax.Array


class NumberTrainPoints(NamedTuple):
    dynamics: int = 0
    matching: int = 0
    smoother: int = 0


class OfflinePlanningParams(NamedTuple):
    xs_and_us_params: jax.Array
    key: jax.Array


class MPCParameters(NamedTuple):
    dynamics_id: DynamicsIdentifier


class TruePolicy(NamedTuple):
    ts: jax.Array
    us: jax.Array

    @jit
    def __call__(self, t):
        assert t.shape == ()
        ts = self.ts.reshape(-1)
        us = self.us
        # We add -1 here
        return us[jnp.digitize(t, ts, right=False) - 1]


class TrackingData(NamedTuple):
    ts: jax.Array
    xs: jax.Array
    us: jax.Array

    final_t: jax.Array
    target_x: jax.Array
    target_u: jax.Array

    def __call__(self, k: chex.Array):
        assert k.shape == () and k.dtype == jnp.int32
        return cond(k >= self.ts.size, self.outside, self.inside, k)

    def inside(self, k) -> Tuple[chex.Array, chex.Array, chex.Array]:
        return self.xs[k], self.us[k], self.ts[k]

    def outside(self, k) -> Tuple[chex.Array, chex.Array, chex.Array]:
        dt = self.ts[1] - self.ts[0]
        num_over = k - self.ts.size + 1
        t = dt * num_over / (self.final_t - self.ts[-1])
        x_k = (1 - t) * self.xs[-1] + t * self.target_x
        u_k = (1 - t) * self.us[-1] + t * self.target_u
        t_k = self.final_t + num_over * dt
        return x_k, u_k, t_k


# class TrackingData(NamedTuple):
#     ts: jax.Array
#     xs: jax.Array
#     us: jax.Array
#
#     final_t: jax.Array
#     target_x: jax.Array
#     target_u: jax.Array
#
#     # Todo: need to rewrite this to the discrete version
#     def __call__(self, t):
#         assert t.shape == (1,)
#         to_return_x = MultivariateConnectingSpline(self.ts, self.xs, self.final_t.reshape(1), self.target_x)(
#             t.reshape())
#         to_return_u = MultivariateConnectingSpline(self.ts, self.us, self.final_t.reshape(1), self.target_u)(
#             t.reshape())
#         return to_return_x, to_return_u


class PlotOpenLoop(NamedTuple):
    times: jax.Array
    states: jax.Array
    controls: jax.Array
    state_prediction: jax.Array
    controls_prediction: jax.Array
    visualization_times: jax.Array


class PlotData(NamedTuple):
    dynamics_der_means: jax.Array
    dynamics_der_vars: jax.Array
    actual_actions: jax.Array
    visualization_times: jax.Array
    observation_times: List[jax.Array]
    observations: List[jax.Array]
    gt_states_vis: jax.Array
    gt_der_vis: jax.Array
    prediction_states: jax.Array | None
    predicted_actions: jax.Array | None


class SampledData(NamedTuple):
    xs: jax.Array
    xs_dot: jax.Array
    std_xs_dot: jax.Array


class Trajectory(NamedTuple):
    ts: jax.Array
    us: jax.Array
    xs: jax.Array
    xs_dot_true: jax.Array
    xs_dot_noise: jax.Array


class MPCCarry(NamedTuple):
    next_update_time: jax.Array | None = None
    key: jax.Array | None = None
    mpc_params: MPCParameters | None = None
    true_policy: TruePolicy | None = None


class HallucinationSetup(NamedTuple):
    time_horizon: jax.Array
    num_steps: int


class CollectorCarry(NamedTuple):
    next_measurement_time: jax.Array | None = None
    key: jax.Array | None = None
    hallucination_setup: HallucinationSetup | None = None


class IntegrationCarry(NamedTuple):
    collector_carry: CollectorCarry = CollectorCarry()
    mpc_carry: MPCCarry = MPCCarry()


class OCSolution(NamedTuple):
    ts: jax.Array
    xs: jax.Array
    us: jax.Array
    opt_value: jax.Array
    dynamics_id: DynamicsIdentifier


class SmootherApply(NamedTuple):
    xs_mean: jax.Array
    xs_var: jax.Array
    xs_dot_mean: jax.Array
    xs_dot_var: jax.Array
    xs_dot_var_given_x: jax.Array
    loss: jax.Array
    updated_stats: FrozenDict


class SmootherPosterior(NamedTuple):
    xs_mean: jax.Array
    xs_var: jax.Array
    xs_dot_mean: jax.Array
    xs_dot_var: jax.Array


class DynamicsModel(NamedTuple):
    params: Any | None = None
    model_stats: Any | None = None
    data_stats: DataStats | None = None
    episode: int | None = None
    beta: jax.Array | None = None
    history: DynamicsData | None = None
    calibration_alpha: jax.Array | None = None


class OfflinePlanningData(NamedTuple):
    ts: jax.Array
    xs: jax.Array
    us: jax.Array
    x0s: jax.Array
    final_t: jax.Array
    target_x: jax.Array
    target_u: jax.Array
    dynamics_ids: DynamicsIdentifier


class MeasurementSelection(NamedTuple):
    potential_ts: jax.Array
    potential_xs: jax.Array
    potential_us: jax.Array
    vars_before_collection: jax.Array
    proposed_ts: jax.Array
    proposed_indices: jax.Array
