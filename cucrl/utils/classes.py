from typing import NamedTuple, List, Any, Tuple

import chex
from flax.core import FrozenDict
from jax import jit
from jax.lax import cond

from cucrl.main.data_stats import DataStats
from cucrl.main.data_stats import DynamicsData


class DynamicsIdentifier(NamedTuple):
    key: chex.Array
    idx: chex.Array
    eta: chex.Array


class NumberTrainPoints(NamedTuple):
    dynamics: int = 0
    matching: int = 0
    smoother: int = 0


class OfflinePlanningParams(NamedTuple):
    xs_and_us_params: chex.Array
    key: chex.Array


class TruePolicy(NamedTuple):
    us: chex.Array

    @jit
    def __call__(self, t_k: chex.Array):
        chex.assert_type(t_k, int)
        assert t_k.shape == ()  # and self.ts_idx[0] <= t_idx <= self.ts_idx[-1]
        return self.us[t_k]


class MPCParameters(NamedTuple):
    dynamics_id: DynamicsIdentifier
    true_policy: TruePolicy


class TrackingData(NamedTuple):
    # ts is the time points at which the tracking data (xs, us) is available
    ts: chex.Array
    xs: chex.Array
    us: chex.Array

    final_t: chex.Array
    target_x: chex.Array
    target_u: chex.Array

    def __call__(self, k: chex.Array):
        assert k.shape == ()
        chex.assert_type(k, int)
        return cond(k >= self.ts.size, self.outside, self.inside, k)

    def inside(self, k) -> Tuple[chex.Array, chex.Array, chex.Array]:
        return self.xs[k], self.us[k], self.ts[k]

    def outside(self, k) -> Tuple[chex.Array, chex.Array, chex.Array]:
        dt = self.ts[1] - self.ts[0]
        num_over = k - self.ts.size + 1
        t = dt * num_over / (self.final_t - self.ts[-1])
        x_k = (1 - t) * self.xs[-1] + t * self.target_x
        u_k = (1 - t) * self.us[-1] + t * self.target_u
        t_k = self.ts[-1] + num_over * dt
        return x_k, u_k, t_k


class PlotOpenLoop(NamedTuple):
    times: chex.Array
    states: chex.Array
    controls: chex.Array
    state_prediction: chex.Array
    controls_prediction: chex.Array
    visualization_times: chex.Array


class PlotData(NamedTuple):
    dynamics_der_means: chex.Array
    dynamics_der_vars: chex.Array
    actual_actions: chex.Array
    visualization_times: chex.Array
    observation_times: List[chex.Array]
    observations: List[chex.Array]
    gt_states_vis: chex.Array
    gt_der_vis: chex.Array
    prediction_states: chex.Array | None
    predicted_actions: chex.Array | None


class SampledData(NamedTuple):
    xs: chex.Array
    xs_dot: chex.Array
    std_xs_dot: chex.Array


class Trajectory(NamedTuple):
    ts: chex.Array
    us: chex.Array
    xs: chex.Array
    xs_dot_true: chex.Array
    xs_dot_noise: chex.Array


class MPCCarry(NamedTuple):
    next_update_time: chex.Array | None = None
    key: chex.Array | None = None
    mpc_params: MPCParameters | None = None


class CollectorCarry(NamedTuple):
    next_measurement_time: chex.Array | None = None
    key: chex.Array | None = None
    hallucination_steps_arr: chex.Array | None = None


class IntegrationCarry(NamedTuple):
    collector_carry: CollectorCarry = CollectorCarry()
    mpc_carry: MPCCarry = MPCCarry()


class OCSolution(NamedTuple):
    ts: chex.Array
    xs: chex.Array
    us: chex.Array
    opt_value: chex.Array
    dynamics_id: DynamicsIdentifier


class SmootherApply(NamedTuple):
    xs_mean: chex.Array
    xs_var: chex.Array
    xs_dot_mean: chex.Array
    xs_dot_var: chex.Array
    xs_dot_var_given_x: chex.Array
    loss: chex.Array
    updated_stats: FrozenDict


class SmootherPosterior(NamedTuple):
    xs_mean: chex.Array
    xs_var: chex.Array
    xs_dot_mean: chex.Array
    xs_dot_var: chex.Array


class DynamicsModel(NamedTuple):
    params: Any | None = None
    model_stats: Any | None = None
    data_stats: DataStats | None = None
    episode: int | None = None
    beta: chex.Array | None = None
    history: DynamicsData | None = None
    calibration_alpha: chex.Array | None = None


class OfflinePlanningData(NamedTuple):
    ts: chex.Array
    xs: chex.Array
    us: chex.Array
    x0s: chex.Array
    final_t: chex.Array
    target_x: chex.Array
    target_u: chex.Array
    dynamics_ids: DynamicsIdentifier


class MeasurementSelection(NamedTuple):
    potential_ts: chex.Array
    potential_xs: chex.Array
    potential_us: chex.Array
    vars_before_collection: chex.Array
    proposed_ts: chex.Array
    proposed_indices: chex.Array
