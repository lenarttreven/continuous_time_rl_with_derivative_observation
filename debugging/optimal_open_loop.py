from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax.lax import cond
from trajax.optimizers import ILQRHyperparams, CEMHyperparams, ILQR_with_CEM_warmstart, ILQR

from cucrl.main.config import PolicyConfig, TimeHorizon, OnlineTrackingConfig
from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.simulator.simulator_dynamics import SimulatorDynamics
from cucrl.utils.classes import OCSolution, DynamicsModel, DynamicsIdentifier
from cucrl.utils.representatives import MinimizationMethod


class OptimalOpenLoop:
    def __init__(self, x_dim: int, u_dim: int, time_horizon: TimeHorizon, dynamics: SimulatorDynamics,
                 simulator_costs: SimulatorCostsAndConstraints, policy_config: PolicyConfig = PolicyConfig()):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.time_horizon = time_horizon
        self.simulator_costs = simulator_costs
        self.dynamics = dynamics

        # Number of parameters for eta + number of parameters for control
        self.policy_config = policy_config
        # Setup nodes
        self.num_control_nodes = policy_config.num_control_steps
        self.num_nodes = self.num_control_nodes + 1
        # Setup time
        total_time = time_horizon.length()
        total_int_steps = policy_config.num_control_steps * policy_config.num_int_step_between_nodes
        self.dt = total_time / total_int_steps

        self.ts = jnp.linspace(self.time_horizon.t_min, self.time_horizon.t_max, self.num_nodes)
        self.between_control_ts = jnp.linspace(self.ts[0], self.ts[1], policy_config.num_int_step_between_nodes)
        self.num_total_params = self.u_dim * self.num_control_nodes

        # Setup optimizer, i.e., either ILQR or ILQR with CEM
        self.minimize_method = policy_config.offline_planning.minimization_method
        if self.minimize_method == MinimizationMethod.ILQR_WITH_CEM:
            self.cem_params = CEMHyperparams(max_iter=10, sampling_smoothing=0.0, num_samples=200,
                                             evolution_smoothing=0.0,
                                             elite_portion=0.1)
            self.ilqr_params = ILQRHyperparams(maxiter=100)
            self.control_low = -10.0 * jnp.ones(shape=(self.u_dim,))
            self.control_high = 10.0 * jnp.ones(shape=(self.u_dim,))
            self.optimizer = ILQR_with_CEM_warmstart(self.cost_fn, self.dynamics_fn)

        elif self.minimize_method == MinimizationMethod.ILQR:
            self.ilqr_params = ILQRHyperparams(maxiter=100)
            self.optimizer = ILQR(self.cost_fn, self.dynamics_fn)

    def example_dynamics_id(self) -> DynamicsIdentifier:
        return DynamicsIdentifier(eta=jnp.ones(shape=(self.num_nodes, self.x_dim)),
                                  idx=jnp.ones(shape=(), dtype=jnp.int32),
                                  key=jnp.ones(shape=(2,), dtype=jnp.int32))

    def example_oc_solution(self) -> OCSolution:
        return OCSolution(ts=self.ts, xs=jnp.ones(shape=(self.num_nodes, self.x_dim)),
                          us=jnp.ones(shape=(self.num_nodes, self.u_dim)), opt_value=jnp.ones(shape=()),
                          dynamics_id=self.example_dynamics_id())

    def ode(self, x: chex.Array, u: chex.Array):
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        return self.dynamics.dynamics(x, u, jnp.zeros(shape=(1,)))

    def dynamics_fn(self, x, u, t, params):
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,) and t.shape == ()
        chex.assert_type(t, int)
        cur_ts = self.ts[t] + self.between_control_ts

        def _next_step(_x: chex.Array, _t: chex.Array) -> Tuple[chex.Array, chex.Array]:
            x_dot = self.ode(_x, u)
            x_next = _x + self.dt * x_dot
            return x_next, x_next

        x_last, _ = jax.lax.scan(_next_step, x, cur_ts)
        return x_last

    def cost_fn(self, x, u, t, params):
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)

        def running_cost(_x, _u, _t):
            cur_ts = self.ts[_t] + self.between_control_ts

            def _next_step(_x_: chex.Array, _t_: chex.Array) -> Tuple[chex.Array, chex.Array]:
                x_dot = self.ode(_x_, _u)
                x_next = _x_ + self.dt * x_dot
                return x_next, self.dt * self.simulator_costs.running_cost(_x_, _u)

            x_last, cs = jax.lax.scan(_next_step, _x, cur_ts)
            assert cs.shape == (self.policy_config.num_int_step_between_nodes,)
            return jnp.sum(cs)

        def terminal_cost(_x, _u, _t):
            return self.simulator_costs.terminal_cost(_x, _u[:self.u_dim])

        return cond(t == self.num_control_nodes, terminal_cost, running_cost, x, u, t)

    def plan_offline(self, dynamics_model: DynamicsModel, key: chex.PRNGKey, x0: chex.Array) -> OCSolution:
        initial_actions = jnp.zeros(shape=(self.num_control_nodes, self.u_dim))
        match self.minimize_method:
            case MinimizationMethod.ILQR_WITH_CEM:
                results = self.optimizer.solve(dynamics_model, dynamics_model, x0, initial_actions,
                                               control_low=self.control_low, control_high=self.control_high,
                                               ilqr_hyperparams=self.ilqr_params, cem_hyperparams=self.cem_params,
                                               random_key=key)
            case MinimizationMethod.ILQR:
                results = self.optimizer.solve(dynamics_model, dynamics_model, x0, initial_actions, self.ilqr_params)
            case _:
                raise NotImplementedError(f'Minimization method {self.minimize_method} not implemented')
        us = jnp.concatenate([results.us, results.us[-1:]], axis=0)
        return OCSolution(ts=self.ts, xs=results.xs, us=us, opt_value=results.obj,
                          dynamics_id=self.example_dynamics_id())


if __name__ == '__main__':
    from cucrl.simulator.simulator_dynamics import Pendulum as PendulumDynamics
    from cucrl.simulator.simulator_costs import Pendulum as PendulumCosts
    from debugging.optimal_tracking import OnlineTracking
    from jax.config import config
    import matplotlib.pyplot as plt
    from cucrl.utils.classes import TrackingData

    config.update('jax_enable_x64', True)

    dynamics = PendulumDynamics()
    costs = PendulumCosts()
    policy_config = PolicyConfig(num_control_steps=30,
                                 online_tracking=OnlineTrackingConfig(control_steps=10))

    planner = OptimalOpenLoop(x_dim=2, u_dim=1, time_horizon=TimeHorizon(t_min=0.0, t_max=10.0),
                              dynamics=dynamics, simulator_costs=costs, policy_config=policy_config)

    tracker = OnlineTracking(x_dim=2, u_dim=1, simulator_costs=costs, dynamics=dynamics,
                             policy_config=policy_config, time_horizon=TimeHorizon(t_min=0.0, t_max=10.0))

    plan_out = planner.plan_offline(None, jax.random.PRNGKey(0), jnp.array([jnp.pi, 0.0]))
    plt.plot(plan_out.ts, plan_out.xs, label='xs')
    plt.step(plan_out.ts, plan_out.us, where='post', label='us')
    plt.legend()
    plt.show()

    tracking_data = TrackingData(xs=plan_out.xs, us=plan_out.us, ts=plan_out.ts,
                                 target_x=jnp.array([0.0, 0.0]),
                                 target_u=jnp.array([0.0]),
                                 mpc_control_steps=policy_config.online_tracking.control_steps)

    indices = jnp.arange(plan_out.ts.size)
    x0s = plan_out.xs


    def track(x0, t_start_idx):
        return tracker.track_online(initial_conditions=x0,
                                    t_start_idx=t_start_idx,
                                    tracking_data=tracking_data,
                                    key=jax.random.PRNGKey(0))


    track_values = jax.vmap(track)(x0s, indices)
    plt.plot(track_values.opt_value, label='Errors')
    plt.yscale('log')
    plt.legend()
    plt.show()
    print(track_values.opt_value)
