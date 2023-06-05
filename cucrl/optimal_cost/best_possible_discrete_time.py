from functools import partial
from typing import Tuple, NamedTuple

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan, cond
from trajax.optimizers import ILQR, ILQRHyperparams, CEMHyperparams, CEM

from cucrl.simulator.simulator_costs import SimulatorCostsAndConstraints
from cucrl.simulator.simulator_dynamics import SimulatorDynamics


class _IntegrateCarry(NamedTuple):
    x: jnp.ndarray
    t: jnp.ndarray


class BestPossibleDiscreteAlgorithm:
    def __init__(self, simulator_dynamics: SimulatorDynamics, simulator_costs: SimulatorCostsAndConstraints,
                 time_horizon: Tuple[float, float], num_nodes: int):
        self.simulator_dynamics = simulator_dynamics
        self.simulator_cost = simulator_costs
        self.time_horizon = time_horizon
        self.num_nodes = num_nodes

        self.num_action_nodes = self.num_nodes - 1
        self.dt = (self.time_horizon[1] - self.time_horizon[0]) / self.num_action_nodes

        self.initial_actions = jnp.zeros(shape=(self.num_action_nodes, self.simulator_dynamics.control_dim))

        self.initial_actions = -0.01 * jnp.ones(shape=(self.num_action_nodes, self.simulator_dynamics.control_dim))

        self.num_low_steps = 50
        self.time_span = self.dt

        self.ilqr = ILQR(self.cost_fn, self.dynamics_fn)
        self.ilqr_params = ILQRHyperparams(maxiter=1000, psd_delta=1e0, make_psd=False)
        # CEM
        self.cem_params = CEMHyperparams(max_iter=10, num_samples=400, elite_portion=0.1, evolution_smoothing=0.0,
                                         sampling_smoothing=0.0)
        self.max_control = 10 * jnp.ones(shape=(self.simulator_dynamics.control_dim,), dtype=jnp.float64)
        self.min_control = -10 * jnp.ones(shape=(self.simulator_dynamics.control_dim,), dtype=jnp.float64)
        self.cem = CEM(self.cost_fn, self.dynamics_fn)

        self.eval_dt = self.dt / self.num_low_steps

    @partial(jit, static_argnums=(0))
    def integrate(self, x: jax.Array, u: jax.Array, t: jax.Array) -> jax.Array:
        _dt = self.time_span / self.num_low_steps

        xt = _IntegrateCarry(x, t * self.dt)

        def f(_xt: _IntegrateCarry, _):
            x_dot = self.simulator_dynamics.dynamics(_xt.x, u, _xt.t.reshape(1, ))
            x_next = _xt.x + x_dot * _dt
            t_next = _xt.t + _dt
            return _IntegrateCarry(x_next, t_next), None

        xt_next, _ = scan(f, xt, None, length=self.num_low_steps)
        return xt_next.x

    def cost_fn(self, x, u, t, params=None):
        assert x.shape == (self.simulator_dynamics.state_dim,) and u.shape == (self.simulator_dynamics.control_dim,)

        def running_cost(x, u, t):
            return self.dt * self.simulator_cost.running_cost(x, u)

        def terminal_cost(x, u, t):
            return self.simulator_cost.terminal_cost(x, u)

        return cond(t == self.num_action_nodes, terminal_cost, running_cost, x, u, t)

    def dynamics_fn(self, x, u, t, params=None):
        assert x.shape == (self.simulator_dynamics.state_dim,) and u.shape == (self.simulator_dynamics.control_dim,)
        return self.integrate(x, u, t)

    def get_optimal_cost(self, initial_state):
        out = self.ilqr.solve(None, None, initial_state, self.initial_actions, self.ilqr_params)
        # cem_out = self.cem.solve(None, None, initial_state, self.initial_actions, control_low=self.min_control,
        #                          control_high=self.max_control, hyperparams=self.cem_params,
        #                          random_key=jax.random.PRNGKey(0))
        # xs, us, obj =cem_out
        # x_last, xs_all = self.rollout_eval(us, initial_state)
        #
        # us_all = jnp.repeat(us[:, None, :], repeats=self.num_low_steps, axis=1)
        #
        # xs_all = xs_all.reshape(-1, self.simulator_dynamics.state_dim)
        # us_all = us_all.reshape(-1, self.simulator_dynamics.control_dim)
        # xs_all = jnp.concatenate([initial_state[None, :], xs_all])
        #
        # self.best_xs_all = xs_all
        # self.best_us_all = us_all
        # self.ts_all = jnp.linspace(self.time_horizon[0], self.time_horizon[1], xs_all.shape[0])
        #
        # true_cost = self.cost_fn_eval(xs_all, us_all)
        x_last, xs_all = self.rollout_eval(out.us, initial_state)

        us_all = jnp.repeat(out.us[:, None, :], repeats=self.num_low_steps, axis=1)

        xs_all = xs_all.reshape(-1, self.simulator_dynamics.state_dim)
        us_all = us_all.reshape(-1, self.simulator_dynamics.control_dim)
        xs_all = jnp.concatenate([initial_state[None, :], xs_all])

        self.best_xs_all = xs_all
        self.best_us_all = us_all
        self.ts_all = jnp.linspace(self.time_horizon[0], self.time_horizon[1], xs_all.shape[0])

        true_cost = self.cost_fn_eval(xs_all, us_all)
        return true_cost

    # Evaluation functions
    @partial(jit, static_argnums=(0,))
    def integrate_eval(self, x: jax.Array, u: jax.Array, t: jax.Array):
        _dt = self.time_span / self.num_low_steps

        xt = _IntegrateCarry(x, t * self.dt)

        def f(_xt: _IntegrateCarry, _):
            x_dot = self.simulator_dynamics.dynamics(_xt.x, u, _xt.t.reshape(1, ))
            x_next = _xt.x + x_dot * _dt
            t_next = _xt.t + _dt
            return _IntegrateCarry(x_next, t_next), x_next

        xt_next, xs_next = scan(f, xt, None, length=self.num_low_steps)
        return xt_next.x, xs_next

    def dynamics_fn_eval(self, x, u, t):
        assert x.shape == (self.simulator_dynamics.state_dim,) and u.shape == (self.simulator_dynamics.control_dim,)
        return self.integrate_eval(x, u, t)

    def rollout_eval(self, U, x0):
        ts = jnp.arange(self.num_action_nodes)

        inputs = (U, ts)

        def dynamics_for_scan(x, _inputs):
            u, t = _inputs
            x_next, xs_next = self.dynamics_fn_eval(x, u, t)
            return x_next, xs_next

        return scan(dynamics_for_scan, x0, inputs)

    def running_cost_eval(self, x, u):
        return self.eval_dt * self.simulator_cost.running_cost(x, u)

    def terminal_cost_eval(self, x, u):
        return self.simulator_cost.terminal_cost(x, u)

    def cost_fn_eval(self, xs, us):
        _running_cost = jnp.sum(vmap(self.running_cost_eval)(xs[:-1], us))
        _terminal_cost = self.terminal_cost_eval(xs[-1], us[-1])
        return _running_cost + _terminal_cost


if __name__ == '__main__':
    from jax.config import config
    import matplotlib.pyplot as plt

    config.update('jax_enable_x64', True)

    # # Pendulum
    # state_scaling = jnp.diag(jnp.array([1.0, 2.0]))
    # simulator_dynamics = PendulumDynamics(state_scaling=state_scaling)
    # simulator_costs = PendulumCosts(state_scaling=state_scaling)
    # best_possible_algorithm = BestPossibleDiscreteAlgorithm(simulator_dynamics, simulator_costs, time_horizon=(0, 10),
    #                                                         num_nodes=11)
    # print(best_possible_algorithm.get_optimal_cost(jnp.array([jnp.pi / 2, 0.0])))
    #
    # plt.plot(best_possible_algorithm.ts_all, best_possible_algorithm.best_xs_all)
    # plt.show()
    # plt.plot(best_possible_algorithm.ts_all[:-1], best_possible_algorithm.best_us_all)
    # plt.show()

    # Race Car
    # state_scaling = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 10.0, 1.0]))
    # simulator_dynamics = RaceCarDynamics(state_scaling=state_scaling)
    # simulator_costs = RaceCarCost(state_scaling=state_scaling)
    # best_possible_algorithm = BestPossibleDiscreteAlgorithm(simulator_dynamics, simulator_costs, time_horizon=(0, 10),
    #                                                         num_nodes=50)
    # print("Best discrete cost: ",
    #       best_possible_algorithm.get_optimal_cost(jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64)))
    #
    # plt.plot(best_possible_algorithm.ts_all, best_possible_algorithm.best_xs_all)
    # plt.show()
    # plt.plot(best_possible_algorithm.ts_all[:-1], jnp.tanh(best_possible_algorithm.best_us_all))
    # plt.show()

    # Quadrotor
    from cucrl.simulator.simulator_costs import QuadrotorEuler as QuadrotorEulerCost
    from cucrl.simulator.simulator_dynamics import QuadrotorEuler as QuadrotorEulerDynamics

    state_scaling = jnp.diag(jnp.array([1, 1, 1, 1, 1, 1, 10, 10, 1, 10, 10, 1], dtype=jnp.float64))
    simulator_dynamics = QuadrotorEulerDynamics(state_scaling=state_scaling)
    simulator_costs = QuadrotorEulerCost(state_scaling=state_scaling)
    best_possible_algorithm = BestPossibleDiscreteAlgorithm(simulator_dynamics, simulator_costs, time_horizon=(0, 15),
                                                            num_nodes=20)
    print('Best discrete cost: ',
          best_possible_algorithm.get_optimal_cost(jnp.array([1.0, 1.0, 1.0,
                                                              0., 0., 0.,
                                                              0.0, 0.0, 0.0,
                                                              0.0, 0.0, 0.0], dtype=jnp.float64)))

    plt.plot(best_possible_algorithm.ts_all, best_possible_algorithm.best_xs_all)
    plt.title('Xs')
    plt.show()
    plt.plot(best_possible_algorithm.ts_all[:-1], jnp.tanh(best_possible_algorithm.best_us_all))
    plt.title('Us')
    plt.show()
