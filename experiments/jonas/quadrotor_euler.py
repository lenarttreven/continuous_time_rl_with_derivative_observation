from functools import partial
from typing import Tuple

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit
from jax.lax import cond

from utils import euler_to_rotation, move_frame, quadratic_cost


class QuadrotorEuler:
    """
    Dynamics of quadrotor with 12 dimensional state space and 4 dimensional control
    Code adapted from : https://github.com/Bharath2/Quadrotor-Simulation
    Theory from: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1705&context=edissertations
    Why 4 dimensional control: https://www.youtube.com/watch?v=UC8W3SfKGmg (talks about that at around 8min in video)
    Short description for 4 dimensional control:
          [ F  ]         [ F1 ]
          | M1 |  = A *  | F2 |
          | M2 |         | F3 |
          [ M3 ]         [ F4 ]
    """

    def __init__(self, time_scaling=None, state_scaling=None,
                 control_scaling=None):
        self.x_dim = 12
        self.u_dim = 4
        if time_scaling is None:
            time_scaling = jnp.ones(shape=(1,))
        if state_scaling is None:
            state_scaling = jnp.eye(self.x_dim)
        if control_scaling is None:
            control_scaling = jnp.eye(self.u_dim)
        self.time_scaling = time_scaling
        self.state_scaling = state_scaling
        self.state_scaling_inv = jnp.linalg.inv(state_scaling)
        self.control_scaling = control_scaling
        self.control_scaling_inv = jnp.linalg.inv(control_scaling)

        self.mass = 0.18  # kg
        self.g = 9.81  # m/s^2
        self.arm_length = 0.086  # meter
        self.height = 0.05

        self.I = jnp.array([(0.00025, 0, 2.55e-6),
                            (0, 0.000232, 0),
                            (2.55e-6, 0, 0.0003738)])

        self.invI = jnp.linalg.inv(self.I)

        self.minF = 0.0
        self.maxF = 2.0 * self.mass * self.g

        self.km = 1.5e-9
        self.kf = 6.11e-8
        self.r = self.km / self.kf

        self.L = self.arm_length
        self.H = self.height
        #  [ F  ]         [ F1 ]
        #  | M1 |  = A *  | F2 |
        #  | M2 |         | F3 |
        #  [ M3 ]         [ F4 ]
        self.A = jnp.array([[1, 1, 1, 1],
                            [0, self.L, 0, -self.L],
                            [-self.L, 0, self.L, 0],
                            [self.r, -self.r, self.r, -self.r]])

        self.invA = jnp.linalg.inv(self.A)

        self.body_frame = jnp.array([(self.L, 0, 0, 1),
                                     (0, self.L, 0, 1),
                                     (-self.L, 0, 0, 1),
                                     (0, -self.L, 0, 1),
                                     (0, 0, 0, 1),
                                     (0, 0, self.H, 1)])

        self.B = jnp.array([[0, self.L, 0, -self.L],
                            [-self.L, 0, self.L, 0]])

        self.internal_control_scaling_inv = jnp.diag(jnp.array([1, 2 * 1e-4, 2 * 1e-4, 1e-3], dtype=jnp.float64))

        # Cost parameters:
        self.state_target = jnp.zeros(shape=(12,), dtype=jnp.float64)
        self.state_target = jnp.array([0, 0, 0,
                                       0, 0, 0,
                                       0, 0, 0.0,
                                       0, 0, 0], dtype=jnp.float64)
        # self.action_target = jnp.array([0.1458, 0., 0., 0.], dtype=jnp.float64)
        self.action_target = jnp.array([0., 0., 0., 0.], dtype=jnp.float64)
        self.running_q = 1.0 * jnp.diag(jnp.array([1, 1, 1, 1,
                                                   1, 1, 0.1, 0.1,
                                                   0.1, 1, 0.1, 0.1], dtype=jnp.float64))
        self.running_r = 1e-2 * 1.0 * jnp.diag(jnp.array([5.0, 0.8, 0.8, 0.3], dtype=jnp.float64))
        self.terminal_q = 5.0 * jnp.eye(self.x_dim)
        self.terminal_r = 0.0 * jnp.eye(self.u_dim)

    def _ode(self, state, u):
        # u = self.scaling_u_inv @ u
        # Here we have to decide in which coordinate system we will operate
        # If we operate with F1, F2, F3 and F4 we need to run
        # u = self.A @ u
        u = self.internal_control_scaling_inv @ u
        F, M = u[0], u[1:]
        x, y, z, xdot, ydot, zdot, phi, theta, psi, p, q, r = state
        angles = jnp.array([phi, theta, psi])
        wRb = euler_to_rotation(angles)
        # acceleration - Newton's second law of motion
        accel = 1.0 / self.mass * (wRb.dot(jnp.array([[0, 0, F]]).T)
                                   - jnp.array([[0, 0, self.mass * self.g]]).T)
        # angular acceleration - Euler's equation of motion
        # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        omega = jnp.array([p, q, r])
        angles_dot = jnp.linalg.inv(move_frame(angles)) @ omega
        pqrdot = self.invI.dot(M.flatten() - jnp.cross(omega, self.I.dot(omega)))
        state_dot_0 = xdot
        state_dot_1 = ydot
        state_dot_2 = zdot
        state_dot_3 = accel[0].reshape()
        state_dot_4 = accel[1].reshape()
        state_dot_5 = accel[2].reshape()
        state_dot_6 = angles_dot[0]
        state_dot_7 = angles_dot[1]
        state_dot_8 = angles_dot[2]
        state_dot_9 = pqrdot[0]
        state_dot_10 = pqrdot[1]
        state_dot_11 = pqrdot[2]
        return jnp.array([state_dot_0, state_dot_1, state_dot_2, state_dot_3, state_dot_4,
                          state_dot_5, state_dot_6, state_dot_7, state_dot_8, state_dot_9,
                          state_dot_10, state_dot_11])

    @partial(jit, static_argnums=0)
    def ode(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        return self.state_scaling @ self._ode(x, u) / self.time_scaling.reshape()

    @partial(jit, static_argnums=0)
    def running_cost(self, x, u) -> jnp.ndarray:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        return self._running_cost(x, u) / self.time_scaling.reshape()

    @partial(jit, static_argnums=0)
    def terminal_cost(self, x, u) -> jnp.ndarray:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        return self._terminal_cost(x, u)

    def _running_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)


class OptimalCost:
    def __init__(self, time_horizon: Tuple[float, float], num_nodes: int = 50):
        self.dynamics = QuadrotorEuler()
        self.num_nodes = num_nodes
        self.time_horizon = time_horizon

        self.dt = (self.time_horizon[1] - self.time_horizon[0]) / num_nodes
        self.ts = jnp.linspace(self.time_horizon[0], self.time_horizon[1], num_nodes + 1)
        self.ilqr_params = ILQRHyperparams(maxiter=1000, make_psd=False, psd_delta=1e0)
        self.ilqr = ILQR(self.cost, self.next_step)
        self.results = None

    def running_cost(self, x, u, t):
        return self.dt * self.dynamics.running_cost(x, u)

    def terminal_cost(self, x, u, t):
        return self.dynamics.terminal_cost(x, u)

    def cost(self, x, u, t, params=None):
        return cond(t == self.num_nodes, self.terminal_cost, self.running_cost, x, u, t.reshape(1, ))

    def next_step(self, x, u, t, params=None):
        assert x.shape == (self.dynamics.x_dim,) and u.shape == (self.dynamics.u_dim,)
        return x + self.dt * self.dynamics.ode(x, u)

    def solve(self, initial_conditions: chex.Array):
        initial_actions = 0.01 * jnp.ones(shape=(self.num_nodes, self.dynamics.u_dim,))
        out = self.ilqr.solve(None, None, initial_conditions, initial_actions, self.ilqr_params)
        self.results = out
        return out


if __name__ == '__main__':
    from jax.config import config
    from trajax.optimizers import ILQR, ILQRHyperparams

    config.update('jax_enable_x64', True)
    sim = QuadrotorEuler()

    time_horizon = (0, 3)
    state_scaling = jnp.diag(jnp.array([1, 1, 1, 1, 1, 1, 10, 10, 1, 10, 10, 1], dtype=jnp.float64))

    x0 = jnp.array([1.0, 1.0, 1.0,
                    0., 0., 0.,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0], dtype=jnp.float64)
    num_nodes = 100
    oc = OptimalCost(time_horizon, num_nodes)
    out = oc.solve(x0)
    plt.title('xs')
    plt.plot(out.xs)
    plt.show()

    plt.title('us')
    plt.plot(out.us)
    plt.show()
