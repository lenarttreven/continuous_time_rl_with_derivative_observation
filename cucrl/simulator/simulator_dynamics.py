from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import jit

from cucrl.main.config import Scaling
from cucrl.simulator.prepare_matrix import create_matrix
from cucrl.utils.euler_angles import euler_to_rotation, move_frame
from cucrl.utils.helper_functions import AngleNormalizer
from cucrl.utils.quaternions import Quaternion
from cucrl.utils.representatives import SimulatorType

pytree = Any


class SimulatorDynamics(ABC):
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        system_params: pytree,
        time_scaling: jnp.ndarray | None = None,
        state_scaling: jnp.ndarray | None = None,
        control_scaling: jnp.ndarray | None = None,
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.system_params = system_params
        if time_scaling is None:
            time_scaling = jnp.ones(shape=(1,))
        if state_scaling is None:
            state_scaling = jnp.eye(self.state_dim)
        if control_scaling is None:
            control_scaling = jnp.eye(self.control_dim)
        self.time_scaling = time_scaling
        self.state_scaling = state_scaling
        self.state_scaling_inv = jnp.linalg.inv(state_scaling)
        self.control_scaling = control_scaling
        self.control_scaling_inv = jnp.linalg.inv(control_scaling)

    @partial(jit, static_argnums=0)
    def dynamics(self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        assert (
            x.shape == (self.state_dim,)
            and u.shape == (self.control_dim,)
            and t.shape == (1,)
        )
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        t = t / self.time_scaling
        return (
            self.state_scaling @ self._dynamics(x, u, t) / self.time_scaling.reshape()
        )

    @abstractmethod
    def _dynamics(self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        pass


class Bicycle(SimulatorDynamics):
    """
    Dynamics of Bicycle from
    https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html
    u represents ($tan(delta)$, a)
    """

    def __init__(
        self,
        system_params=jnp.array([1.0]),
        time_scaling=None,
        state_scaling=None,
        control_scaling=None,
    ):
        super().__init__(
            state_dim=4,
            control_dim=2,
            system_params=system_params,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )
        self.length = self.system_params[0]

    def _dynamics(self, x, u, t):
        x0_dot = x[3] * jnp.cos(x[2])
        x1_dot = x[3] * jnp.sin(x[2])
        x2_dot = x[3] * u[0] / self.length
        x3_dot = u[1]
        return jnp.array([x0_dot, x1_dot, x2_dot, x3_dot], dtype=jnp.float64)


class Pendulum(SimulatorDynamics):
    """
    Dynamics of pendulum
    """

    def __init__(
        self,
        system_params=jnp.array([5.0, 9.81]),
        time_scaling=None,
        state_scaling=None,
        control_scaling=None,
    ):
        super().__init__(
            state_dim=2,
            control_dim=1,
            system_params=system_params,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )

    def _dynamics(self, x, u, t):
        return jnp.array(
            [
                x[1],
                self.system_params[1] / self.system_params[0] * jnp.sin(x[0])
                + u.reshape(),
            ]
        )


class Quadrotor2D(SimulatorDynamics):
    """
    Dynamics of pendulum
    """

    def __init__(
        self,
        system_params=None,
        time_scaling=None,
        state_scaling=None,
        control_scaling=None,
    ):
        self.g = 2.0
        self.m = 0.1
        self.I_xx = 0.1

        super().__init__(
            state_dim=6,
            control_dim=2,
            system_params=None,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )

    def _dynamics(self, x, u, t):
        z, y, phi, z_dot, y_dot, phi_dot = x
        x0_dot = z_dot
        x1_dot = y_dot
        x2_dot = phi_dot
        x3_dot = self.g - u[0] / self.m * jnp.cos(phi)
        x4_dot = u[0] / self.m * jnp.sin(phi)
        x5_dot = u[1] / self.I_xx
        return jnp.array(
            [x0_dot, x1_dot, x2_dot, x3_dot, x4_dot, x5_dot], dtype=jnp.float64
        )


class Linear(SimulatorDynamics):
    """
    Linear dynamics
    """

    def __init__(
        self,
        triplet: Tuple[int, int, int] = (2, 2, 2),
        key: int = 12345,
        system_matrix: jnp.array = None,
        control_matrix: jnp.array = None,
        time_scaling=None,
        state_scaling=None,
        control_scaling=None,
    ):
        """
        Parameters
        ----------
        triplet (a, b, c)
        a   number of stable modes
        b   number of marginally stable modes
        c   number of unstable modes
        key random number for jax randomness

        """
        self.triplet = triplet
        super().__init__(
            state_dim=sum(self.triplet),
            control_dim=control_matrix.shape[0],
            system_params=None,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )
        if system_matrix is None:
            system_matrix = create_matrix(triplet, key)
        self.system_matrix = system_matrix
        self.control_matrix = control_matrix

    def _dynamics(self, x, t, u):
        return jnp.dot(x, self.system_matrix) + jnp.dot(u, self.control_matrix)


class LotkaVolterra(SimulatorDynamics):
    """
    Lotka Volterra dynamics
    """

    def __init__(
        self,
        system_params=jnp.array([1.0, 1.0, 1.0, 1.0]),
        time_scaling=None,
        state_scaling=None,
        control_scaling=None,
    ):
        super().__init__(
            state_dim=2,
            control_dim=2,
            system_params=system_params,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )

    def _dynamics(self, x, u, t):
        return 10 * (
            jnp.array(
                [
                    self.system_params[0] * x[0] - self.system_params[1] * x[0] * x[1],
                    self.system_params[2] * x[0] * x[1] - self.system_params[3] * x[1],
                ]
            )
            + u
        )


class CartPole(SimulatorDynamics):
    """
    Dynamics of CartPole
    x represents: (\theta, \\dot{\theta}, x, \\dot{x})
    """

    def __init__(
        self,
        system_params=jnp.array([0.5, 1.0, 0.5]),
        g=0.2,
        time_scaling=None,
        state_scaling=None,
        control_scaling=None,
    ):
        super().__init__(
            state_dim=4,
            control_dim=1,
            system_params=system_params,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )
        length, m, m_p = system_params
        self.L = length
        self.M = m
        self.m_p = m_p
        self.g = g

    def _dynamics(self, x, u, t):
        x0_dot = x[1]

        num = -self.m_p * self.L * jnp.sin(x[0]) * jnp.cos(x[0]) * x[1] ** 2
        num = num + (self.m_p + self.M) * self.g * jnp.sin(x[0]) + jnp.cos(x[0]) * u[0]
        denom = (self.M + self.m_p * (1 - jnp.cos(x[0]) ** 2)) * self.L

        x1_dot = num / denom
        x2_dot = x[3]

        num = (
            -self.m_p * self.L * jnp.sin(x[0]) * x[1] ** 2
            + self.m_p * self.g * jnp.sin(x[0]) * jnp.cos(x[0])
            + u[0]
        )
        denom = self.M + self.m_p * (1 - jnp.cos(x[0]) ** 2)

        x3_dot = num / denom
        return jnp.array([x0_dot, x1_dot, x2_dot, x3_dot])


class VanDerPolOscilator(SimulatorDynamics):
    def __init__(self, time_scaling=None, state_scaling=None, control_scaling=None):
        super().__init__(
            state_dim=2,
            control_dim=1,
            system_params=None,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )

    def _dynamics(self, x, u, t):
        x0_dot = x[1]
        x1_dot = (1 - x[0] ** 2) * x[1] - x[0] + u[0]
        return jnp.array([x0_dot, x1_dot])


class Glucose(SimulatorDynamics):
    def __init__(self, time_scaling=None, state_scaling=None, control_scaling=None):
        super().__init__(
            state_dim=2,
            control_dim=1,
            system_params=None,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )

        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
        self.A = 2.0
        self.l = 0.5
        self.T = 0.2
        self.x0 = jnp.array([0.75, 0.0])

    def _dynamics(self, x, u, t):
        d_x = jnp.array([-self.a * x[0] - self.b * x[1], -self.c * x[1] + u[0]])

        return d_x


class MountainCar(SimulatorDynamics):
    def __init__(self, time_scaling=None, state_scaling=None, control_scaling=None):
        super().__init__(
            state_dim=2,
            control_dim=1,
            system_params=None,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )

    def _dynamics(self, x, u, t):
        x0_dot = 10 * x[1]
        x1_dot = 3.5 * u[0] - 2.5 * jnp.cos(3 * x[0])
        return jnp.array([x0_dot, x1_dot])


class FurutaPendulum(SimulatorDynamics):
    def __init__(
        self,
        system_params=jnp.array(
            [8.084e-01, 9.894e-04, 6.361e-03, 7.027e-04, 2.000e0, 1.650e0]
        ),
        g=9.81,
        time_scaling=None,
        state_scaling=None,
        control_scaling=None,
    ):
        # system_params = jnp.array([0.1, 0.0, 1.0, 0.001, 0.1, 1.0])
        # g = 0.2
        super().__init__(
            state_dim=4,
            control_dim=1,
            system_params=system_params,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )
        self.normalizer = AngleNormalizer(
            state_dim=self.state_dim,
            control_dim=self.control_dim,
            angles_dim=[0, 2],
            state_scaling=self.state_scaling,
        )
        """
        https://lucris.lub.lu.se/ws/files/4453844/8727127.pdf
        systems_params = (J, M, m_a, m_p, l_a, l_p)
        Second angle x[2] is the coordinate of the angle which we want on top i.e. 0
        """
        (J, M, m_a, m_p, l_a, l_p) = system_params
        self.alpha = J + (M + 1 / 3 * m_a + m_p) * l_a**2
        self.beta = (M + 1 / 3 * m_p) * l_p**2
        self.gamma = (M + 1 / 2 * m_p) * l_a * l_p
        self.delta = (M + 1 / 2 * m_p) * g * l_p
        # self.alpha = 0.0033472
        # self.beta = 0.0038852
        # self.gamma = 0.0024879
        # self.delta = 0.097625

    def _dynamics(self, x, u, t):
        return self.true_dynamics(x, u, t)

    def true_dynamics(self, x, u, t):
        """
        x is represented in form (phi, dot{phi}, theta, dot{theta})
        """
        x = self.normalizer.transform_x(x)
        t_phi = u[0]
        t_theta = 0

        x0_dot = x[1]

        denom = (
            self.alpha * self.beta
            - self.gamma**2
            + (self.beta**2 + self.gamma**2) * jnp.sin(x[2]) ** 2
        )

        num = (
            self.beta
            * self.gamma
            * (jnp.sin(x[2]) ** 2 - 1)
            * jnp.sin(x[2])
            * x[1] ** 2
        )
        num = num - 2 * self.beta**2 * jnp.cos(x[2]) * jnp.sin(x[2]) * x[1] * x[3]
        num = num + self.beta * self.gamma * jnp.sin(x[2]) * x[3] ** 2
        num = num - self.gamma * self.delta * jnp.cos(x[2]) * jnp.sin(x[2])
        num = num + self.beta * t_phi - self.gamma * jnp.cos(x[2]) * t_theta
        x1_dot = num / denom

        x2_dot = x[3]

        num = (
            self.beta
            * (self.alpha + self.beta * jnp.sin(x[2]) ** 2)
            * jnp.cos(x[2])
            * jnp.sin(x[2])
            * x[1] ** 2
        )
        num = (
            num
            + 2
            * self.beta
            * self.gamma
            * (1 - jnp.sin(x[2]) ** 2)
            * jnp.sin(x[2])
            * x[1]
            * x[3]
        )
        num = num - self.gamma**2 * jnp.cos(x[2]) * jnp.sin(x[2]) * x[3] ** 2
        num = num + self.delta * (
            self.alpha + self.beta * jnp.sin(x[2]) ** 2
        ) * jnp.sin(x[2])
        num = (
            num
            - self.gamma * jnp.cos(x[2]) * t_phi
            + (self.alpha + self.beta * jnp.sin(x[2]) ** 2) * t_theta
        )

        x3_dot = num / denom
        return jnp.array([x0_dot, x1_dot, x2_dot, x3_dot])


class Acrobot(SimulatorDynamics):
    def __init__(
        self,
        system_params=jnp.array([1.0, 1.0, 0.2]),
        time_scaling=None,
        state_scaling=None,
        control_scaling=None,
    ):
        """
        Initialization of Double Pendulum dynamics
        Parameters
        ----------
        m   mass of the rods
        l   length of the rods
        g   gravity
        """
        super().__init__(
            state_dim=4,
            control_dim=1,
            system_params=system_params,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )
        self.m = system_params[0]
        self.l = system_params[1]
        self.g = system_params[2]

    def _dynamics(self, x, u, t):
        theta_1 = x[0]
        theta_2 = x[1]
        x1_dot = (
            6
            / (self.m * self.l**2)
            * (2 * x[2] - 3 * jnp.cos(theta_1 - theta_2) * x[3])
            / (16 - 9 * jnp.cos(theta_1 - theta_2) ** 2)
        )
        x2_dot = (
            6
            / (self.m * self.l**2)
            * (8 * x[3] - 3 * jnp.cos(theta_1 - theta_2) * x[2])
            / (16 - 9 * jnp.cos(theta_1 - theta_2) ** 2)
        )
        x3_dot = (
            (-0.5)
            * self.m
            * self.l**2
            * (
                x1_dot * x2_dot * jnp.sin(theta_1 - theta_2)
                + 3 * self.g / self.l * jnp.sin(theta_1)
            )
        )
        x4_dot = (
            (-0.5)
            * self.m
            * self.l**2
            * (
                -x1_dot * x2_dot * jnp.sin(theta_1 - theta_2)
                + 3 * self.g / self.l * jnp.sin(theta_2)
            )
        )
        return jnp.array([x1_dot, x2_dot, x3_dot, x4_dot + u[0]])


class Swimmer(SimulatorDynamics):
    def __init__(
        self,
        system_params=jnp.array([1.0, 1.0, 1.0, 1.0, 10]),
        time_scaling=None,
        state_scaling=None,
        control_scaling=None,
    ):
        super().__init__(
            state_dim=8,
            control_dim=1,
            system_params=system_params,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )
        self.m_1 = system_params[0]
        self.l_1 = system_params[1]
        self.m_2 = system_params[2]
        self.l_2 = system_params[3]
        self.k = system_params[4]

    def running_cost(
        self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        a_0 = x[0:2]
        a_0_dot = x[2:4]
        theta_1 = x[4]
        theta_1_dot = x[5]
        theta_2 = x[6]
        theta_2_dot = x[7]

        def n(theta):
            return jnp.array([-jnp.sin(theta), jnp.cos(theta)], dtype=jnp.float64)

        a_1_dot = a_0_dot + self.l_1 * theta_1_dot * n(theta_1)
        a_2_dot = a_1_dot + self.l_2 * theta_2_dot * n(theta_2)

        g_1_dot = (a_0_dot + a_1_dot) / 2
        g_2_dot = (a_1_dot + a_2_dot) / 2

        g_center_for_mass_dot = (self.m_1 * g_1_dot + self.m_2 * g_2_dot) / (
            self.m_1 + self.m_2
        )
        g_center_for_mass_x_dot = g_center_for_mass_dot[0]
        # TODO: add constant
        # TODO: rewrite constraints and cost to dynamical systems
        return -g_center_for_mass_x_dot[-1] + u[0]

    def terminal_cost(self):
        pass

    def _dynamics(self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        a_0 = x[0:2]
        a_0_dot = x[2:4]
        theta_1 = x[4]
        theta_1_dot = x[5]
        theta_2 = x[6]
        theta_2_dot = x[7]

        def n(theta):
            return jnp.array([-jnp.sin(theta), jnp.cos(theta)], dtype=jnp.float64)

        def p(theta):
            return jnp.array([jnp.cos(theta), jnp.sin(theta)], dtype=jnp.float64)

        a_1 = a_0 + self.l_1 * p(theta_1)
        a_2 = a_1 + self.l_2 * p(theta_2)

        a_1_dot = a_0_dot + self.l_1 * theta_1_dot * n(theta_1)
        a_2_dot = a_1_dot + self.l_2 * theta_2_dot * n(theta_2)

        g_1 = (a_0 + a_1) / 2
        g_2 = (a_1 + a_2) / 2

        ga_1 = a_1 - g_1
        ga_2 = a_2 - g_2

        ga_1_dot = (a_1_dot - a_0_dot) / 2
        ga_2_dot = (a_2_dot - a_1_dot) / 2

        F_1 = -self.k * self.l_1 * jnp.dot(ga_1_dot, n(theta_1)) * n(theta_1)
        F_2 = -self.k * self.l_2 * jnp.dot(ga_2_dot, n(theta_2)) * n(theta_2)

        M_1 = -self.k * theta_1_dot * self.l_1**3 / 12
        M_2 = -self.k * theta_2_dot * self.l_2**3 / 12

        A = jnp.array(
            [
                [
                    self.m_1,
                    0.0,
                    -1.0,
                    0.0,
                    -0.5 * self.m_1 * self.l_1 * jnp.sin(theta_1),
                    0.0,
                ],
                [
                    0.0,
                    self.m_1,
                    0.0,
                    -1.0,
                    0.5 * self.m_1 * self.l_1 * jnp.cos(theta_1),
                    0.0,
                ],
                [
                    self.m_2,
                    0.0,
                    1.0,
                    0.0,
                    -self.m_2 * self.l_1 * jnp.sin(theta_1),
                    -0.5 * self.m_2 * self.l_2 * jnp.sin(theta_2),
                ],
                [
                    0.0,
                    self.m_2,
                    0.0,
                    1.0,
                    self.m_2 * self.l_1 * jnp.cos(theta_1),
                    0.5 * self.m_2 * self.l_2 * jnp.cos(theta_2),
                ],
                [0.0, 0.0, -ga_1[1], ga_1[0], -self.m_1 * self.l_1 / 12, 0.0],
                [0.0, 0.0, -ga_2[1], ga_2[0], 0.0, -self.m_2 * self.l_2 / 12],
            ]
        )

        b = jnp.array(
            [
                F_1[0]
                + 0.5 * self.m_1 * self.l_1 * theta_1_dot**2 * jnp.cos(theta_1),
                F_1[1]
                + 0.5 * self.m_1 * self.l_1 * theta_1_dot**2 * jnp.sin(theta_1),
                F_2[0]
                + self.m_2 * self.l_1 * theta_1_dot**2 * jnp.cos(theta_1)
                + 0.5 * self.m_2 * self.l_2 * theta_2_dot**2 * jnp.cos(theta_2),
                F_2[1]
                + self.m_2 * self.l_1 * theta_1_dot**2 * jnp.sin(theta_1)
                + 0.5 * self.m_2 * self.l_2 * theta_2_dot**2 * jnp.sin(theta_2),
                u[0] - M_1,
                -u[0] - M_2,
            ]
        )

        y = jnp.linalg.solve(A, b)

        x_dot_0 = a_0_dot[0]
        x_dot_1 = a_0_dot[1]
        x_dot_2 = y[0]
        x_dot_3 = y[1]
        x_dot_4 = theta_1_dot
        x_dot_5 = y[4]
        x_dot_6 = theta_2_dot
        x_dot_7 = y[5]

        return jnp.array(
            [x_dot_0, x_dot_1, x_dot_2, x_dot_3, x_dot_4, x_dot_5, x_dot_6, x_dot_7],
            dtype=jnp.float64,
        )


class QuadrotorEuler(SimulatorDynamics):
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

    def __init__(self, time_scaling=None, state_scaling=None, control_scaling=None):
        super().__init__(
            state_dim=12,
            control_dim=4,
            system_params=None,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )
        # self.scaling_u = jnp.diag(jnp.array([1, 10, 10, 10], dtype=jnp.float64))
        # self.scaling_u_inv = jnp.diag(jnp.array([1, 0.1, 0.1, 0.1], dtype=jnp.float64))
        self.mass = 0.18  # kg
        self.g = 0.81  # m/s^2
        self.arm_length = 0.086  # meter
        self.height = 0.05

        self.I = jnp.array(
            [(0.00025, 0, 2.55e-6), (0, 0.000232, 0), (2.55e-6, 0, 0.0003738)]
        )

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
        self.A = jnp.array(
            [
                [1, 1, 1, 1],
                [0, self.L, 0, -self.L],
                [-self.L, 0, self.L, 0],
                [self.r, -self.r, self.r, -self.r],
            ]
        )

        self.invA = jnp.linalg.inv(self.A)

        self.body_frame = jnp.array(
            [
                (self.L, 0, 0, 1),
                (0, self.L, 0, 1),
                (-self.L, 0, 0, 1),
                (0, -self.L, 0, 1),
                (0, 0, 0, 1),
                (0, 0, self.H, 1),
            ]
        )

        self.B = jnp.array([[0, self.L, 0, -self.L], [-self.L, 0, self.L, 0]])

        self.internal_control_scaling_inv = jnp.diag(
            jnp.array([1, 2 * 1e-4, 2 * 1e-4, 1e-3], dtype=jnp.float64)
        )

    def _dynamics(self, state, u, t):
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
        accel = (
            1.0
            / self.mass
            * (
                wRb.dot(jnp.array([[0, 0, F]]).T)
                - jnp.array([[0, 0, self.mass * self.g]]).T
            )
        )
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
        return jnp.array(
            [
                state_dot_0,
                state_dot_1,
                state_dot_2,
                state_dot_3,
                state_dot_4,
                state_dot_5,
                state_dot_6,
                state_dot_7,
                state_dot_8,
                state_dot_9,
                state_dot_10,
                state_dot_11,
            ]
        )


class QuadrotorQuaternions(SimulatorDynamics):
    """
    Dynamics of quadrotor with 13 dimensional state space and 4 dimensional control
    Code adapted from : https://github.com/Bharath2/Quadrotor-Simulation
    Theory from: https://repository.upenn.edu/cgi/viewcontent.cgi?article=1705&context=edissertations
    Why 4 dimensional control: https://www.youtube.com/watch?v=UC8W3SfKGmg (talks about that at around 8min in video)
    Short description for 4 dimensional control:
          [ F  ]         [ F1 ]
          | M1 |  = A *  | F2 |
          | M2 |         | F3 |
          [ M3 ]         [ F4 ]
    """

    def __init__(self, time_scaling=None, state_scaling=None, control_scaling=None):
        super().__init__(
            state_dim=13,
            control_dim=4,
            system_params=None,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )
        # self.scaling_u = jnp.diag(jnp.array([1, 10, 10, 10], dtype=jnp.float64))
        # self.scaling_u_inv = jnp.diag(jnp.array([1, 0.1, 0.1, 0.1], dtype=jnp.float64))
        self.mass = 0.18  # kg
        self.g = 9.81  # m/s^2
        self.arm_length = 0.086  # meter
        self.height = 0.05

        self.I = jnp.array(
            [(0.00025, 0, 2.55e-6), (0, 0.000232, 0), (2.55e-6, 0, 0.0003738)]
        )

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
        self.A = jnp.array(
            [
                [1, 1, 1, 1],
                [0, self.L, 0, -self.L],
                [-self.L, 0, self.L, 0],
                [self.r, -self.r, self.r, -self.r],
            ]
        )

        self.invA = jnp.linalg.inv(self.A)

        self.body_frame = jnp.array(
            [
                (self.L, 0, 0, 1),
                (0, self.L, 0, 1),
                (-self.L, 0, 0, 1),
                (0, -self.L, 0, 1),
                (0, 0, 0, 1),
                (0, 0, self.H, 1),
            ]
        )

        self.B = jnp.array([[0, self.L, 0, -self.L], [-self.L, 0, self.L, 0]])

        self.internal_control_scaling_inv = jnp.diag(
            jnp.array([1, 2 * 1e-4, 2 * 1e-4, 1e-3], dtype=jnp.float64)
        )

    def _dynamics(self, state, u, t):
        # u = self.scaling_u_inv @ u
        # Here we have to decide in which coordinate system we will operate
        # If we operate with F1, F2, F3 and F4 we need to run
        # u = self.A @ u
        u = self.internal_control_scaling_inv @ u
        F, M = u[0], u[1:]
        x, y, z, xdot, ydot, zdot, qw, qx, qy, qz, p, q, r = state
        quat = jnp.array([qw, qx, qy, qz])

        bRw = Quaternion(quat).as_rotation_matrix()  # world to body rotation matrix
        wRb = bRw.T  # orthogonal matrix inverse = transpose
        # acceleration - Newton's second law of motion
        accel = (
            1.0
            / self.mass
            * (
                wRb.dot(jnp.array([[0, 0, F]]).T)
                - jnp.array([[0, 0, self.mass * self.g]]).T
            )
        )
        # angular velocity - using quternion
        # http://www.euclideanspace.com/physics/kinematics/angularvelocity/
        K_quat = 2.0  # this enforces the magnitude 1 constraint for the quaternion
        quaterror = 1.0 - (qw**2 + qx**2 + qy**2 + qz**2)
        qdot = (-1.0 / 2) * jnp.array(
            [[0, -p, -q, -r], [p, 0, -r, q], [q, r, 0, -p], [r, -q, p, 0]]
        ).dot(quat) + K_quat * quaterror * quat

        # angular acceleration - Euler's equation of motion
        # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        omega = jnp.array([p, q, r])
        pqrdot = self.invI.dot(M.flatten() - jnp.cross(omega, self.I.dot(omega)))
        state_dot_0 = xdot
        state_dot_1 = ydot
        state_dot_2 = zdot
        state_dot_3 = accel[0].reshape()
        state_dot_4 = accel[1].reshape()
        state_dot_5 = accel[2].reshape()
        state_dot_6 = qdot[0]
        state_dot_7 = qdot[1]
        state_dot_8 = qdot[2]
        state_dot_9 = qdot[3]
        state_dot_10 = pqrdot[0]
        state_dot_11 = pqrdot[1]
        state_dot_12 = pqrdot[2]
        return jnp.array(
            [
                state_dot_0,
                state_dot_1,
                state_dot_2,
                state_dot_3,
                state_dot_4,
                state_dot_5,
                state_dot_6,
                state_dot_7,
                state_dot_8,
                state_dot_9,
                state_dot_10,
                state_dot_11,
                state_dot_12,
            ]
        )


class CarParams(NamedTuple):
    """
    Range taken from: https://www.jstor.org/stable/pdf/44470677.pdf
    d_f, d_r : Represent grip of the car -> High grip means d_f, d_r = 1.0. Low grip d_f, d_r ~ 0.0,
                Typically 0.8 - 0.9
    b_f, b_r: Slope of the pacejka. Typically, between [0.5 - 2.5].

    delta_limit: [0.3 - 0.5] -> Limit of the steering angle.

    c_f, c_r: [1.0 2.0] # motor parameters: source https://web.mit.edu/drela/Public/web/qprop/motor1_theory.pdf,
    https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Embedded
    -Control-Systems/LectureNotes/6_Motor_Control.pdf # or look at:
    https://video.ethz.ch/lectures/d-mavt/2021/spring/151-0593-00L/00718f4f-116b-4645-91da-b9482164a3c7.html :
    lecture 2 part 2
    c_m_1: max current of motor: [0.2 - 0.5] c_m_2: motor resistance due to shaft: [0.01 - 0.15]

    c_rr: zero torque current: [0 0.1]

    tv_p: [0.0 0.1]

    """

    m: jax.Array = jnp.array(0.05)  # [0.04, 0.08]
    i_com: jax.Array = jnp.array(27.8e-6)  # [1e-6, 5e-6]
    l_f: jax.Array = jnp.array(0.03)  # [0.025, 0.05]
    l_r: jax.Array = jnp.array(0.035)  # [0.025, 0.05]
    g: jax.Array = jnp.array(9.81)
    d_f: jax.Array = jnp.array(0.02)  # [0.015, 0.025]
    c_f: jax.Array = jnp.array(1.2)  # [1.2, 1.5]
    b_f: jax.Array = jnp.array(2.5)  # [2.2, 2.8]
    d_r: jax.Array = jnp.array(0.017)  # [0.015, 0.025]
    c_r: jax.Array = jnp.array(1.25)  # [1.2, 1.5]
    b_r: jax.Array = jnp.array(2.6)  # [2.2, 2.8]
    c_m_1: jax.Array = jnp.array(0.2)  # [0.2, 0.4]
    c_m_2: jax.Array = jnp.array(0.05)  # [0.04, 0.007]
    c_rr: jax.Array = jnp.array(0.003)  # [0.001, 0.01]
    c_d: jax.Array = jnp.array(0.052)  # [0.001, 0.1]
    steering_limit: jax.Array = jnp.array(0.35)
    use_blend: bool = True
    use_kinematic_model: bool = True
    tv_p: jax.Array = jnp.array(0.0)


class RaceCar(SimulatorDynamics):
    def __init__(
        self,
        time_scaling=None,
        state_scaling=None,
        params: CarParams = CarParams(),
        control_scaling=None,
        control_ratio: int = 10,
    ):
        super().__init__(
            state_dim=6,
            control_dim=2,
            system_params=None,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )
        self.control_ratio = control_ratio
        self.system_params = params

    @staticmethod
    def _ode_kin(x, u, params: CarParams):
        p_x, p_y, theta, v_x = x[0], x[1], x[2], x[3]  # progress
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d = params.c_d
        c_rr = params.c_rr
        delta, d = u[0], u[1]

        delta = jnp.clip(
            delta, a_min=-params.steering_limit, a_max=params.steering_limit
        )
        d = jnp.clip(d, a_min=-1, a_max=1)

        # d_0 = (c_rr + c_d * (v_x ** 2)) / (c_m_1 - c_m_2 * v_x)
        # d_slow = jnp.maximum(d, d_0)
        # d_fast = d

        # slow_ind = v_x <= 0.1
        # d_applied = d_slow * slow_ind + d_fast * (~slow_ind)
        f_r_x = ((c_m_1 - c_m_2 * v_x) * d - c_rr - c_d * (v_x**2)) / m

        beta = jnp.arctan(l_r * jnp.tan(delta) / (l_r + l_f))
        p_x_dot = v_x * jnp.cos(beta)  # s_dot
        p_y_dot = v_x * jnp.sin(beta)  # d_dot
        w = v_x * jnp.sin(beta) / l_r

        dx_kin = jnp.asarray([p_x_dot, p_y_dot, w, f_r_x])
        return dx_kin

    def _accelerations(self, x, u, params: CarParams):
        i_com = params.i_com
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        d_f = params.d_f * params.g
        d_r = params.d_r * params.g
        c_f = params.c_f
        c_r = params.c_r
        b_f = params.b_f
        b_r = params.b_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_rr = params.c_rr

        c_d = params.c_d

        delta, d = u[0], u[1]

        alpha_f = -jnp.arctan((w * l_f + v_y) / (v_x + 1e-6)) + delta
        alpha_r = jnp.arctan((w * l_r - v_y) / (v_x + 1e-6))
        f_f_y = d_f * jnp.sin(c_f * jnp.arctan(b_f * alpha_f))
        f_r_y = d_r * jnp.sin(c_r * jnp.arctan(b_r * alpha_r))
        f_r_x = (c_m_1 - c_m_2 * v_x) * d - c_rr - c_d * (v_x**2)

        v_x_dot = (f_r_x - f_f_y * jnp.sin(delta) + m * v_y * w) / m
        v_y_dot = (f_r_y + f_f_y * jnp.cos(delta) - m * v_x * w) / m
        w_dot = (f_f_y * l_f * jnp.cos(delta) - f_r_y * l_r) / i_com

        acceleration = jnp.array([v_x_dot, v_y_dot, w_dot])
        return acceleration

    def _ode_dyn(self, x, u, params: CarParams):
        # state = [p_x, p_y, theta, v_x, v_y, w]. Velocities are in local coordinate frame.
        # Inputs: [\delta, d] -> \delta steering angle and d duty cycle of the electric motor.
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        p_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        theta_dot = w
        p_x_dot = jnp.array([p_x_dot, p_y_dot, theta_dot])

        accelerations = self._accelerations(x, u, params)

        x_dot = jnp.concatenate([p_x_dot, accelerations], axis=-1)
        return x_dot

    def _compute_dx_kin(self, x, u, params: CarParams):
        l_r = params.l_r
        l_f = params.l_f
        v_x = x[3]
        v_y = x[4]
        x_kin = jnp.asarray([x[0], x[1], x[2], jnp.sqrt(v_x**2 + v_y**2)])
        dxkin = self._ode_kin(x_kin, u, params)
        delta = u[0]
        beta = jnp.arctan(l_r * jnp.tan(delta) / l_f + l_r)
        v_x_state = dxkin[3] * jnp.cos(beta)  # V*cos(beta)
        v_y_state = dxkin[3] * jnp.sin(beta)  # V*sin(beta)
        w = v_x_state * jnp.arctan(delta) / (l_f + l_r)
        dx_kin_full = jnp.asarray(
            [dxkin[0], dxkin[1], dxkin[2], v_x_state, v_y_state, w]
        )
        return dx_kin_full

    def _compute_dx(self, x, u, params: CarParams):
        use_kin = params.use_kinematic_model * (1 - params.use_blend)
        use_dyn = (1 - params.use_kinematic_model) * (1 - params.use_blend)

        v_x = x[3]
        blend_ratio = (v_x - 0.3) / (0.2)
        lambda_blend = jnp.min(jnp.asarray([jnp.max(jnp.asarray([blend_ratio, 0])), 1]))
        dx_kin_full = self._compute_dx_kin(x, u, params)
        dxdyn = self._ode_dyn(x=x, u=u, params=params)
        dx_blend = lambda_blend * dxdyn + (1 - lambda_blend) * dx_kin_full
        dx_kin = self._compute_dx_kin(x, u, params)
        dx_dyn = self._ode_dyn(x=x, u=u, params=params)
        dx = params.use_blend * dx_blend + use_kin * dx_kin + use_dyn * dx_dyn
        return dx

    def _ode(self, x, u, params: CarParams):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/manish-pra/copg/blob/4a370594ab35f000b7b43b1533bd739f70139e4e/car_racing_simulator/VehicleModel.py#L381
        """
        delta, d = u[0], u[1]
        delta = jnp.clip(
            delta, a_min=-params.steering_limit, a_max=params.steering_limit
        )
        d = jnp.clip(d, a_min=-1, a_max=1)
        d = d * 0.6 + 0.4
        u = u.at[0].set(delta)
        u = u.at[1].set(d)
        v_x = x[3]
        idx = jnp.logical_and(
            v_x <= 0.1,
            u[0]
            <= (params.c_rr + params.c_d * v_x**2)
            / (params.c_m_1 - params.c_m_2 * v_x),
        )

        def stop_acceleration_update(x, u, param: CarParams):
            return jnp.zeros_like(x)

        dx = jax.lax.cond(
            idx,
            stop_acceleration_update,
            self._compute_dx,
            x,
            u,
            params,
        )
        return dx

    def _dynamics(self, x: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return self._ode(x, u, self.system_params)


class CancerTreatment(SimulatorDynamics):
    def __init__(self, time_scaling=None, state_scaling=None, control_scaling=None):
        super().__init__(
            state_dim=1,
            control_dim=1,
            system_params=None,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )

        self.r = 0.3
        """Growth rate of the tumour"""
        self.a = 3.0
        """Positive weight parameter"""
        self.delta = 0.45
        """Magnitude of the dose administered"""
        self.T = 20.0
        self.x0 = jnp.array([0.975])

    def _dynamics(self, x, u, t):
        d_x = self.r * x[0] * jnp.log(1 / x[0]) - u[0] * self.delta * x[0]
        return jnp.array(d_x).reshape(
            self.state_dim,
        )


class HIVTreatment(SimulatorDynamics):
    def __init__(self, time_scaling=None, state_scaling=None, control_scaling=None):
        super().__init__(
            state_dim=3,
            control_dim=1,
            system_params=None,
            time_scaling=time_scaling,
            state_scaling=state_scaling,
            control_scaling=control_scaling,
        )

        self.s = 10.0
        self.m_1 = 0.02
        self.m_2 = 0.5
        self.m_3 = 4.4
        self.r = 0.03
        self.T_max = 1500.0
        self.k = 0.000024
        self.N = 300.0
        self.x_0 = (800.0, 0.04, 1.5)
        self.A = 0.05
        self.T = 20.0

    def _dynamics(self, x, u, t):
        d_x = jnp.array(
            [
                self.s / (1 + x[2])
                - self.m_1 * x[0]
                + self.r * x[0] * (1 - (x[0] + x[1]) / self.T_max)
                - u[0] * self.k * x[0] * x[2],
                u[0] * self.k * x[0] * x[2] - self.m_2 * x[1],
                self.N * self.m_2 * x[1] - self.m_3 * x[2],
            ]
        )
        return d_x


def get_simulator_dynamics(
    simulator: SimulatorType, scaling: Scaling
) -> SimulatorDynamics:
    if simulator == SimulatorType.LOTKA_VOLTERRA:
        return LotkaVolterra(**scaling._asdict())
    elif simulator == SimulatorType.LINEAR:
        return Linear(**scaling._asdict())
    elif simulator == SimulatorType.PENDULUM:
        return Pendulum(**scaling._asdict())
    elif simulator == SimulatorType.VAN_DER_POL_OSCILATOR:
        return VanDerPolOscilator(**scaling._asdict())
    elif simulator == SimulatorType.MOUNTAIN_CAR:
        return MountainCar(**scaling._asdict())
    elif simulator == SimulatorType.CARTPOLE:
        return CartPole(**scaling._asdict())
    elif simulator == SimulatorType.BICYCLE:
        return Bicycle(**scaling._asdict())
    elif simulator == SimulatorType.FURUTA_PENUDLUM:
        return FurutaPendulum(**scaling._asdict())
    elif simulator == SimulatorType.ACROBOT:
        return Acrobot(**scaling._asdict())
    elif simulator == SimulatorType.QUADROTOR_QUATERNIONS:
        return QuadrotorQuaternions(**scaling._asdict())
    elif simulator == SimulatorType.QUADROTOR_EULER:
        return QuadrotorEuler(**scaling._asdict())
    elif simulator == SimulatorType.QUADROTOR_2D:
        return Quadrotor2D(**scaling._asdict())
    elif simulator == SimulatorType.RACE_CAR:
        return RaceCar(**scaling._asdict())
    elif simulator == SimulatorType.CANCER_TREATMENT:
        return CancerTreatment(**scaling._asdict())
    elif simulator == SimulatorType.GLUCOSE:
        return Glucose(**scaling._asdict())


if __name__ == "__main__":
    Pendulum()
