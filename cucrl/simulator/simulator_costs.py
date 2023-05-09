from abc import ABC, abstractmethod
from functools import partial
from typing import Any

import jax.numpy as jnp
from jax import jit

from cucrl.cost.quadratic_cost import quadratic_cost
from cucrl.main.config import Scaling
from cucrl.utils.race_car_params import CarParams
from cucrl.utils.representatives import SimulatorType

pytree = Any


class SimulatorCostsAndConstraints(ABC):
    def __init__(self, state_dim: int, control_dim: int, system_params: pytree, time_scaling: jnp.ndarray | None = None,
                 state_scaling: jnp.ndarray | None = None, control_scaling: jnp.ndarray | None = None):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.system_params = system_params

        self.state_target = None
        self.action_target = None

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
    def running_cost(self, x, u) -> jnp.ndarray:
        assert x.shape == (self.state_dim,) and u.shape == (self.control_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        return self._running_cost(x, u) / self.time_scaling.reshape()

    @abstractmethod
    def _running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        pass

    @partial(jit, static_argnums=0)
    def tracking_running_cost(self, x, u) -> jnp.ndarray:
        assert x.shape == (self.state_dim,) and u.shape == (self.control_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        return self._tracking_running_cost(x, u) / self.time_scaling.reshape()

    @abstractmethod
    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        pass

    @partial(jit, static_argnums=0)
    def tracking_terminal_cost(self, x, u) -> jnp.ndarray:
        assert x.shape == (self.state_dim,) and u.shape == (self.control_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        return self._tracking_running_cost(x, u) / self.time_scaling.reshape()

    @abstractmethod
    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        pass

    @partial(jit, static_argnums=0)
    def terminal_cost(self, x, u) -> jnp.ndarray:
        assert x.shape == (self.state_dim,) and u.shape == (self.control_dim,)
        x = self.state_scaling_inv @ x
        u = self.control_scaling_inv @ u
        return self._terminal_cost(x, u)

    @abstractmethod
    def _terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        pass

    @partial(jit, static_argnums=0)
    def inequality(self, x, u) -> jnp.ndarray:
        assert x.shape == (self.state_dim,) and u.shape == (self.control_dim,)
        return self._inequality(x, u)

    @abstractmethod
    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        pass


class Pendulum(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=jnp.array([5.0, 9.81]), time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=2, control_dim=1, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([0, 0], dtype=jnp.float64)
        self.action_target = jnp.array([0], dtype=jnp.float64)
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)
        self.terminal_q = jnp.eye(self.state_dim)
        self.terminal_q_mpc = jnp.eye(self.state_dim)
        self.terminal_r = 0 * jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = jnp.eye(self.state_dim)
        self.tracking_r_T = 0 * jnp.eye(self.control_dim)

    def _running_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _terminal_cost_mpc(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q_mpc,
                              r=self.terminal_r)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q, r=self.tracking_r)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q_T, r=self.tracking_r_T)


class Acrobot(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=jnp.array([1.0, 1.0, 0.2]), time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=4, control_dim=1, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)

        self.state_target = jnp.array([jnp.pi, jnp.pi, 0, 0], dtype=jnp.float64)
        self.action_target = jnp.array([0], dtype=jnp.float64)
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)
        self.terminal_q = jnp.eye(self.state_dim)
        self.terminal_q_mpc = jnp.eye(self.state_dim)
        self.terminal_r = 0 * jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = jnp.eye(self.state_dim)
        self.tracking_r_T = 0 * jnp.eye(self.control_dim)

    def _running_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _terminal_cost_mpc(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q_mpc,
                              r=self.terminal_r)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q, r=self.tracking_r)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q_T, r=self.tracking_r_T)


class Quadrotor2D(SimulatorCostsAndConstraints):
    def __init__(self, system_params=None, time_scaling=None, state_scaling=None, control_scaling=None):
        self.g = 2.0
        self.m = 0.1
        self.I_xx = 0.1
        super().__init__(state_dim=6, control_dim=2, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([0, 0, 0, 0, 0, 0], dtype=jnp.float64)
        self.action_target = jnp.array([self.g * self.m, 0], dtype=jnp.float64)
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = 10 * jnp.eye(self.control_dim)
        self.terminal_q = jnp.eye(self.state_dim)
        self.terminal_q_mpc = jnp.eye(self.state_dim)
        self.terminal_r = 0 * jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = jnp.eye(self.state_dim)
        self.tracking_r_T = 0 * jnp.eye(self.control_dim)

    def _running_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _terminal_cost_mpc(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q_mpc,
                              r=self.terminal_r)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q, r=self.tracking_r)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q_T, r=self.tracking_r_T)


class Glucose(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=None, time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=2, control_dim=1, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([0.47773321, 0.33197129], dtype=jnp.float64)
        self.action_target = jnp.zeros(shape=(self.control_dim,), dtype=jnp.float64)
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)
        self.terminal_q = jnp.eye(self.state_dim)
        self.terminal_r = jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = jnp.eye(self.state_dim)
        self.tracking_r_T = jnp.zeros(self.control_dim)

        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
        self.A = 2.0
        self.l = 0.5
        self.T = 0.2
        self.x0 = jnp.array([.75, 0.])

    def _running_cost(self, x, u):
        return 100 * (self.A * (x[0] - self.l) ** 2 + u[0] ** 2)

    def _terminal_cost(self, x, u):
        return 0.0

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q, r=self.tracking_r)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q_T, r=self.tracking_r_T)


class CancerTreatment(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=None, time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=1, control_dim=1, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([0.54251185], dtype=jnp.float64)
        self.action_target = jnp.zeros(shape=(self.control_dim,), dtype=jnp.float64)
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)
        self.terminal_q = jnp.eye(self.state_dim)
        self.terminal_r = jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = jnp.eye(self.state_dim)
        self.tracking_r_T = jnp.zeros(self.control_dim)

        self.r = 0.3
        """Growth rate of the tumour"""
        self.a = 3.0
        """Positive weight parameter"""
        self.delta = 0.45
        """Magnitude of the dose administered"""
        self.T = 20.0
        self.x0 = jnp.array([0.975])

    def _running_cost(self, x, u):
        return self.a * x[0] ** 2 + u[0] ** 2

    def _terminal_cost(self, x, u):
        return 0.0

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q, r=self.tracking_r)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q_T, r=self.tracking_r_T)


class MountainCar(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=None, time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=2, control_dim=1, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([jnp.pi / 6, 0.0], dtype=jnp.float64)
        self.action_target = jnp.array([0.0], dtype=jnp.float64)
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)
        self.terminal_q = jnp.eye(self.state_dim)
        self.terminal_r = jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = jnp.eye(self.state_dim)
        self.tracking_r_T = jnp.zeros(self.control_dim)

    def _running_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q, r=self.tracking_r)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q_T, r=self.tracking_r_T)


class RaceCar(SimulatorCostsAndConstraints):
    def __init__(self, system_params: CarParams = CarParams(), time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=6, control_dim=2, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([5, -2, 0, 0, 0, 0], dtype=jnp.float64)

        self.final_tracking_state = jnp.array([5.16399081e+00, -2.06869377e+00, -4.02571791e-01, 1.35987639e-01,
                                               -2.20793977e-04, -7.56103134e-04], dtype=jnp.float64)

        self.action_target = jnp.array([0, 0], dtype=jnp.float64)
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)
        self.terminal_q = 0 * jnp.eye(self.state_dim)
        self.terminal_r = 0 * jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = 5 * jnp.eye(self.state_dim)
        self.tracking_r_T = jnp.zeros(self.control_dim)

    def _running_cost(self, x, u):
        u = jnp.tanh(u)
        u = u.at[0].set(u[0] * self.system_params.max_steering)
        return 100 * jnp.sum(u ** 2) + jnp.sum((x[:2] - self.state_target[:2]) ** 2)

    def _terminal_cost(self, x, u):
        # return jnp.zeros(shape=())
        u = jnp.tanh(u)
        u = u.at[0].set(u[0] * self.system_params.max_steering)
        return jnp.sum(u ** 2) + jnp.sum((x[:2] - self.state_target[:2]) ** 2)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(u ** 2) + jnp.sum(x[:2] ** 2)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(u ** 2) + jnp.sum(x[:2] ** 2)


class Bicycle(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=jnp.array([0.1]), time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=4, control_dim=2, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([2.0, 1.0, jnp.pi / 6, 0.0], dtype=jnp.float64)
        self.action_target = jnp.array([0.0, 0.0], dtype=jnp.float64)
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)
        self.terminal_q = jnp.eye(self.state_dim)
        self.terminal_r = jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = 5 * jnp.eye(self.state_dim)
        self.tracking_r_T = 0 * jnp.eye(self.control_dim)

    def _running_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q, r=self.tracking_r)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q_T, r=self.tracking_r_T)


class VanDerPolOscilator(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=jnp.array([0.1]), time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=2, control_dim=1, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([0, 0], dtype=jnp.float64)
        self.action_target = jnp.array([0], dtype=jnp.float64)
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)
        self.terminal_q = 0 * jnp.eye(self.state_dim)
        self.terminal_r = 0 * jnp.eye(self.control_dim)

    def _running_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])


class LotkaVolterra(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=jnp.array([0.1]), time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=2, control_dim=2, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([1.0, 3.0], dtype=jnp.float64)
        self.action_target = jnp.array([2.0, 0.0], dtype=jnp.float64)
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)
        self.terminal_q = jnp.eye(self.state_dim)
        self.terminal_r = 0 * jnp.eye(self.control_dim)

    def _running_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])


class FurutaPendulum(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=jnp.array([0.1]), time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=4, control_dim=1, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        self.action_target = jnp.array([0.0])
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)

        self.terminal_q = jnp.eye(self.state_dim)
        self.terminal_r = 0 * jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = 5 * jnp.eye(self.state_dim)
        self.tracking_r_T = 0 * jnp.eye(self.control_dim)

    def _running_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q, r=self.tracking_r)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q_T, r=self.tracking_r_T)


class SwimmerMujoco(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=None, time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=10, control_dim=2, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.zeros(shape=(self.state_dim,), dtype=jnp.float64)
        self.action_target = jnp.zeros(shape=(self.control_dim,), dtype=jnp.float64)
        self.running_q = 0 * jnp.eye(self.state_dim)
        self.running_r = 0.1 * jnp.eye(self.control_dim)
        self.terminal_q = 0 * jnp.eye(self.state_dim)
        self.terminal_r = 0 * jnp.eye(self.control_dim)

    def _running_cost(self, x, u):
        q_vel = x[5:]
        return -q_vel[0] + quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target,
                                          q=self.running_q, r=self.running_r)

    def _terminal_cost(self, x, u):
        return 0.0

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([1 - u, u + 1])


class CartPole(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=jnp.array([0.1]), time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=4, control_dim=1, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        self.action_target = jnp.array([0.0])

        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)
        self.terminal_q = jnp.eye(self.state_dim)
        self.terminal_r = jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = jnp.eye(self.state_dim)
        self.tracking_r_T = 0 * jnp.eye(self.control_dim)

    def _running_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q, r=self.tracking_r)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q_T, r=self.tracking_r_T)


class QuadrotorQuaternions(SimulatorCostsAndConstraints):
    def __init__(self, system_params=None, time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=13, control_dim=4, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.array([0.0, 0.0, 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], dtype=jnp.float64)
        # self.action_target = jnp.array([0.1458, 0., 0., 0.], dtype=jnp.float64)
        self.action_target = jnp.array([1.0458, 0., 0., 0.], dtype=jnp.float64)

        self.scaling_u = jnp.diag(jnp.array([1, 10, 10, 10], dtype=jnp.float64))
        self.scaling_u_inv = jnp.diag(jnp.array([1, 0.1, 0.1, 0.1], dtype=jnp.float64))

        self.running_q = jnp.eye(self.state_dim)
        self.running_r = 0.1 * jnp.eye(self.control_dim)
        self.terminal_q = 0 * jnp.eye(self.state_dim)
        self.terminal_r = 0 * jnp.eye(self.control_dim)

    def _running_cost(self, x, u):
        u = self.scaling_u_inv @ u
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        u = self.scaling_u_inv @ u
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _terminal_cost_mpc(self, x, u):
        u = self.scaling_u_inv @ u
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])


class QuadrotorEuler(SimulatorCostsAndConstraints):
    def __init__(self, system_params=None, time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=12, control_dim=4, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.state_target = jnp.zeros(shape=(12,), dtype=jnp.float64)
        self.state_target = jnp.array([0, 0, 0,
                                       0, 0, 0,
                                       0, 0, 1.0,
                                       0, 0, 0], dtype=jnp.float64)
        self.action_target = jnp.array([0.1458, 0., 0., 0.], dtype=jnp.float64)
        self.running_q = 1.0 * jnp.diag(jnp.array([1, 1, 1, 1,
                                                   1, 1, 0.1, 0.1,
                                                   0.1, 1, 0.1, 0.1], dtype=jnp.float64))
        self.running_r = 1.0 * jnp.diag(jnp.array([5.0, 0.8, 0.8, 0.3], dtype=jnp.float64))
        self.terminal_q = 1.0 * jnp.eye(self.state_dim)
        self.terminal_r = 0.0 * jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = jnp.eye(self.state_dim)
        self.tracking_r_T = 0 * jnp.eye(self.control_dim)

    def _running_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.running_q,
                              r=self.running_r)

    def _terminal_cost(self, x, u):
        return quadratic_cost(x, u, x_target=self.state_target, u_target=self.action_target, q=self.terminal_q,
                              r=self.terminal_r)

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q, r=self.tracking_r)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q_T, r=self.tracking_r_T)


class HIVTreatment(SimulatorCostsAndConstraints):
    """
    Dynamics of pendulum
    """

    def __init__(self, system_params=None, time_scaling=None, state_scaling=None,
                 control_scaling=None):
        super().__init__(state_dim=3, control_dim=1, system_params=system_params, time_scaling=time_scaling,
                         state_scaling=state_scaling, control_scaling=control_scaling)
        self.s = 10.
        self.m_1 = .02
        self.m_2 = .5
        self.m_3 = 4.4
        self.r = .03,
        self.T_max = 1500.
        self.k = .000024
        self.N = 300.
        self.x_0 = (800., .04, 1.5)
        self.A = .05
        self.T = 20.

        self.state_target = jnp.array([0.54251185], dtype=jnp.float64)
        self.action_target = jnp.zeros(shape=(self.control_dim,), dtype=jnp.float64)
        self.running_q = jnp.eye(self.state_dim)
        self.running_r = jnp.eye(self.control_dim)
        self.terminal_q = jnp.eye(self.state_dim)
        self.terminal_r = jnp.eye(self.control_dim)

        self.tracking_q = jnp.eye(self.state_dim)
        self.tracking_r = jnp.eye(self.control_dim)
        self.tracking_q_T = jnp.eye(self.state_dim)
        self.tracking_r_T = jnp.zeros(self.control_dim)

    def _running_cost(self, x, u):
        return -self.A * x[0] ** 2 + (1 - u[0]) ** 2

    def _terminal_cost(self, x, u):
        return 0.0

    def _inequality(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.0])

    def _tracking_running_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q, r=self.tracking_r)

    def _tracking_terminal_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return quadratic_cost(x, u, x_target=jnp.zeros(shape=(self.state_dim,)),
                              u_target=jnp.zeros(shape=(self.control_dim,)), q=self.tracking_q_T, r=self.tracking_r_T)


def get_simulator_costs(simulator: SimulatorType, scaling: Scaling) -> SimulatorCostsAndConstraints:
    if simulator == SimulatorType.PENDULUM:
        return Pendulum(**scaling._asdict())
    elif simulator == SimulatorType.MOUNTAIN_CAR:
        return MountainCar(**scaling._asdict())
    elif simulator == SimulatorType.BICYCLE:
        return Bicycle(**scaling._asdict())
    elif simulator == SimulatorType.VAN_DER_POL_OSCILATOR:
        return VanDerPolOscilator(**scaling._asdict())
    elif simulator == SimulatorType.LOTKA_VOLTERRA:
        return LotkaVolterra(**scaling._asdict())
    elif simulator == SimulatorType.FURUTA_PENUDLUM:
        return FurutaPendulum(**scaling._asdict())
    elif simulator == SimulatorType.SWIMMER_MUJOCO:
        return SwimmerMujoco(**scaling._asdict())
    elif simulator == SimulatorType.CARTPOLE:
        return CartPole(**scaling._asdict())
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
