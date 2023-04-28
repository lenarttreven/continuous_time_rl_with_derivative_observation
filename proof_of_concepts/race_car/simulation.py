import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from jax import vmap
from jax.experimental.ode import odeint

from cucrl.simulator.simulator_dynamics import RaceCar
from cucrl.utils.helper_functions import sample_func
from cucrl.utils.splines import MultivariateSplineExt

gamma = 0.1


def prepare_tracking_data():
    def k(x, y):
        return jnp.exp(-jnp.sum(x - y) ** 2 / gamma)

    func = sample_func(k, jr.PRNGKey(0), n_dim=2, t_min=0, t_max=10, num_samples=10, max_value=2, decay_factor=0.4)
    ts = jnp.linspace(0, 10, 100)
    plt.plot(ts, jnp.tanh(vmap(func)(ts)))
    plt.title('Us')
    plt.show()
    sim_dyn = RaceCar()

    def dynamics(x, t):
        u = func(t)
        return sim_dyn.dynamics(x, u, t.reshape(1, ))

    xs = odeint(dynamics, jnp.zeros(shape=(6,)), ts)
    plt.plot(ts, xs)
    plt.title('Xs')
    plt.show()

    return MultivariateSplineExt(ts, xs)


if __name__ == '__main__':
    spline = prepare_tracking_data()
    ts = jnp.linspace(0, 12, 100)
    xs = vmap(spline)(ts)
    plt.plot(ts, xs)
    plt.show()
