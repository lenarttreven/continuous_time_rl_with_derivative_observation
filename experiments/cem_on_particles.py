import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.lax import cond
from trajax.optimizers import ilqr

x_dim = 2
u_dim = 1
dt = 0.01
T = 10

g = 9.81
l = 5.0

initial_state = jnp.array([jnp.pi / 2, 0.0])
initial_actions = jnp.zeros(shape=(int(T / dt), u_dim))
control_low = jnp.array([-5.0]).reshape(
    1,
)
control_high = jnp.array([5.0]).reshape(
    1,
)
num_steps = initial_actions.shape[0]


def cost_fn(x, u, t):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)

    def running_cost(x, u, t):
        return dt * (jnp.sum(x**2) + jnp.sum(u**2))

    def terminal_cost(x, u, t):
        return jnp.sum(x**2)

    return cond(t == num_steps, terminal_cost, running_cost, x, u, t)


def dynamics_fn(x, u, t):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)
    x0 = x[1]
    x1 = u[0] + g / l * jnp.sin(x[0])
    return x + jnp.array([x0, x1]) * dt


from jax import random, vmap, jit
from cucrl.utils.ensembles import DeterministicEnsemble, DataStatsBNN, DataRepr
from cucrl.main.data_stats import Normalizer, Stats
from cucrl.utils.helper_functions import AngleLayerDynamics

xs = random.uniform(key=random.PRNGKey(0), minval=-5, maxval=5, shape=(100, 2))
us = random.uniform(key=random.PRNGKey(0), minval=-5, maxval=5, shape=(100, 1))
ts = random.uniform(key=random.PRNGKey(0), minval=0, maxval=10, shape=(100, 1))

xs_next = vmap(dynamics_fn)(xs, us, ts)

input_data = jnp.concatenate([xs, us], axis=1)
output_data = xs_next

input_dim = 3
output_dim = 2

noise_level = 0.1

data_std = noise_level * jnp.ones(shape=output_data.shape)
data_stats = DataStatsBNN(
    input_stats=Stats(
        mean=jnp.mean(input_data, axis=0), std=jnp.std(input_data, axis=0)
    ),
    output_stats=Stats(
        mean=jnp.mean(output_data, axis=0), std=jnp.std(output_data, axis=0)
    ),
)

angle_layer = AngleLayerDynamics(
    state_dim=input_dim,
    control_dim=0,
    angles_dim=[],
    state_scaling=jnp.eye(input_dim),
)
normalizer = Normalizer(
    state_dim=input_dim, action_dim=output_dim, angle_layer=angle_layer
)

num_particles = 10
model = DeterministicEnsemble(
    input_dim=input_dim,
    output_dim=output_dim,
    features=[64, 64],
    num_particles=num_particles,
    normalizer=normalizer,
)

train_data = DataRepr(xs=input_data, ys=output_data)
start_time = time.time()

model_params, model_stats = model.fit_model(
    dataset=train_data,
    num_epochs=1000,
    data_stats=data_stats,
    data_std=data_std,
    batch_size=64,
)
print(f"Training time: {time.time() - start_time:.2f} seconds")


def dynamics_fn(x, u, t):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)
    x0 = x[1]
    x1 = u[0] + g / l * jnp.sin(x[0])
    return x + jnp.array([x0, x1]) * dt


@jit
def dynamics(x, u, t):
    assert x.shape == (x_dim,) and u.shape == (u_dim,)
    z = jnp.concatenate([x, u], axis=0)
    return jnp.mean(
        vmap(model.apply_eval, in_axes=(0, 0, None, None))(
            model_params, model_stats, z, data_stats
        ),
        axis=0,
    )


out = dynamics(x=initial_state, u=initial_actions[0], t=jnp.array(0.0))

ts = jnp.arange(0, T, dt)

start_time = time.time()
out = ilqr(cost_fn, dynamics, initial_state, initial_actions)
print("Cost: ", out[2])
print("Time taken: ", time.time() - start_time)

start_time = time.time()
out = ilqr(cost_fn, dynamics, initial_state, initial_actions)
print("Cost: ", out[2])
print("Time taken: ", time.time() - start_time)

plt.plot(ts, out[0][:-1, :], label="xs")
plt.title("iLQR warmup")
plt.plot(ts, out[1], label="us")
plt.legend()
plt.show()
