from datetime import datetime

import chex
import jax.random as jr
import matplotlib.pyplot as plt
import wandb
from brax import envs
from flax import struct
from jax import jit
from jax import numpy as jnp

from cucrl.brax_sac_implementation.sac import SAC

if __name__ == '__main__':
    wandb.init(
        project='Pendulum SAC',
        group='test group',
    )

    env_name = 'pendulum'
    backend = 'spring'  # @param ['generalized', 'positional', 'spring']
    env = envs.get_environment(env_name=env_name, backend=backend)
    state = jit(env.reset)(rng=jr.PRNGKey(0))

    sac_trainer = SAC(
        environment=env,
        num_timesteps=30_000, num_evals=20, reward_scaling=10,
        episode_length=100, normalize_observations=True, action_repeat=1,
        discounting=0.999, lr_policy=3e-4, lr_alpha=3e-4, lr_q=3e-4, num_envs=16,
        batch_size=64, grad_updates_per_step=32, max_replay_size=2 ** 14, min_replay_size=2 ** 8, num_eval_envs=1,
        deterministic_eval=True, tau=0.005, wd_policy=1e-2, wd_q=1e-2, wd_alpha=1e-2, wandb_logging=True
    )

    max_y = 0
    min_y = -100

    xdata, ydata = [], []
    times = [datetime.now()]


    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics['eval/episode_reward'])
        plt.xlim([0, sac_trainer.num_timesteps])
        plt.ylim([min_y, max_y])
        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.plot(xdata, ydata)
        plt.show()


    make_inference_fn, params, _ = sac_trainer.run_training(key=jr.PRNGKey(0))

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')


    def policy(x, parameters, key):
        return make_inference_fn(parameters, deterministic=True)(x, key)[0]


    init_state = jnp.array([jnp.pi, 0.0])
    dt = 0.1
    T = 100


    def _ode(x, u):
        assert x.shape == (2,) and u.shape == (1,)
        system_params = jnp.array([5.0, 9.81])
        u = 4 * u
        return jnp.array([x[1], system_params[1] / system_params[0] * jnp.sin(x[0]) + u.reshape()])


    def convert_angle(x):
        return jnp.arctan2(jnp.sin(x), jnp.cos(x))


    def next_step(x: chex.Array, parameters):
        u = policy(x, parameters, jr.PRNGKey(0))
        x_next = x + dt * _ode(x, u)
        x_next = x_next.at[0].set(convert_angle(x_next[0]))
        return x_next, 4 * u


    @struct.dataclass
    class StateCarry:
        x: chex.Array
        params: chex.Array


    from jax.lax import scan


    def f(carry: StateCarry, _):
        x_next, u = next_step(carry.x, carry.params)
        carry = carry.replace(x=x_next)
        return carry, (carry.x, u)


    new_carry, (xs, us) = scan(f, StateCarry(x=init_state, params=params), xs=None, length=T)
    ts = jnp.linspace(0, T, 100)
    plt.plot(ts, xs, label='xs')
    plt.plot(ts, us, label='us')
    plt.legend()
    plt.show()
