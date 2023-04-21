from typing import NamedTuple

import jax
import numpy as np
import tensorflow as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer, py_uniform_replay_buffer


data_spec = (
    tf.TensorSpec([3], tf.float32, 'action'),
    (
        tf.TensorSpec([5], tf.float32, 'lidar'),
        tf.TensorSpec([3, 2], tf.float32, 'camera')
    )
)


class Data(NamedTuple):
    xs: jax.Array
    ys: jax.Array


batch_size = 32
max_length = 1000

sample_data = Data(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2, 3], [4, 5, 6]]))

replay_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
    sample_data,
    capacity=1000 * 32,
    )
