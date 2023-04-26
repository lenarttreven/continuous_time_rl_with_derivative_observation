import jax


def jax_has_gpu():
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        print('Yay')
    except Exception as err:
        print('Boop', err)


if __name__ == '__main__':
    jax_has_gpu()
