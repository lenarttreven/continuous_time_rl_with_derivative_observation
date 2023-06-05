import jax.numpy as jnp


def quadratic_cost(
    x: jnp.ndarray,
    u: jnp.ndarray,
    x_target: jnp.ndarray,
    u_target: jnp.ndarray,
    q: jnp.ndarray,
    r: jnp.ndarray,
) -> jnp.ndarray:
    assert (
        x.ndim == u.ndim == 1
        and q.ndim == r.ndim == 2
        and x.shape[0] == q.shape[0]
        and u.shape[0] == r.shape[0]
    )
    norm_x = x - x_target
    norm_u = u - u_target
    return norm_x @ q @ norm_x + norm_u @ r @ norm_u


def test():
    state_dim, control_dim = 3, 2
    x = jnp.ones(shape=(state_dim,))
    u = jnp.ones(shape=(control_dim,))
    q, r = jnp.eye(state_dim), jnp.eye(control_dim)
    out = quadratic_cost(x, u, x, u, q, r)
    assert out.shape == ()


if __name__ == "__main__":
    test()
