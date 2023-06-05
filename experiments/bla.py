import chex
import jax.numpy as jnp


@chex.dataclass
class Parameters:
    x: int
    y: str


parameters = Parameters(
    x=3,
    y="test",
)
