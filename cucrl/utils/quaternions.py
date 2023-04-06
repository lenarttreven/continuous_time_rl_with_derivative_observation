import jax.numpy as jnp
from jax.lax import cond
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class Quaternion:
    """Quaternions for 3D rotations"""

    def __init__(self, x):
        self.x = jnp.asarray(x, dtype=jnp.float64)

    def tree_flatten(self):
        children = self.x  # arrays / dynamic values
        aux_data = {}  # static values
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, _, children):
        x = children
        return cls(x=x)

    @classmethod
    def from_v_theta(cls, v, theta):
        """ Construct quaternion from unit vector v and rotation angle theta"""
        theta = jnp.asarray(theta)
        v = jnp.asarray(v)

        s = jnp.sin(0.5 * theta)
        c = jnp.cos(0.5 * theta)
        vnrm = jnp.sqrt(jnp.sum(v * v))

        q = jnp.concatenate([[c], s * v / vnrm])
        return cls(q)

    def get_array(self):
        return self.x

    def __eq__(self, other):
        return jnp.array_equal(self.x, other.xs)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "Quaternion:\n" + self.x.__repr__()

    def __mul__(self, other):
        # multiplication of two quaternions.
        prod = self.x[:, None] * other.xs

        return self.__class__([(prod[0, 0] - prod[1, 1]
                                - prod[2, 2] - prod[3, 3]),
                               (prod[0, 1] + prod[1, 0]
                                + prod[2, 3] - prod[3, 2]),
                               (prod[0, 2] - prod[1, 3]
                                + prod[2, 0] + prod[3, 1]),
                               (prod[0, 3] + prod[1, 2]
                                - prod[2, 1] + prod[3, 0])])

    def as_v_theta(self):
        """Return the v, theta equivalent of the (normalized) quaternion"""
        # compute theta
        norm = jnp.sqrt((self.x ** 2).sum(0))
        # assert (norm != 0)
        theta = 2 * jnp.arccos(self.x[0] / norm)

        # compute the unit vector
        # v = jnp.array(self.x[1:], order='F', copy=True)
        v = jnp.array(self.x[1:], copy=True)
        length = jnp.sqrt(jnp.sum(v ** 2, 0))

        def true_fun(v, length):
            v /= length
            return v

        def false_fun(v, _):
            return v

        v = cond(length > 0, true_fun, false_fun, v, length)
        # if length > 0.0:
        #     v /= length
        return v, theta

    def as_rotation_matrix(self):
        """Return the rotation matrix of the (normalized) quaternion
           https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
           Improving computation speed https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4435132/
           """
        v, theta = self.as_v_theta()
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        return jnp.array([[v[0] * v[0] * (1. - c) + c,
                           v[0] * v[1] * (1. - c) - v[2] * s,
                           v[0] * v[2] * (1. - c) + v[1] * s],
                          [v[1] * v[0] * (1. - c) + v[2] * s,
                           v[1] * v[1] * (1. - c) + c,
                           v[1] * v[2] * (1. - c) - v[0] * s],
                          [v[2] * v[0] * (1. - c) - v[1] * s,
                           v[2] * v[1] * (1. - c) + v[0] * s,
                           v[2] * v[2] * (1. - c) + c]])
