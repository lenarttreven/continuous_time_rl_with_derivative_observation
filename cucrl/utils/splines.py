import jax.numpy as np
import matplotlib.pyplot as plt
from jax import jit, vmap
from jax.lax import cond
from jax.numpy import array
from jax.numpy import concatenate
from jax.numpy import zeros
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class InterpolatedUnivariateSpline:
    def __init__(self, x, y, k=3, endpoints='not-a-knot', coefficients=None):
        """JAX implementation of kth-order spline interpolation.
        This class aims to reproduce scipy's InterpolatedUnivariateSpline
        functionality using JAX. Not all of the original class's features
        have been implemented yet, notably
        - `w`    : no weights are used in the spline fitting.
        - `bbox` : we assume the boundary to always be [x[0], x[-1]].
        - `ext`  : extrapolation is always active, i.e., `ext` = 0.
        - `k`    : orders `k` > 3 are not available.
        - `check_finite` : no such check is performed.
        (The relevant lines from the original docstring have been included
        in the following.)
        Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
        Spline function passes through all provided points. Equivalent to
        `UnivariateSpline` with s = 0.
        Parameters
        ----------
        x : (N,) array_like
            Input dimension of data points -- must be strictly increasing
        y : (N,) array_like
            input dimension of data points
        k : int, optional
            Degree of the smoothing spline.  Must be 1 <= `k` <= 3.
        endpoints : str, optional, one of {'natural', 'not-a-knot'}
            Endpoint condition for cubic splines, i.e., `k` = 3.
            'natural' endpoints enforce a vanishing second derivative
            of the spline at the two endpoints, while 'not-a-knot'
            ensures that the third derivatives are equal for the two
            left-most `x` of the domain, as well as for the two
            right-most `x`. The original scipy implementation uses
            'not-a-knot'.
        coefficients: list, optional
            Precomputed parameters for spline interpolation. Shouldn't be set
            manually.
        See Also
        --------
        UnivariateSpline : Superclass -- allows knots to be selected by a
            smoothing condition
        LSQUnivariateSpline : spline for which knots are user-selected
        splrep : An older, non object-oriented wrapping of FITPACK
        splev, sproot, splint, spalde
        BivariateSpline : A similar class for two-dimensional spline interpolation
        Notes
        -----
        The number of data points must be larger than the spline degree `k`.
        The general form of the spline can be written as
          f[i](x) = a[i] + b[i](x - x[i]) + c[i](x - x[i])^2 + d[i](x - x[i])^3,
          i = 0, ..., n-1,
        where d = 0 for `k` = 2, and c = d = 0 for `k` = 1.
        The unknown coefficients (a, b, c, d) define a symmetric, diagonal
        linear system of equations, Az = s, where z = b for `k` = 1 and `k` = 2,
        and z = c for `k` = 3. In each case, the coefficients defining each
        spline piece can be expressed in terms of only z[i], z[i+1],
        y[i], and y[i+1]. The coefficients are solved for using
        `np.linalg.solve` when `k` = 2 and `k` = 3.
        """
        # Verify inputs
        self._endpoints = endpoints
        k = int(k)
        assert k in (1, 2, 3), 'Order k must be in {1, 2, 3}.'
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        assert len(x) == len(y), 'Input arrays must be the same length.'
        assert x.ndim == 1 and y.ndim == 1, 'Input arrays must be 1D.'
        n_data = len(x)

        # Difference vectors
        h = np.diff(x)  # x[i+1] - x[i] for i=0,...,n-1
        p = np.diff(y)  # y[i+1] - y[i]

        if coefficients is None:
            # Build the linear system of equations depending on k
            # (No matrix necessary for k=1)
            if k == 1:
                assert n_data > 1, 'Not enough input points for linear spline.'
                coefficients = p / h

            if k == 2:
                assert n_data > 2, 'Not enough input points for quadratic spline.'
                assert endpoints == 'not-a-knot'  # I have only validated this
                # And actually I think it's probably the best choice of border condition

                # The knots are actually in between data points
                knots = (x[1:] + x[:-1]) / 2.0
                # We add 2 artificial knots before and after
                knots = np.concatenate(
                    [
                        np.array([x[0] - (x[1] - x[0]) / 2.0]),
                        knots,
                        np.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                    ]
                )
                n = len(knots)
                # Compute interval lenghts for these new knots
                h = np.diff(knots)
                # postition of data point inside the interval
                dt = x - knots[:-1]

                # Now we build the system natrix
                A = np.diag(
                    np.concatenate(
                        [
                            np.ones(1),
                            (
                                    2 * dt[1:]
                                    - dt[1:] ** 2 / h[1:]
                                    - dt[:-1] ** 2 / h[:-1]
                                    + h[:-1]
                            ),
                            np.ones(1),
                        ]
                    )
                )

                A += np.diag(
                    np.concatenate([-np.array([1 + h[0] / h[1]]), dt[1:] ** 2 / h[1:]]),
                    k=1,
                )
                A += np.diag(
                    np.concatenate([np.atleast_1d(h[0] / h[1]), np.zeros(n - 3)]), k=2
                )

                A += np.diag(
                    np.concatenate(
                        [
                            h[:-1] - 2 * dt[:-1] + dt[:-1] ** 2 / h[:-1],
                            -np.array([1 + h[-1] / h[-2]]),
                        ]
                    ),
                    k=-1,
                )
                A += np.diag(
                    np.concatenate([np.zeros(n - 3), np.atleast_1d(h[-1] / h[-2])]),
                    k=-2,
                )

                # And now we build the RHS vector
                s = np.concatenate([np.zeros(1), 2 * p, np.zeros(1)])

                # Compute spline coefficients by solving the system
                coefficients = np.linalg.solve(A, s)

            if k == 3:
                assert n_data > 3, 'Not enough input points for cubic spline.'
                if endpoints not in ('natural', 'not-a-knot'):
                    print('Warning : endpoints not recognized. Using natural.')
                    endpoints = 'natural'

                # Special values for the first and last equations
                zero = array([0.0])
                one = array([1.0])
                A00 = one if endpoints == 'natural' else array([h[1]])
                A01 = zero if endpoints == 'natural' else array([-(h[0] + h[1])])
                A02 = zero if endpoints == 'natural' else array([h[0]])
                ANN = one if endpoints == 'natural' else array([h[-2]])
                AN1 = (
                    -one if endpoints == 'natural' else array([-(h[-2] + h[-1])])
                )  # A[N, N-1]
                AN2 = zero if endpoints == 'natural' else array([h[-1]])  # A[N, N-2]

                # Construct the tri-diagonal matrix A
                A = np.diag(concatenate((A00, 2 * (h[:-1] + h[1:]), ANN)))
                upper_diag1 = np.diag(concatenate((A01, h[1:])), k=1)
                upper_diag2 = np.diag(concatenate((A02, zeros(n_data - 3))), k=2)
                lower_diag1 = np.diag(concatenate((h[:-1], AN1)), k=-1)
                lower_diag2 = np.diag(concatenate((zeros(n_data - 3), AN2)), k=-2)
                A += upper_diag1 + upper_diag2 + lower_diag1 + lower_diag2

                # Construct RHS vector s
                center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
                s = concatenate((zero, center, zero))
                # Compute spline coefficients by solving the system
                coefficients = np.linalg.solve(A, s)

        # Saving spline parameters for evaluation later
        self.k = k
        self._x = x
        self._y = y
        self._coefficients = coefficients

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        children = (self._x, self._y, self._coefficients)
        aux_data = {'endpoints': self._endpoints, 'k': self.k}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x, y, coefficients = children
        return cls(x, y, coefficients=coefficients, **aux_data)

    def __call__(self, x):
        """Evaluation of the spline.
        Notes
        -----
        Values are extrapolated if x is outside of the original domain
        of knots. If x is less than the left-most knot, the spline piece
        f[0] is used for the evaluation; similarly for x beyond the
        right-most point.
        """
        if self.k == 1:
            t, a, b = self._compute_coeffs(x)
            result = a + b * t

        if self.k == 2:
            t, a, b, c = self._compute_coeffs(x)
            result = a + b * t + c * t ** 2

        if self.k == 3:
            t, a, b, c, d = self._compute_coeffs(x)
            result = a + b * t + c * t ** 2 + d * t ** 3

        return result

    def _compute_coeffs(self, xs):
        """Compute the spline coefficients for a given x."""
        # Retrieve parameters
        x, y, coefficients = self._x, self._y, self._coefficients

        # In case of quadratic, we redefine the knots
        if self.k == 2:
            knots = (x[1:] + x[:-1]) / 2.0
            # We add 2 artificial knots before and after
            knots = np.concatenate(
                [
                    np.array([x[0] - (x[1] - x[0]) / 2.0]),
                    knots,
                    np.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                ]
            )
        else:
            knots = x

        # Determine the interval that x lies in
        ind = np.digitize(xs, knots) - 1
        # Include the right endpoint in spline piece C[m-1]
        ind = np.clip(ind, 0, len(knots) - 2)
        t = xs - knots[ind]
        h = np.diff(knots)[ind]

        if self.k == 1:
            a = y[ind]
            result = (t, a, coefficients[ind])

        if self.k == 2:
            dt = (x - knots[:-1])[ind]
            b = coefficients[ind]
            b1 = coefficients[ind + 1]
            a = y[ind] - b * dt - (b1 - b) * dt ** 2 / (2 * h)
            c = (b1 - b) / (2 * h)
            result = (t, a, b, c)

        if self.k == 3:
            c = coefficients[ind]
            c1 = coefficients[ind + 1]
            a = y[ind]
            a1 = y[ind + 1]
            b = (a1 - a) / h - (2 * c + c1) * h / 3.0
            d = (c1 - c) / (3 * h)
            result = (t, a, b, c, d)

        return result

    def derivative(self, x, n=1):
        """Analytic nth derivative of the spline.
        The spline has derivatives up to its order k.
        """
        assert n in range(self.k + 1), 'Invalid n.'

        if n == 0:
            result = self.__call__(x)
        else:
            # Linear
            if self.k == 1:
                t, a, b = self._compute_coeffs(x)
                result = b

            # Quadratic
            if self.k == 2:
                t, a, b, c = self._compute_coeffs(x)
                if n == 1:
                    result = b + 2 * c * t
                if n == 2:
                    result = 2 * c

            # Cubic
            if self.k == 3:
                t, a, b, c, d = self._compute_coeffs(x)
                if n == 1:
                    result = b + 2 * c * t + 3 * d * t ** 2
                if n == 2:
                    result = 2 * c + 6 * d * t
                if n == 3:
                    result = 6 * d

        return result

    def antiderivative(self, xs):
        """
        Computes the antiderivative of first order of this spline
        """
        # Retrieve parameters
        x, y, coefficients = self._x, self._y, self._coefficients

        # In case of quadratic, we redefine the knots
        if self.k == 2:
            knots = (x[1:] + x[:-1]) / 2.0
            # We add 2 artificial knots before and after
            knots = np.concatenate(
                [
                    np.array([x[0] - (x[1] - x[0]) / 2.0]),
                    knots,
                    np.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
                ]
            )
        else:
            knots = x

        # Determine the interval that x lies in
        ind = np.digitize(xs, knots) - 1
        # Include the right endpoint in spline piece C[m-1]
        ind = np.clip(ind, 0, len(knots) - 2)
        t = xs - knots[ind]

        if self.k == 1:
            a = y[:-1]
            b = coefficients
            h = np.diff(knots)
            cst = np.concatenate([np.zeros(1), np.cumsum(a * h + b * h ** 2 / 2)])
            return cst[ind] + a[ind] * t + b[ind] * t ** 2 / 2

        if self.k == 2:
            h = np.diff(knots)
            dt = x - knots[:-1]
            b = coefficients[:-1]
            b1 = coefficients[1:]
            a = y - b * dt - (b1 - b) * dt ** 2 / (2 * h)
            c = (b1 - b) / (2 * h)
            cst = np.concatenate(
                [np.zeros(1), np.cumsum(a * h + b * h ** 2 / 2 + c * h ** 3 / 3)]
            )
            return cst[ind] + a[ind] * t + b[ind] * t ** 2 / 2 + c[ind] * t ** 3 / 3

        if self.k == 3:
            h = np.diff(knots)
            c = coefficients[:-1]
            c1 = coefficients[1:]
            a = y[:-1]
            a1 = y[1:]
            b = (a1 - a) / h - (2 * c + c1) * h / 3.0
            d = (c1 - c) / (3 * h)
            cst = np.concatenate(
                [
                    np.zeros(1),
                    np.cumsum(a * h + b * h ** 2 / 2 + c * h ** 3 / 3 + d * h ** 4 / 4),
                ]
            )
            return (
                    cst[ind]
                    + a[ind] * t
                    + b[ind] * t ** 2 / 2
                    + c[ind] * t ** 3 / 3
                    + d[ind] * t ** 4 / 4
            )

    def integral(self, a, b):
        """
        Compute a definite integral over a piecewise polynomial.
        Parameters
        ----------
        a : float
            Lower integration bound
        b : float
            Upper integration bound
        Returns
        -------
        ig : array_like
            Definite integral of the piecewise polynomial over [a, b]
        """
        # Swap integration bounds if needed
        sign = 1
        # if b < a:
        #     a, b = b, a
        #     sign = -1
        xs = np.array([a, b])
        return sign * np.diff(self.antiderivative(xs))


@register_pytree_node_class
class MultivariateSpline:
    def __init__(self, x, y, k=3):
        self.x = x
        self.y = y
        self.num_dim = y.shape[1]
        self.splines = [InterpolatedUnivariateSpline(x, y[:, i], endpoints='not-a-knot', k=k) for i in
                        range(self.num_dim)]

    def __call__(self, x):
        return np.stack([spline(x) for spline in self.splines], axis=1)

    def derivative(self, x):
        return np.stack([spline.derivative(x) for spline in self.splines], axis=1)

    def tree_flatten(self):
        children = (self.y, self.x,)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class LinearInterpolation:
    def __init__(self, xs, fs, fs_der):
        assert fs.ndim == fs_der.ndim == 2 and xs.ndim == 1
        # assert xs.shape[0] == fs.shape[0] == fs_der.shape[0] == 2
        self.xs = xs
        self.fs = fs
        self.fs_der = fs_der
        self.coeffs = vmap(self.compute_coeffs, (None, 0, 0))(self.xs, self.fs, self.fs_der)

    def tree_flatten(self):
        children = (self.xs, self.fs, self.fs_der)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @staticmethod
    def compute_coeffs(xs, fs, fs_der):
        assert xs.shape == fs.shape == fs_der.shape == (2,)
        a = np.stack([xs, np.ones_like(xs)], axis=1)
        b = fs
        coefs = np.linalg.solve(a, b)
        return coefs

    def _one_dim(self, x, coeffs):
        assert x.shape == ()
        return coeffs[0] * x + coeffs[1]

    def __call__(self, x):
        return vmap(self._one_dim, in_axes=(None, 0))(x, self.coeffs)


@register_pytree_node_class
class CubicInterpolation:
    def __init__(self, xs, fs, fs_der):
        assert fs.ndim == fs_der.ndim == 2 and xs.ndim == 1
        # assert xs.shape[0] == fs.shape[0] == fs_der.shape[0] == 2
        self.xs = xs
        self.fs = fs
        self.fs_der = fs_der
        self.coeffs = vmap(self.compute_coeffs, (None, 0, 0))(self.xs, self.fs, self.fs_der)

    def tree_flatten(self):
        children = (self.xs, self.fs, self.fs_der)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @staticmethod
    def compute_coeffs(xs, fs, fs_der):
        assert xs.shape == fs.shape == fs_der.shape == (2,)
        first_two_rows = np.stack([xs ** 3, xs ** 2, xs, np.ones_like(xs)], axis=1)
        last_two_rows = np.stack([3 * xs ** 2, 2 * xs, np.ones_like(xs), np.zeros_like(xs)], axis=1)
        a = np.concatenate([first_two_rows, last_two_rows])
        b = np.concatenate([fs, fs_der])
        coefs = np.linalg.solve(a, b)
        return coefs

    def _one_dim(self, x, coeffs):
        assert x.shape == ()
        return coeffs[0] * x ** 3 + coeffs[1] * x ** 2 + coeffs[2] * x + coeffs[3]

    def __call__(self, x):
        return vmap(self._one_dim, in_axes=(None, 0))(x, self.coeffs)


@register_pytree_node_class
class MultivariateConnectingSpline:
    def __init__(self, x, y, x_final, y_target, k=3):
        assert x.ndim == 1 and y.ndim == 2
        assert x_final.shape == (1,) and y_target.ndim == 1
        self.ts_nodes = x
        self.xs_nodes = y
        self.x_final = x_final
        self.y_target = y_target
        self.spline = MultivariateSpline(x, y, k=3)
        final_der = self.spline.derivative(x[-1].reshape(1)).reshape(-1)
        self.connection = LinearInterpolation(xs=np.stack([x[-1], x_final.reshape()]),
                                              fs=np.stack([y[-1], y_target.reshape(-1)], axis=1),
                                              fs_der=np.stack([final_der, np.zeros_like(final_der)], axis=1))

    def tree_flatten(self):
        children = (self.ts_nodes, self.xs_nodes, self.x_final, self.y_target)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __call__(self, x):
        assert x.shape == ()

        def x_too_large(x):
            # If x > ts[-1]
            return self.connection(x)

        def x_in_bounds(x):
            return self.spline(x.reshape(1, )).reshape(-1)

        return cond(x > self.ts_nodes[-1], x_too_large, x_in_bounds, x)


@register_pytree_node_class
class MultivariateSplineExt:
    def __init__(self, x, y, k=3):
        assert x.ndim == 1 and y.ndim == 2
        self.ts_nodes = x
        self.xs_nodes = y
        self.spline = MultivariateSpline(x, y, k=3)

    def tree_flatten(self):
        children = (self.ts_nodes, self.xs_nodes)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __call__(self, x):
        assert x.shape == ()

        def x_too_large(x):
            # If x > ts[-1]
            return self.xs_nodes[-1]

        def x_too_small(x):
            # If x < ts[-1]
            return self.xs_nodes[0]

        def x_in_bounds(x):
            return self.spline(x.reshape(1, )).reshape(-1)

        def too_large_test(x):
            return cond(x > self.ts_nodes[-1], x_too_large, x_in_bounds, x)

        return cond(x < self.ts_nodes[0], x_too_small, too_large_test, x)

    def derivative(self, x):
        def x_too_large(x):
            # If x > ts[-1]
            return np.zeros(shape=self.xs_nodes[0].shape)

        def x_too_small(x):
            # If x < ts[-1]
            return np.zeros(shape=self.xs_nodes[0].shape)

        def x_in_bounds(x):
            return self.spline.derivative(x.reshape(1, )).reshape(-1)

        def too_large_test(x):
            return cond(x > self.ts_nodes[-1], x_too_large, x_in_bounds, x)

        return cond(x < self.ts_nodes[0], x_too_small, too_large_test, x)


def test_spline_der():
    x = np.linspace(0, 5, 50)
    y = np.stack([np.sin(x), np.cos(x), x ** 2], axis=1)
    spline = MultivariateSpline(x, y)
    d_ys = spline.derivative(x)
    print(d_ys.shape)

    for i in range(y.shape[1]):
        plt.plot(x, y[:, i])
        plt.title('Xs')
    plt.show()

    for i in range(y.shape[1]):
        plt.plot(x, d_ys[:, i])
        plt.title('Derivative Xs')
    plt.show()


def test_spline_integral():
    x = np.linspace(0, 5, 50)
    y = x ** 2
    spline = InterpolatedUnivariateSpline(x, y)
    integral = spline.integral(0, 5)
    print(integral)


def test_jit():
    x = np.linspace(0, 5, 10)
    test_x = np.linspace(0, 5, 100)
    y = np.stack([np.sin(x), np.cos(x), x ** 2], axis=1)

    @jit
    def test(x, y, x_test):
        spline = MultivariateSpline(x, y)
        return spline(x_test)

    predictions = test(x, y, test_x)

    for i in range(y.shape[1]):
        plt.scatter(x, y[:, i])
        plt.plot(test_x, predictions[:, i])
    plt.show()


if __name__ == '__main__':
    # test_jit()
    test_spline_integral()
    # test_spline_der()
