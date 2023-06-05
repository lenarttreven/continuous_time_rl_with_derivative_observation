import jax.numpy as jnp
from cyipopt import minimize_ipopt
from jax import random, grad, jit

system_params = jnp.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
(J, M, m_a, m_p, l_a, l_p) = system_params

g = 0.2

true_alpha = J + (M + 1 / 3 * m_a + m_p) * l_a**2
true_beta = (M + 1 / 3 * m_p) * l_p**2
true_gamma = (M + 1 / 2 * m_p) * l_a * l_p
true_delta = (M + 1 / 2 * m_p) * g * l_p

print(true_alpha, true_beta, true_gamma, true_delta)


@jit
def tune_params(system_params):
    (J, M, m_a, m_p, l_a, l_p) = system_params
    alpha = J + (M + 1 / 3 * m_a + m_p) * l_a**2
    beta = (M + 1 / 3 * m_p) * l_p**2
    gamma = (M + 1 / 2 * m_p) * l_a * l_p
    delta = (M + 1 / 2 * m_p) * 9.81 * l_p
    return (
        (alpha - true_alpha) ** 2
        + (beta - true_beta) ** 2
        + (gamma - true_gamma) ** 2
        + (delta - true_delta) ** 2
    )


init_key = random.PRNGKey(0)
init_params = random.uniform(init_key, shape=(6,))

bnds = [(0, 20) for _ in range(6)]
out = minimize_ipopt(
    tune_params,
    init_params,
    jac=jit(grad(tune_params)),
    bounds=bnds,
    options={"maxiter": 10000},
)
print(out)
