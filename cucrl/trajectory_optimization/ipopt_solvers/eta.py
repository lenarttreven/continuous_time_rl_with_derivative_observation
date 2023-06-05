import jax.numpy as jnp
from jax import random, jit
from jax.tree_util import register_pytree_node_class

from cucrl.utils.classes import DynamicsModel


@register_pytree_node_class
class EtaSolver:
    def __init__(
        self,
        objective,
        gradient,
        eq_constrains,
        eq_jacobian,
        ineq_simulator,
        ineq_simulator_jac,
        ineq_eta,
        ineq_eta_jac,
    ):
        self._objective = objective
        self._gradient = gradient
        self._eq_eta = eq_constrains
        self._eq_eta_jac = eq_jacobian
        self._ineq_simulator = ineq_simulator
        self._ineq_simulator_jac = ineq_simulator_jac
        self._ineq_eta = ineq_eta
        self._ineq_eta_jac = ineq_eta_jac
        self.jac_discovery_key = random.PRNGKey(0)
        self.dynamics_model: DynamicsModel | None = None
        self.initial_conditions = None
        self.params_for_opt = None

    def tree_flatten(self):
        children = (
            self.dynamics_model,
            self.initial_conditions,
            self.params_for_opt,
            self.jac_discovery_key,
            self.indices,
        )
        aux_data = {
            "objective": self._objective,
            "gradient": self._gradient,
            "eq_constrains": self._eq_eta,
            "eq_jacobian": self._eq_eta_jac,
            "ineq_simulator": self._ineq_simulator,
            "ineq_simulator_jac": self._ineq_simulator_jac,
            "ineq_eta": self._ineq_eta,
            "ineq_eta_jac": self._ineq_eta_jac,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            dynamics_model,
            initial_conditions,
            params_for_opt,
            jac_discovery_key,
            indices,
        ) = children
        new_class = cls(
            objective=aux_data["objective"],
            gradient=aux_data["gradient"],
            eq_constrains=aux_data["eq_constrains"],
            eq_jacobian=aux_data["eq_jacobian"],
            ineq_simulator=aux_data["ineq_simulator"],
            ineq_simulator_jac=aux_data["ineq_simulator_jac"],
            ineq_eta=aux_data["ineq_eta"],
            ineq_eta_jac=aux_data["ineq_eta_jac"],
        )
        new_class.initial_conditions = initial_conditions
        new_class.params_for_opt = params_for_opt
        new_class.dynamics_model = dynamics_model
        new_class.jac_discovery_key = jac_discovery_key
        new_class.indices = indices
        return new_class

    def update_args(self, dynamics_model, initial_conditions, params_for_opt):
        self.dynamics_model = dynamics_model
        self.initial_conditions = initial_conditions
        self.params_for_opt = params_for_opt
        self._prepare_jacobian_structure()
        self.cl, self.cu = self.prepare_lower_bound()

    def _prepare_jacobian_structure(self):
        self.jac_discovery_key, subkey = random.split(self.jac_discovery_key)
        random_params_for_opt = random.normal(
            key=subkey, shape=self.params_for_opt.shape
        )
        J = self._full_jacobian(random_params_for_opt)
        self.indices = jnp.nonzero(J)
        full_num_jac = jnp.prod(jnp.array(J.shape))
        sparse_num_jac = self.indices[0].size
        self.improvement = 1 - sparse_num_jac / full_num_jac
        print("Sparsity improvement: ", self.improvement)

    def prepare_lower_bound(self):
        num_eq_cons = self._eq_eta(
            self.params_for_opt, self.initial_conditions, self.dynamics_model
        ).size
        num_ineq_cons_sim = self._ineq_simulator(self.params_for_opt).size
        num_ineq_cons_eta = self._ineq_eta(
            self.params_for_opt, self.dynamics_model.beta
        ).size
        cl = jnp.zeros(shape=(num_eq_cons + num_ineq_cons_sim + num_ineq_cons_eta,))
        cu = jnp.concatenate(
            [
                jnp.zeros(shape=(num_eq_cons,)),
                2.0e19 * jnp.ones(shape=(num_ineq_cons_sim + num_ineq_cons_eta,)),
            ]
        )
        return cl, cu

    def jacobianstructure(self):
        return self.indices

    def objective(self, x):
        return self._objective(x)

    def gradient(self, x):
        return self._gradient(x)

    def constraints(self, x):
        equality = self._eq_eta(x, self.initial_conditions, self.dynamics_model)
        inequality_simulator = self._ineq_simulator(x)
        inequality_eta = self._ineq_eta(x, self.dynamics_model.beta)
        return jnp.concatenate([equality, inequality_simulator, inequality_eta])

    def _full_jacobian(self, x):
        equality = self._eq_eta_jac(x, self.initial_conditions, self.dynamics_model)
        inequality_simulator = self._ineq_simulator_jac(x)
        inequality_eta = self._ineq_eta_jac(x, self.dynamics_model.beta)
        return jnp.concatenate([equality, inequality_simulator, inequality_eta])

    @jit
    def jacobian(self, x):
        J = self._full_jacobian(x)
        row, col = self.jacobianstructure()
        return J[row, col]
