import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.config import config

from cucrl.optimal_cost.optimal_cost_ilqr import OptimalCost
from cucrl.simulator.simulator_costs import HIVTreatment as HIVTreatmentCosts
from cucrl.simulator.simulator_dynamics import HIVTreatment

config.update('jax_enable_x64', True)


def run():
    time_horizon = (0, 20.0)

    time_scaling = jnp.ones(shape=(1,))
    state_scaling = jnp.diag(jnp.array([1., 1e2, 1e2]))
    control_scaling = jnp.eye(1)

    system = HIVTreatment(time_scaling=time_scaling, state_scaling=state_scaling, control_scaling=control_scaling)
    cost = HIVTreatmentCosts(time_scaling=time_scaling, state_scaling=state_scaling, control_scaling=control_scaling)
    initial_state = jnp.array([800., .04, 1.5])
    num_nodes = 200
    optimizer = OptimalCost(simulator_dynamics=system, simulator_costs=cost, num_nodes=num_nodes,
                            time_horizon=time_horizon)

    out = optimizer.solve(initial_state)
    print('We completed the test successfully')
    ts = jnp.linspace(time_horizon[0], time_horizon[1], num_nodes + 1)
    xs = out.xs
    us = out.us
    plt.plot(ts, xs, label='xs')
    plt.plot(ts[:-1], us, label='us')
    plt.legend()
    plt.show()
    print('Optimal cost: ', out.obj)
    return out


if __name__ == '__main__':
    out = run()
