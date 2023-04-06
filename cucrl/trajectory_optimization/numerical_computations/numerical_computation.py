from cucrl.trajectory_optimization.numerical_computations.classic import Classic
from cucrl.trajectory_optimization.numerical_computations.legendre_gauss_lobatto import LegendreGaussLobatto
from cucrl.trajectory_optimization.numerical_computations.local_point_splines import LocalPointSplines
from cucrl.trajectory_optimization.numerical_computations.local_splines import LocalSplines
from cucrl.trajectory_optimization.numerical_computations.splines import Splines
from cucrl.utils.representatives import NumericalComputation


def get_numerical_computation(numerical_computation: NumericalComputation, num_nodes, time_horizon):
    if numerical_computation == NumericalComputation.LGL:
        return LegendreGaussLobatto(num_nodes=num_nodes, time_horizon=time_horizon)
    elif numerical_computation == NumericalComputation.SPLINES:
        return Splines(num_nodes=num_nodes, time_horizon=time_horizon)
    elif numerical_computation == NumericalComputation.CLASSIC:
        return Classic(num_nodes=num_nodes, time_horizon=time_horizon)
    elif numerical_computation == NumericalComputation.LOCAL_SPLINES:
        return LocalSplines(num_nodes=num_nodes, time_horizon=time_horizon)
    elif numerical_computation == NumericalComputation.LOCAL_POINT_SPLINES:
        return LocalPointSplines(num_nodes=num_nodes, time_horizon=time_horizon)
