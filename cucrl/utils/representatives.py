from enum import Enum, auto


class Space(Enum):
    STATE = auto()
    DERIVATIVE = auto()


class BetaType(Enum):
    GP = auto()
    BNN = auto()


class Optimizer(Enum):
    ADAM = auto()
    LBFGS = auto()
    SGD = auto()


class Norm(Enum):
    L_2 = auto()
    L_INF = auto()


class BatchStrategy(Enum):
    MAX_DETERMINANT_GREEDY = auto()
    MAX_KERNEL_DISTANCE_GREEDY = auto()
    EQUIDISTANT = auto()


class BNNTypes(Enum):
    FSVGD = auto()
    DETERMINISTIC_ENSEMBLE = auto()


class TimeHorizonType(Enum):
    ADAPTIVE_TRUE = auto()
    FIXED = auto()


class Dynamics(Enum):
    FSVGD_PENDULUM = auto()
    FSVGD_LV = auto()
    FSVGD_VAN_DER_POOL = auto()
    FSVGD_MOUNTAIN_CART = auto()
    FSVGD_CARTPOLE = auto()
    FSVGD_BICYCLE = auto()
    FSVGD_FURUTA_PENDULUM = auto()
    FSVGD_QUADROTOR_EULER = auto()
    FSVGD_GENERAL_AFFINE = auto()
    GP = auto()
    BNN = auto()


class SimulatorType(Enum):
    LOTKA_VOLTERRA = auto()
    LORENZ = auto()
    ACROBOT = auto()
    LINEAR = auto()
    PENDULUM = auto()
    VAN_DER_POL_OSCILATOR = auto()
    MOUNTAIN_CAR = auto()
    CARTPOLE = auto()
    BICYCLE = auto()
    FURUTA_PENUDLUM = auto()
    SWIMMER_MUJOCO = auto()
    QUADROTOR_QUATERNIONS = auto()
    QUADROTOR_EULER = auto()
    QUADROTOR_2D = auto()
    RACE_CAR = auto()
    CANCER_TREATMENT = auto()
    GLUCOSE = auto()


class MinimizationMethod(Enum):
    IPOPT = auto()
    ILQR = auto()
    ILQR_WITH_CEM = auto()


class Statistics(Enum):
    MEDIAN = auto()
    MEAN = auto()


class SmootherType(Enum):
    FSVGD = auto()
    FSVGD_TIME_ONLY = auto()
    GP_TIME_ONLY = auto()


class DynamicsTracking(Enum):
    BEST = auto()
    MEAN = auto()


class ExplorationNorm(Enum):
    L_2 = auto()
    L_INFINITY = auto()


class ExplorationStrategy(Enum):
    MEAN = auto()
    OPTIMISTIC_PARTICLES = auto()
    OPTIMISTIC_ETA = auto()
    OPTIMISTIC_ETA_TIME = auto()
    OPTIMISTIC_ETA_TIME_MEAN = auto()
    OPTIMISTIC_GP = auto()
    THOMPSON_SAMPLING = auto()


class NumericalComputation(Enum):
    LGL = auto()
    SPLINES = auto()
    CLASSIC = auto()
    LOCAL_SPLINES = auto()
    LOCAL_POINT_SPLINES = auto()
