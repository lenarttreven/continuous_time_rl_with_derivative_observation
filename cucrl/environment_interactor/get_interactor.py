from cucrl.environment_interactor.mpc_interactor import MPCInteractor
from cucrl.environment_interactor.sac_interactor import SACInteractor

def get_interactor(*args):
    return SACInteractor(*args)
    # return MPCInteractor(*args)
