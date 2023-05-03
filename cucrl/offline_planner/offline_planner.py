from cucrl.offline_planner.mean_planner import MeanOfflinePlanner
from cucrl.offline_planner.offline_planner_eta import EtaOfflinePlanner
from cucrl.offline_planner.offline_planner_eta_time import EtaTimeOfflinePlanner
from cucrl.offline_planner.offline_planner_gp import GPOfflinePlanner
from cucrl.utils.representatives import ExplorationStrategy


def get_offline_planner(exploration_strategy: ExplorationStrategy):
    if exploration_strategy == ExplorationStrategy.OPTIMISTIC_ETA:
        return EtaOfflinePlanner
    elif exploration_strategy == ExplorationStrategy.OPTIMISTIC_GP:
        return GPOfflinePlanner
    elif exploration_strategy == ExplorationStrategy.OPTIMISTIC_ETA_TIME:
        return EtaTimeOfflinePlanner
    elif exploration_strategy == ExplorationStrategy.MEAN:
        return MeanOfflinePlanner
    else:
        raise NotImplementedError('This strategy has not been implemented yet.')
