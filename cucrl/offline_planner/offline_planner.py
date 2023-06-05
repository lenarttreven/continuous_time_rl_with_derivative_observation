from cucrl.offline_planner.offline_planner_eta_time import EtaTimeOfflinePlanner
from cucrl.utils.representatives import ExplorationStrategy


def get_offline_planner(exploration_strategy: ExplorationStrategy):
    match exploration_strategy:
        case ExplorationStrategy.OPTIMISTIC_ETA_TIME:
            return EtaTimeOfflinePlanner
        # elif exploration_strategy == ExplorationStrategy.OPTIMISTIC_ETA:
        #     return EtaOfflinePlanner
        # elif exploration_strategy == ExplorationStrategy.OPTIMISTIC_GP:
        #     return GPOfflinePlanner
        # elif exploration_strategy == ExplorationStrategy.MEAN:
        #     return MeanOfflinePlanner
        case _:
            raise NotImplementedError("This strategy has not been implemented yet.")
