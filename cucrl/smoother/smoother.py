from typing import Any

from cucrl.main.config import SmootherConfig
from cucrl.smoother.abstract_smoother import AbstractSmoother
from cucrl.smoother.fSVGD_smoother import FSVGDEnsemble
from cucrl.smoother.fSVGD_time_only import FSVGDTimeOnly
from cucrl.smoother.gp_time_only import GPTimeOnly
from cucrl.utils.representatives import SmootherType

pytree = Any


class SmootherFactory:

    def __init__(self):
        pass

    @staticmethod
    def make_smoother(smoother_type: SmootherType, smoother_config: SmootherConfig, state_dim: int, noise_stds,
                      normalizer) -> AbstractSmoother:
        if smoother_type == SmootherType.FSVGD:
            return FSVGDEnsemble(state_dim=state_dim, num_members=smoother_config.num_particles,
                                 features=smoother_config.features, noise_stds=noise_stds, normalizer=normalizer,
                                 prior_h=smoother_config.bandwidth_prior)
        elif smoother_type == SmootherType.GP_TIME_ONLY:
            return GPTimeOnly(state_dim=state_dim, noise_stds=noise_stds, normalizer=normalizer)
        elif smoother_type == SmootherType.FSVGD_TIME_ONLY:
            return FSVGDTimeOnly(state_dim=state_dim, num_particles=smoother_config.num_particles,
                                 features=smoother_config.features, noise_stds=noise_stds,
                                 bandwidth_svgd=smoother_config.bandwidth_svgd, normalizer=normalizer,
                                 bandwidth_prior=smoother_config.bandwidth_prior)

        else:
            raise Exception('SmootherType not supported.')
