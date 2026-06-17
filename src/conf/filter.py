from dataclasses import field
import enum
from typing import Any, List, Optional

import omegaconf
from omegaconf import II
import hydra_orm
from hydra_orm import orm
import sqlalchemy as sa

import conf.prob_path
import conf.flow_matching_guidance


class Filter(orm.InheritableTable):
    vmap_chunk_size: int = field(default=8192)

    ensemble_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    ensemble_initial_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)

    inflation_scale: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)
    model_noise_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)


class AddNoiseToObservationFilter(Filter):
    noise_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=II('oc.select:..setting.obs_noise_std,???'))

    def __post_init__(self):
        if self.ensemble_initial_std != 0:
            raise ValueError(f'For the AddNoiseToObservationFilter, set filter.ensemble_initial_std=0, not filter.ensemble_initial_std={self.ensemble_initial_std}.')


class BootstrapParticleFilter(Filter):
    pass


class KalmanFilter(Filter):
    def __post_init__(self):
        if self.ensemble_size != 1:
            raise ValueError(f'For the Kalman filter, set filter.ensemble_size=1 (not {self.ensemble_size}) as the single ensemble member will represent the mean of the posterior.')
        if self.inflation_scale != 1:
            raise ValueError(f'Inflation is not supported by the Kalman filter, so set filter.inflation_scale=1, not {self.inflation_scale}.')


class EnsembleKalmanFilterPerturbedObservations(Filter):
    gaspari_cohn_localization_radius: Optional[float] = orm.make_field(sa.Column(sa.Double), default=omegaconf.MISSING)
    use_torch_rng_for_perturbed_observations: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)


class EnsembleKalmanFilterPerturbedObservationsIterative(Filter):
    lag: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)
    niter: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=10)
    wtol: Optional[float] = orm.make_field(sa.Column(sa.Double), default=1e-5)


class EnsembleRandomizedSquareRootFilter(Filter):
    pass


class LocalEnsembleTransformKalmanFilter(Filter):
    gaspari_cohn_localization_radius: float = orm.make_field(sa.Column(sa.Double), default=omegaconf.MISSING)


class Sampler(enum.Enum):
    EULER = enum.auto()
    EULER_MARUYAMA = enum.auto()
    HEUN = enum.auto()


class EnsembleScoreFilter(Filter):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(prob_path=omegaconf.MISSING),
        '_self_',
    ])
    sampling_time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    sampler: Sampler = orm.make_field(orm.ColumnRequired(sa.Enum(Sampler)), default=Sampler.EULER_MARUYAMA)
    sampling_max_score_norm: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)

    prob_path = orm.OneToManyField(conf.prob_path.ProbPath, default=omegaconf.MISSING)


class EnsembleFlowFilter(Filter):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(prob_path=omegaconf.MISSING),
        dict(guidance=omegaconf.MISSING),
        '_self_',
    ])
    sampling_time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    sampler: Sampler = orm.make_field(orm.ColumnRequired(sa.Enum(Sampler)), default=Sampler.EULER)

    prob_path = orm.OneToManyField(conf.prob_path.ProbPath, default=omegaconf.MISSING)
    guidance = orm.OneToManyField(conf.flow_matching_guidance.EnergyGuidance, default=omegaconf.MISSING)
