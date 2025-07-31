import enum
from typing import List, Any

import omegaconf
import hydra_orm.utils
from hydra_orm import orm
import sqlalchemy as sa

from conf import diffusion_path as diff_path
from conf import inflation_scale as inflation
from conf import flow_matching_guidance


class Model(orm.InheritableTable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(inflation_scale=omegaconf.MISSING),
        '_self_',
    ])
    epoch_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    epoch_count_sampling: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    train_on_initial_predicted_state: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    train_when_ignoring_observation: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    resample_initial_predicted_state: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    ignore_observations: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    resample_predicted_state_when_ignoring_observation: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    inflation_scale = orm.OneToManyField(inflation.InflationScale, default=omegaconf.MISSING)
    use_state_perturbation: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    state_perturbation_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0)

    def __post_init_(self):
        if self.use_state_perturbation and self.state_perturbation_std == 0:
            raise ValueError('Expected model.state_perturbation_std to be greater than zero (i.e., set model.state_perturbation_std=0.1)')


class Trainable(Model):
    batch_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1000)
    shuffle_training_samples: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    learning_rate_when_training_on_initial_predicted_state: float = orm.make_field(orm.ColumnRequired(sa.Double), default=5e-3)
    learning_rate: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-2)


class Sampler(enum.Enum):
    EULER = enum.auto()
    EULER_MARUYAMA = enum.auto()
    HEUN = enum.auto()


class LNorm(enum.Enum):
    RMS = enum.auto()
    LInfty = enum.auto()


class ScoreMatching(Trainable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(diffusion_path=omegaconf.MISSING),
        dict(inflation_scale=omegaconf.MISSING),
        '_self_',
    ])
    embedding_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=50)
    residual_block_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2)
    use_batch_norm: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    sampling_time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    sampler: Sampler = orm.make_field(orm.ColumnRequired(sa.Enum(Sampler)), default=Sampler.EULER_MARUYAMA)
    sampling_max_score_norm: float = orm.make_field(orm.ColumnRequired(sa.Double), default=50.)
    sampling_score_norm: LNorm = orm.make_field(orm.ColumnRequired(sa.Enum(LNorm)), default=LNorm.RMS)

    diffusion_path: diff_path.DiffusionPath = orm.OneToManyField(diff_path.DiffusionPath, default=omegaconf.MISSING)

    def __post_init__(self):
        if self.diffusion_path != omegaconf.MISSING and not isinstance(self.diffusion_path, omegaconf.DictConfig) and not isinstance(self.diffusion_path, diff_path.VarianceExploding):
            raise ValueError(
                f'The score matching model only supports the variance exploding diffusion path, not {self.diffusion_path.__class__.__name__}.'
                ' Please set model/diffusion_path=VarianceExploding.'
            )


class ScoreMatchingMarginal(Trainable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(diffusion_path=omegaconf.MISSING),
        dict(inflation_scale=omegaconf.MISSING),
        '_self_',
    ])
    sampling_time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    sampler: Sampler = orm.make_field(orm.ColumnRequired(sa.Enum(Sampler)), default=Sampler.EULER_MARUYAMA)
    sampling_max_score_norm: float = orm.make_field(orm.ColumnRequired(sa.Double), default=50.)
    sampling_score_norm: LNorm = orm.make_field(orm.ColumnRequired(sa.Enum(LNorm)), default=LNorm.RMS)

    diffusion_path: diff_path.DiffusionPath = orm.OneToManyField(diff_path.DiffusionPath, default=omegaconf.MISSING)

    particle_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)

    def __post_init__(self):
        if self.diffusion_path != omegaconf.MISSING and isinstance(self.diffusion_path, diff_path.ConditionalOptimalTransport):
            raise ValueError(
                f'The score matching marginal model does not support {self.diffusion_path.__class__.__name__}.'
                ' Please set model/diffusion_path=VarianceExploding or model/diffusion-path=Bao2024EnsembleScoreMatching.'
            )


class FlowMatching(Trainable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(diffusion_path=omegaconf.MISSING),
        dict(inflation_scale=omegaconf.MISSING),
        dict(guidance=omegaconf.MISSING),
        '_self_',
    ])
    embedding_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=50)
    residual_block_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2)
    use_batch_norm: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    sampling_time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    sampler: Sampler = orm.make_field(orm.ColumnRequired(sa.Enum(Sampler)), default=Sampler.HEUN)
    use_expectation_of_sum: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    loss_expectation_sample_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)

    diffusion_path: diff_path.DiffusionPath = orm.OneToManyField(diff_path.DiffusionPath, default=omegaconf.MISSING)

    use_divergence_matching: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    divergence_matching_loss_coefficient: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-4)
    divergence_matching_use_hutchinson_trace_for_target_divergence: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    guidance: flow_matching_guidance.EnergyGuidance = orm.OneToManyField(flow_matching_guidance.EnergyGuidance, default=omegaconf.MISSING)

    def __post_init__(self):
        if self.guidance != omegaconf.MISSING and isinstance(self.guidance, flow_matching_guidance.MonteCarlo):
            if not isinstance(self.guidance.diffusion_path, self.diffusion_path.__class__):
                raise ValueError(
                    f'model/guidance/diffusion_path ({self.guidance.diffusion_path.__class__.__name__}) is not an instance of model/diffusion_path ({self.diffusion_path.__class__.__name__}).'
                    f' Please set model/guidance/diffusion_path={self.diffusion_path.__class__.__name__} so that the diffusion path of the guidance agrees with the diffusion path used by the flow matching model.'
                )
        if self.guidance != omegaconf.MISSING and isinstance(self.guidance, flow_matching_guidance.Local):
            if self.diffusion_path != omegaconf.MISSING and not isinstance(self.diffusion_path, (diff_path.ConditionalOptimalTransport, diff_path.PreviousPosteriorToPredictive)):
                raise ValueError(
                    'The local approximation of the flow matching guidance is only valid for affine conditional probability paths.'
                    f' Please use an affine conditional probability path (e.g., set model/diffusion_path=ConditionalOptimalTransport) instead of {self.diffusion_path.__class__.__name__}.'
                )


class FlowMatchingMarginal(Trainable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(diffusion_path=omegaconf.MISSING),
        dict(inflation_scale=omegaconf.MISSING),
        dict(guidance=omegaconf.MISSING),
        '_self_',
    ])

    sampling_time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    sampler: Sampler = orm.make_field(orm.ColumnRequired(sa.Enum(Sampler)), default=Sampler.HEUN)

    diffusion_path: diff_path.DiffusionPath = orm.OneToManyField(diff_path.DiffusionPath, default=omegaconf.MISSING)

    use_velocity_of_conditional_flow_map: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    resample_noise_when_estimating_vector_field: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    use_divergence_matching: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    divergence_matching_loss_coefficient: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-4)
    divergence_matching_use_hutchinson_trace_for_target_divergence: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    guidance: flow_matching_guidance.EnergyGuidance = orm.OneToManyField(flow_matching_guidance.EnergyGuidance, default=omegaconf.MISSING)

    def __post_init__(self):
        if self.guidance != omegaconf.MISSING and isinstance(self.guidance, flow_matching_guidance.MonteCarlo):
            if not isinstance(self.guidance.diffusion_path, self.diffusion_path.__class__):
                raise ValueError(
                    f'The model/guidance/diffusion_path ({self.guidance.diffusion_path.__class__.__name__}) is not an instance of model/diffusion_path ({self.diffusion_path.__class__.__name__}).'
                    f' Please set model/guidance/diffusion_path={self.diffusion_path.__class__.__name__} so that the diffusion path of the guidance agrees with the diffusion path used by the flow matching model.'
                )
        if self.guidance != omegaconf.MISSING and isinstance(self.guidance, flow_matching_guidance.Local):
            if self.diffusion_path != omegaconf.MISSING and not isinstance(self.diffusion_path, (diff_path.ConditionalOptimalTransport, diff_path.PreviousPosteriorToPredictive)):
                raise ValueError(
                    'The local approximation of the flow matching guidance is only valid for affine conditional probability paths.'
                    f' Please use an affine conditional probability path (e.g., set model/diffusion_path=ConditionalOptimalTransport) instead of {self.diffusion_path.__class__.__name__}.'
                )


class FlowMatchingGaussianTarget(Model):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(diffusion_path=omegaconf.MISSING),
        dict(inflation_scale=omegaconf.MISSING),
        dict(guidance=omegaconf.MISSING),
        '_self_',
    ])
    sampling_time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    sampler: Sampler = orm.make_field(orm.ColumnRequired(sa.Enum(Sampler)), default=Sampler.HEUN)

    diffusion_path: diff_path.DiffusionPath = orm.OneToManyField(diff_path.DiffusionPath, default=omegaconf.MISSING)

    guidance: flow_matching_guidance.EnergyGuidance = orm.OneToManyField(flow_matching_guidance.EnergyGuidance, default=omegaconf.MISSING)

    def __post_init__(self):
        if self.guidance != omegaconf.MISSING and isinstance(self.guidance, flow_matching_guidance.MonteCarlo):
            if not isinstance(self.guidance.diffusion_path, self.diffusion_path.__class__):
                raise ValueError(
                    f'The model/guidance/diffusion_path ({self.guidance.diffusion_path.__class__.__name__}) is not an instance of model/diffusion_path ({self.diffusion_path.__class__.__name__}).'
                    f' Please set model/guidance/diffusion_path={self.diffusion_path.__class__.__name__} so that the diffusion path of the guidance agrees with the diffusion path used by the flow matching model.'
                )
        if self.guidance != omegaconf.MISSING and isinstance(self.guidance, flow_matching_guidance.Local):
            if self.diffusion_path != omegaconf.MISSING and not isinstance(self.diffusion_path, (diff_path.ConditionalOptimalTransport, diff_path.PreviousPosteriorToPredictive)):
                raise ValueError(
                    'The local approximation of the flow matching guidance is only valid for affine conditional probability paths.'
                    f' Please use an affine conditional probability path (e.g., set model/diffusion_path=ConditionalOptimalTransport) instead of {self.diffusion_path.__class__.__name__}.'
                )


class BootstrapParticleFilter(Model):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(inflation_scale=omegaconf.MISSING),
        '_self_',
    ])


class EnsembleKalmanFilterPerturbedObservations(Model):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(inflation_scale=omegaconf.MISSING),
        '_self_',
    ])
    loc_radius_gc: float = orm.make_field(orm.ColumnRequired(sa.Double), default=5.)  # Effective radius for Gaspari-Cohn like localization


class EnsembleKalmanFilterPerturbedObservationsIterative(Model):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(inflation_scale=omegaconf.MISSING),
        '_self_',
    ])


class EnsembleRandomizedSquareRootFilter(Model):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(inflation_scale=omegaconf.MISSING),
        '_self_',
    ])
    loc_radius_gc: float = orm.make_field(orm.ColumnRequired(sa.Double), default=5.)  # Effective radius for Gaspari-Cohn like localization


class LocalEnsembleTransformKalmanFilter(Model):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(inflation_scale=omegaconf.MISSING),
        '_self_',
    ])
    loc_radius_gc: float = orm.make_field(orm.ColumnRequired(sa.Double), default=5.)  # Effective radius for Gaspari-Cohn like localization
