from dataclasses import field
import enum
from typing import List, Any

import omegaconf
import hydra_orm.utils
from hydra_orm import orm
import sqlalchemy as sa

from conf import diffusion_path as diff_path


class Model(orm.InheritableTable):
    ignore_observations: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)


class Trainable(Model):
    epoch_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    batch_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1000)
    shuffle_training_samples: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    embedding_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=50)
    residual_block_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2)
    use_batch_norm: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    train_on_initial_predicted_state: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    resample_initial_predicted_state: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)


class Sampler(enum.Enum):
    EULER = enum.auto()
    EULER_MARUYAMA = enum.auto()
    HEUN = enum.auto()


class ScoreMatching(Trainable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(diffusion_path=omegaconf.MISSING),
        '_self_',
    ])

    sampling_time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    sampling_max_score_norm: float = orm.make_field(orm.ColumnRequired(sa.Double), default=50.)
    sampler: Sampler = orm.make_field(orm.ColumnRequired(sa.Enum(Sampler)), default=Sampler.EULER_MARUYAMA)

    diffusion_path: diff_path.DiffusionPath = orm.OneToManyField(diff_path.DiffusionPath, default=omegaconf.MISSING)

    def __post_init__(self):
        if self.diffusion_path != omegaconf.MISSING and not isinstance(self.diffusion_path, diff_path.VarianceExploding):
            raise ValueError(
                f'The score matching model only supports the variance exploding diffusion path, not {self.diffusion_path.__class__.__name__}.'
                ' Please set model/diffusion_path=VarianceExploding.'
            )



class FlowMatching(Trainable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(diffusion_path=omegaconf.MISSING),
        '_self_',
    ])

    sampling_time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    sampler: Sampler = orm.make_field(orm.ColumnRequired(sa.Enum(Sampler)), default=Sampler.HEUN)
    loss_sample_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)
    softmax_loss_weighting: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    sampling_use_observation_likelihood: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    diffusion_path: diff_path.DiffusionPath = orm.OneToManyField(diff_path.DiffusionPath, default=omegaconf.MISSING)
