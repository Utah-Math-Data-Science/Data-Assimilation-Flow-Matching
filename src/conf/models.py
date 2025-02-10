import enum

from hydra_orm import orm
import sqlalchemy as sa


class Model(orm.InheritableTable):
    ignore_observations: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)


class Trainable(Model):
    epoch_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    batch_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1000)
    shuffle_training_samples: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    embedding_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=50)
    residual_block_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2)
    use_batch_norm: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    train_on_initial_predicted_state: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    resample_initial_predicted_state: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)


class Sampler(enum.Enum):
    EULER = enum.auto()
    EULER_MARUYAMA = enum.auto()
    HEUN = enum.auto()


class ScoreMatching(Trainable):
    time_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)
    sigma_max: float = orm.make_field(orm.ColumnRequired(sa.Double), default=25.0)
    sampling_time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    sampler: Sampler = orm.make_field(orm.ColumnRequired(sa.Enum(Sampler)), default=Sampler.EULER_MARUYAMA)


class FlowMatching(Trainable):
    sampling_time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    sampler: Sampler = orm.make_field(orm.ColumnRequired(sa.Enum(Sampler)), default=Sampler.HEUN)
    loss_sample_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)
