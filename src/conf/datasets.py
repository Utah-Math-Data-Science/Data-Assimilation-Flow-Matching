from dataclasses import field

from hydra_orm import orm
import sqlalchemy as sa

import conf.observe


class Dataset(orm.InheritableTable):
    predicted_state_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1000)
    time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=100)
    time_step_count_drop_first: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    time_step_size: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.1)
    observe_every_n_time_steps: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)
    observe: conf.observe.Observe = orm.OneToManyField(conf.observe.Observe, default_factory=conf.observe.Full)


class DoubleWell(Dataset):
    state_dimension: int = field(init=False, default=1)
    model_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.2)
    observation_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.1)
    true_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.02)
    predicted_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.2)
    predicted_state_model_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)


class Lorenz63(Dataset):
    state_dimension: int = field(init=False, default=3)
    model_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.2)
    observation_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.1)
    true_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.02)
    predicted_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.2)
    predicted_state_model_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)

    sigma: float = orm.make_field(orm.ColumnRequired(sa.Double), default=28.)
    rho: float = orm.make_field(orm.ColumnRequired(sa.Double), default=10.)
    beta: float = orm.make_field(orm.ColumnRequired(sa.Double), default=8/3)
    rescaling: float = orm.make_field(orm.ColumnRequired(sa.Double), default=20.)


class Lorenz96(Dataset):
    model_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.1)
    observation_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.5)
    true_state_initial_condition_mean: float = orm.make_field(orm.ColumnRequired(sa.Double), default=4.)
    true_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=2.)
    predicted_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.2)
    predicted_state_model_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)

    state_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=10)
    forcing: float = orm.make_field(orm.ColumnRequired(sa.Double), default=8.)

    def __post_init__(self):
        if self.state_dimension < 3:
            raise ValueError(f"The dimension of Lorenz '96 must be at least 4, not {self.state_dimension}")
