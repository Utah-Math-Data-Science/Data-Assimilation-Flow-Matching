import enum

from typing import List, Any
from dataclasses import field

import omegaconf
import hydra_orm
from hydra_orm import orm
import sqlalchemy as sa

import conf.observe


class Integrator(enum.Enum):
    RUNGE_KUTTA_4 = enum.auto()
    EULER_MARUYAMA = enum.auto()


class StatePerturbation(enum.Enum):
    IDENTITY = enum.auto()
    BAO_ET_AL_DOUBLE_WELL = enum.auto()
    BAO_ET_AL_LORENZ_96 = enum.auto()


class Dataset(orm.InheritableTable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(observe='Full'),
        '_self_',
    ])
    predicted_state_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1000)
    time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=100)
    time_step_count_drop_first: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    time_step_size: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.1)
    observe_every_n_time_steps: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)
    observe: conf.observe.Observe = orm.OneToManyField(conf.observe.Observe, default=omegaconf.MISSING)
    integrator: Integrator = orm.make_field(orm.ColumnRequired(sa.Enum(Integrator)), default=Integrator.EULER_MARUYAMA)
    state_perturbation: StatePerturbation = orm.make_field(orm.ColumnRequired(sa.Enum(StatePerturbation)), default=StatePerturbation.IDENTITY)


class DoubleWell(Dataset):
    state_dimension: int = field(default=1)
    model_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.2)
    observation_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.1)
    true_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.02)
    predicted_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.2)
    predicted_state_model_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)


class Lorenz63(Dataset):
    state_dimension: int = field(default=3)
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
    predicted_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.5)
    predicted_state_model_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)

    state_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=10)
    forcing: float = orm.make_field(orm.ColumnRequired(sa.Double), default=8.)

    def __post_init__(self):
        if self.state_dimension < 3:
            raise ValueError(f"The dimension of Lorenz '96 must be at least 4, not {self.state_dimension}")
