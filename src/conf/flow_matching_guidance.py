from typing import Any, List

import omegaconf
import hydra_orm.utils
from hydra_orm import orm
import sqlalchemy as sa

import conf.prob_path


class EnergyGuidance(orm.InheritableTable):
    use_approximate_conditional_velocity_for_unguided_velocity: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    approximate_conditional_velocity_scale_data_by_time: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)


class No(EnergyGuidance):
    pass


class MonteCarlo(EnergyGuidance):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(prob_path=omegaconf.MISSING),
        '_self_',
    ])
    sample_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    prob_path = orm.OneToManyField(conf.prob_path.ProbPath, default=omegaconf.MISSING)


class Schedule(orm.InheritableTable):
    pass


class Constant(Schedule):
    constant: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)


class Local(EnergyGuidance):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(schedule=omegaconf.MISSING),
        '_self_',
    ])
    schedule = orm.OneToManyField(Schedule, default=omegaconf.MISSING)
