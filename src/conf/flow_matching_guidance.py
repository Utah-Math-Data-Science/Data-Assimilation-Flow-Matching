from typing import Any, List

import omegaconf
import hydra_orm.utils
from hydra_orm import orm
import sqlalchemy as sa


class EnergyGuidance(orm.InheritableTable):
    pass


class No(EnergyGuidance):
    pass


class MonteCarlo(EnergyGuidance):
    sample_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1024)
    time_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)


class Schedule(orm.InheritableTable):
    pass


class Constant(Schedule):
    constant: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1)


class Local(EnergyGuidance):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(schedule=omegaconf.MISSING),
        '_self_',
    ])
    schedule = orm.OneToManyField(Schedule, default=omegaconf.MISSING)
