from typing import Any, List

import omegaconf
import hydra_orm.utils
from hydra_orm import orm
import sqlalchemy as sa

import conf.diffusion_path


class EnergyGuidance(orm.InheritableTable):
    use_approximate_conditional_velocity_for_unguided_velocity: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    approximate_conditional_velocity_scale_data_by_time: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)


class No(EnergyGuidance):
    pass


class MonteCarlo(EnergyGuidance):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(diffusion_path=omegaconf.MISSING),
        '_self_',
    ])
    sample_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1024)
    diffusion_path = orm.OneToManyField(conf.diffusion_path.DiffusionPath, default=omegaconf.MISSING)


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
