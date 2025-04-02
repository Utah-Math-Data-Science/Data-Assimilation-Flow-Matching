from hydra_orm import orm
import sqlalchemy as sa


class EnergyGuidance(orm.InheritableTable):
    pass


class No(EnergyGuidance):
    pass


class MonteCarlo(EnergyGuidance):
    sample_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1024)
    time_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)


class Local(EnergyGuidance):
    pass
