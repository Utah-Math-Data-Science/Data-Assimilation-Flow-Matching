import sqlalchemy as sa
from hydra_orm import orm


class DiffusionPath(orm.InheritableTable):
    pass


class ConditionalOptimalTransport(DiffusionPath):
    sigma_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)


class VarianceExploding(DiffusionPath):
    time_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)
    sigma_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)
    sigma_max: float = orm.make_field(orm.ColumnRequired(sa.Double), default=25.0)
