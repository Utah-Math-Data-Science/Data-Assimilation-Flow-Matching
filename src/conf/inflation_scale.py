from hydra_orm import orm
import sqlalchemy as sa


class InflationScale(orm.InheritableTable):
    pass


class NoScaling(InflationScale):
    pass


class ConstantScale(InflationScale):
    constant: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)
