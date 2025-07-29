from hydra_orm import orm
import sqlalchemy as sa


class Observe(orm.InheritableTable):
    pass


class Full(Observe):
    pass


class EveryNthDimension(Observe):
    start_at_zero: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    n: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)


class Exponentiate(Observe):
    exponent: float = orm.make_field(orm.ColumnRequired(sa.Double), default=3.)


class ATan(Observe):
    pass


class ATanEveryNthDimension(Observe):
    start_at_zero: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    n: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)
