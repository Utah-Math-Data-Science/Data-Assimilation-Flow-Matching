import omegaconf
from omegaconf import OmegaConf, II
from hydra_orm import orm
import sqlalchemy as sa


OmegaConf.register_new_resolver('add', lambda x, y: x + y)
OmegaConf.register_new_resolver('sub', lambda x, y: x - y)
OmegaConf.register_new_resolver('mul', lambda x, y: x * y)
OmegaConf.register_new_resolver('div', lambda x, y: x / y)


class Splitter(orm.InheritableTable):
    pass


class StartAndLen(Splitter):
    start_train: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)  # default to 1 to drop initial condition
    len_train: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    start_val: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=II('add:${.start_train},${.len_train}'))
    len_val: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    start_test: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=II('add:${.start_val},${.len_val}'))
    len_test: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
