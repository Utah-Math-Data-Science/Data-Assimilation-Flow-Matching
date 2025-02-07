from dataclasses import field
from pathlib import Path

import sqlalchemy as sa
import hydra_orm.utils
from hydra_orm import orm

from dafm import utils


def get_engine(dir=str(utils.DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


class Conf(orm.Table):
    root_dir: str = field(default=str(utils.DIR_ROOT.resolve()))
    out_dir: str = field(default=str((utils.DIR_ROOT/'..'/'..'/'out'/'dafm').resolve()))
    run_subdir: str = field(default='runs')
    prediction_filename: str = field(default='prediction.pt')
    device: str = field(default='cuda')

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=42)
    fit: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=100)
    epoch_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=600)
    batch_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1000)
    shuffle_training_samples: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id


sa.event.listens_for(Conf, 'before_insert')(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)


orm.store_config(Conf)
