from typing import List, Any
from dataclasses import field
from pathlib import Path

import sqlalchemy as sa
import omegaconf
import hydra
import hydra_orm.utils
from hydra_orm import orm

from conf import datasets, models, diffusion_path
from dafm import utils


def get_engine(dir=str(utils.DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


class Conf(orm.Table):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(dataset=omegaconf.MISSING),
        dict(model=omegaconf.MISSING),
        '_self_',
    ])
    root_dir: str = field(default=str(utils.DIR_ROOT.resolve()))
    out_dir: str = field(default=str((utils.DIR_ROOT/'..'/'..'/'out'/'dafm').resolve()))
    run_subdir: str = field(default='runs')
    prediction_filename: str = field(default='trajectories.parquet')
    device: str = field(default='cuda')

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2376999025)
    fit: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    dataset = orm.OneToManyField(datasets.Dataset, required=True, default=omegaconf.MISSING)
    model = orm.OneToManyField(models.Model, required=True, default=omegaconf.MISSING)

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id


sa.event.listens_for(Conf, 'before_insert')(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)


orm.store_config(Conf)
orm.store_config(datasets.DoubleWell, group=Conf.dataset.key)
cs = hydra.core.config_store.ConfigStore.instance()
cs.store(group=Conf.dataset.key, name='_Lorenz63', node=datasets.Lorenz63)
orm.store_config(models.ScoreMatching, group=Conf.model.key)
orm.store_config(models.FlowMatching, group=Conf.model.key)
orm.store_config(diffusion_path.ConditionalOptimalTransport, group=f'{Conf.model.key}/{models.FlowMatching.diffusion_path.key}')
orm.store_config(diffusion_path.VarianceExploding, group=f'{Conf.model.key}/{models.FlowMatching.diffusion_path.key}')
