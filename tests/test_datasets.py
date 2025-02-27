import pytest
from omegaconf import OmegaConf
import lightning.pytorch as pl

from fixtures import init_hydra_cfg, engine

from conf import conf
from dafm import datasets, datasets_v1


@pytest.mark.parametrize('overrides', [
    ['dataset=DoubleWell'],
    ['dataset=Lorenz63'],
    ['dataset=Lorenz96'],
])
def test_datasets_equals_datasets_v1(engine, overrides):
    cfg = init_hydra_cfg('conf', ['model=ScoreMatching', 'model/diffusion_path=VarianceExploding', *overrides])
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg))
        pl.seed_everything(cfg.rng_seed)
        with pl.utilities.seed.isolate_rng():
            ds = datasets.get_dynamics_dataset(cfg.dataset, cfg.device)
        with pl.utilities.seed.isolate_rng():
            ds_v1 = datasets_v1.get_dynamics_dataset(cfg.dataset, cfg.device)

        for k in ds.data:
            if isinstance(ds.data[k], list):
                assert (ds.data[k][0] == ds_v1.data[k][0]).all(), k
            else:
                assert (ds.data[k] == ds_v1.data[k]).all(), k
