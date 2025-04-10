import pytest
import torch

from fixtures import init_hydra_cfg, engine

from conf import conf
from dafm import diffusion_path


def test_variance_exploding_target_distribution_at_time_0(engine):
    cfg = init_hydra_cfg('conf', ['dataset=DoubleWell', 'model=ScoreMatching', 'model/diffusion_path=VarianceExploding'])
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, cfg)
        path = diffusion_path.get_diffusion_path(cfg.model.diffusion_path, target_distribution_at_time_1=False)

    data = torch.tensor(1.)
    t_data = torch.tensor(cfg.model.diffusion_path.time_min)
    t_noise = torch.tensor(1.)
    assert path.mean(t_data, data) == data
    assert path.mean(t_noise, data) == data
    assert path.std(t_data, data) < path.std(t_noise, data)
    assert path.dt_std(t_data, data) < path.dt_std(t_noise, data)
    assert path.g(t_data) < path.g(t_noise)


def test_variance_exploding_target_distribution_at_time_1(engine):
    cfg = init_hydra_cfg('conf', ['dataset=DoubleWell', 'model=FlowMatching', 'model/diffusion_path=VarianceExploding'])
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, cfg)
        path = diffusion_path.get_diffusion_path(cfg.model.diffusion_path, target_distribution_at_time_1=True)

    data = torch.tensor(1.)
    t_noise = torch.tensor(cfg.model.diffusion_path.time_min)
    t_data = torch.tensor(1.)
    assert path.mean(t_noise, data) == data
    assert path.mean(t_data, data) == data
    assert path.std(t_data, data) < path.std(t_noise, data)
    assert -path.dt_std(t_data, data) < -path.dt_std(t_noise, data)
    assert path.g(t_data) < path.g(t_noise)


def test_conditional_optimal_transport_target_distribution_at_time_1(engine):
    cfg = init_hydra_cfg('conf', ['dataset=DoubleWell', 'model=FlowMatching', 'model/diffusion_path=ConditionalOptimalTransport'])
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, cfg)
        path = diffusion_path.get_diffusion_path(cfg.model.diffusion_path, target_distribution_at_time_1=True)

    data = torch.tensor(1.)
    t_noise = torch.tensor(0.)
    t_data = torch.tensor(1.)
    assert path.mean(t_noise, data) == t_noise * data
    assert path.mean(t_data, data) == t_data * data
    assert path.std(t_data, data) < path.std(t_noise, data)
    assert path.dt_std(t_data, data) == path.dt_std(t_noise, data)
