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
    assert path.dt_mean(t_data, data) == 0
    assert path.dt_mean(t_noise, data) == 0
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
    assert path.dt_mean(t_data, data) == 0
    assert path.dt_mean(t_noise, data) == 0
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
    assert path.dt_mean(t_data, data) == data
    assert path.dt_mean(t_noise, data) == data
    assert path.std(t_data, data) < path.std(t_noise, data)
    assert path.dt_std(t_data, data) == path.dt_std(t_noise, data)


def test_bao_2024_ensemble_score_matching(engine):
    cfg = init_hydra_cfg('conf', ['dataset=DoubleWell', 'model=ScoreMatchingMarginalBao2024EnSF'])
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, cfg)
        path = diffusion_path.get_diffusion_path(cfg.model.diffusion_path, target_distribution_at_time_1=False)

    data = torch.tensor(1.)
    t_noise = torch.tensor(0.)
    t_data = torch.tensor(1.)
    assert path.alpha(0) == 1.
    assert path.alpha(1) == cfg.model.diffusion_path.epsilon_alpha
    assert path.beta(0)**2 == cfg.model.diffusion_path.epsilon_beta
    assert path.beta(1)**2 == 1.
    with torch.enable_grad():
        for t in (t_noise, t_data, (t_noise + t_data) / 2):
            t = t.clone().requires_grad_()
            dt_log_alpha, *_ = torch.autograd.grad(
                outputs=path.alpha(t).log(),
                inputs=t,
            )
            assert dt_log_alpha == path.dt_log_alpha(t)
            dt_squared_beta, *_ = torch.autograd.grad(
                outputs=path.beta(t).square(),
                inputs=t,
            )
            assert dt_squared_beta == path.dt_squared_beta(t)


def test_bao_2024_ensemble_score_matching_renormalize_sampled_noise(engine):
    cfg = init_hydra_cfg('conf', ['dataset=DoubleWell', 'model=ScoreMatchingMarginalBao2024EnSF', 'model.diffusion_path.renormalize_sampled_noise=true'])
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, cfg)
        path = diffusion_path.get_diffusion_path(cfg.model.diffusion_path, target_distribution_at_time_1=False)

    data = torch.ones(2)
    t_noise = torch.tensor(0.)
    noise = path.sample_noise(t_noise, data)
    assert (noise.mean(0) < 1e-7).all()
    assert (noise.std(0) == 1.).all()
