import pytest
import torch
import numpy as np

import conf.prob_path
from dafm import prob_path


def test_variance_exploding_target_distribution_at_time_0():
    cfg = conf.prob_path.VarianceExploding()
    path = prob_path.get_prob_path(cfg, np.random.default_rng(0), target_distribution_at_time_1=False)

    data = torch.tensor(1.)
    t_data = torch.tensor(cfg.time_min)
    t_noise = torch.tensor(1.)
    assert path.mean(t_data, data) == data
    assert path.mean(t_noise, data) == data
    assert path.dt_mean(t_data, data) == 0
    assert path.dt_mean(t_noise, data) == 0
    assert path.std(t_data, data) < path.std(t_noise, data)
    assert path.dt_std(t_data, data) < path.dt_std(t_noise, data)
    assert path.g(t_data) < path.g(t_noise)


def test_variance_exploding_target_distribution_at_time_1():
    cfg = conf.prob_path.VarianceExploding()
    path = prob_path.get_prob_path(cfg, np.random.default_rng(0), target_distribution_at_time_1=True)

    data = torch.tensor(1.)
    t_noise = torch.tensor(cfg.time_min)
    t_data = torch.tensor(1.)
    assert path.mean(t_noise, data) == data
    assert path.mean(t_data, data) == data
    assert path.dt_mean(t_data, data) == 0
    assert path.dt_mean(t_noise, data) == 0
    assert path.std(t_data, data) < path.std(t_noise, data)
    assert -path.dt_std(t_data, data) < -path.dt_std(t_noise, data)
    assert path.g(t_data) < path.g(t_noise)


def test_conditional_optimal_transport_target_distribution_at_time_1():
    cfg = conf.prob_path.ConditionalOptimalTransport(sigma_min=0.001)
    path = prob_path.get_prob_path(cfg, np.random.default_rng(0), target_distribution_at_time_1=True)

    data = torch.tensor(1.)
    t_noise = torch.tensor(0.)
    t_data = torch.tensor(1.)
    assert path.mean(t_noise, data) == t_noise * data
    assert path.mean(t_data, data) == t_data * data
    assert path.dt_mean(t_data, data) == data
    assert path.dt_mean(t_noise, data) == data
    assert path.std(t_data, data) < path.std(t_noise, data)
    assert path.dt_std(t_data, data) == path.dt_std(t_noise, data)


def test_bao_2024_ensemble_score_matching():
    cfg = conf.prob_path.Bao2024EnsembleScoreMatching()
    path = prob_path.get_prob_path(cfg, np.random.default_rng(0), target_distribution_at_time_1=False)

    data = torch.tensor(1.)
    t_noise = torch.tensor(0.)
    t_data = torch.tensor(1.)
    assert path.alpha(0) == 1.
    assert path.alpha(1) == cfg.epsilon_alpha
    assert path.beta(0)**2 == cfg.epsilon_beta
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


def test_bao_2024_ensemble_score_matching_renormalize_sampled_noise():
    cfg = conf.prob_path.Bao2024EnsembleScoreMatching(renormalize_sampled_noise=True)
    path = prob_path.get_prob_path(cfg, np.random.default_rng(0), target_distribution_at_time_1=False)

    data = torch.ones(2)
    t_noise = torch.tensor(0.)
    noise = path.sample_noise(t_noise, data)
    assert (noise.mean(0) < 1e-7).all()
    assert (noise.std(0) == 1.).all()
