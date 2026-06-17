import numpy as np
import pytest
import torch

from conf import flow_matching_guidance, prob_path
from dafm import flow_matching_guidance as guidance
from dafm import utils


@pytest.mark.parametrize('dim', range(1, 4))
def test_multivariate_distribution(dim):
    loc = torch.zeros(dim)
    dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril=torch.eye(dim))
    dist2 = utils.Independent(utils.Normal(loc=loc, scale=torch.ones(dim)), 1)
    assert torch.allclose(dist.log_prob(loc), dist2.log_prob(loc), rtol=0, atol=1e-6)


def test_get_guidance_monte_carlo_requires_rng():
    cfg = flow_matching_guidance.MonteCarlo(
        sample_count=4,
        prob_path=prob_path.ConditionalOptimalTransport(sigma_min=0.01),
    )

    with pytest.raises(ValueError, match='requires a random number generator'):
        guidance.get_guidance(cfg)


def test_get_guidance_monte_carlo_constructs_with_rng():
    cfg = flow_matching_guidance.MonteCarlo(
        sample_count=4,
        prob_path=prob_path.ConditionalOptimalTransport(sigma_min=0.01),
    )

    guide = guidance.get_guidance(cfg, rng=np.random.default_rng(0))

    assert isinstance(guide, guidance.MonteCarlo)
