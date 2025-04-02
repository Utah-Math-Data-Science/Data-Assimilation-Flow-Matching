import pytest
import torch


@pytest.mark.parametrize('dim', range(1, 4))
def test_multivariate_distribution(dim):
    loc = torch.zeros(dim)
    dist = torch.distributions.MultivariateNormal(loc=loc, scale_tril=torch.eye(dim))
    dist2 = torch.distributions.Independent(torch.distributions.Normal(loc=loc, scale=torch.ones(dim)), 1)
    assert torch.allclose(dist.log_prob(loc), dist2.log_prob(loc), rtol=0, atol=1e-6)
