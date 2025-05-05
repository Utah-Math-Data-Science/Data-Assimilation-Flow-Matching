from pathlib import Path

import torch.distributions
from torch.distributions.utils import _sum_rightmost


DIR_ROOT = (Path(__file__).parent/'..'/'..').resolve()
HYDRA_INIT = dict(version_base=None, config_path='../../conf', config_name='conf')


def unpack_batch(batch):
    batch, _, epoch = batch
    batch, batch_idx, _ = batch
    return batch, batch_idx, epoch


def inner_product(a, b):
    return (a * b).reshape(a.shape[0], -1).sum(-1, keepdim=True)


class Independent(torch.distributions.Independent):
    def log_prob_unnormalized(self, value):
        log_prob_unnormalized = self.base_dist.log_prob_unnormalized(value)
        return _sum_rightmost(log_prob_unnormalized, self.reinterpreted_batch_ndims)


class Normal(torch.distributions.Normal):
    def log_prob_unnormalized(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = self.scale**2
        return -((value - self.loc) ** 2) / (2 * var)
