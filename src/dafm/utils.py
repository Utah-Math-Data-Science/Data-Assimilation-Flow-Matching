import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch.distributions
from torch.distributions.utils import _sum_rightmost

from conf import conf


DIR_ROOT = (Path(__file__).parent/'..'/'..').resolve()
HYDRA_INIT = dict(version_base=None, config_path='../../conf', config_name='conf')

# generated using f'0x{secrets.randbits(128):x}'
RNG_RANDBITS = {
    # train
    2376999025: 0x43e2a09d8b8e89c269435eb906c9890f,
    # test
    462133975: 0x9ce12aa97c67b0e330eba6143aeb4c01,
    979497033: 0xd1af79b87b0b0c062cc49d79f053f541,
    97616566: 0xabba188e774c6f1a69bff838fc031252,
    715319214: 0x704e81235a98055f9902925f54f4f3cf,
    19704671: 0xf117601173413e80e52730bbc35a9a30,
}


def get_run_dir(hydra_init=HYDRA_INIT, commit=True, engine_name='runs'):
    if '-m' in sys.argv or '--multirun' in sys.argv:
        raise ValueError("The flags '-m' and '--multirun' are not supported. Use GNU parallel instead.")
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = conf.get_engine(name=engine_name)
        conf.orm.create_all(engine)
        with conf.sa.orm.Session(engine, expire_on_commit=False) as db:
            cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            # if commit and '-c' not in sys.argv:
            if commit:
                db.commit()
                cfg.run_dir.mkdir(exist_ok=True)
            return last_override, str(cfg.run_dir)


def set_run_dir(last_override, run_dir):
    run_dir_override = f'hydra.run.dir={run_dir}'
    if last_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(last_override + 1, run_dir_override)


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
