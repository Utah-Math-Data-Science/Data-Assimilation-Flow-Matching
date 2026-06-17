import logging
import os
import sys
import tempfile
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributions
from torch.distributions.utils import _sum_rightmost

def filename_relative_to_dir_root(filename):
    return Path(filename).relative_to(DIR_ROOT)


def getLoggerByFilename(filename):
    return logging.getLogger(str(filename_relative_to_dir_root(filename)))


DIR_ROOT = (Path(__file__).parent/'..'/'..').resolve()
HYDRA_INIT = dict(version_base=None, config_path='../../conf', config_name='conf')

# generated using f'0x{secrets.randbits(128):x}'
RNG_RANDBITS = {
    # train
    2376999025: dict(
        DATASET=0x54c4a70db3a25f0f09d635fecc999819,
        DATA_ASSIMILATION=0x5c4cf7e963e4b87af50b9138af7c34e5,
        FILTER=0x9f05b6e9d819168892b980c2692360a3,
        OBSERVE=0xca4ae75a9bf325a1ad6fc0d87f1e368b,
    ),
    # test
    462133975: dict(
        DATASET=0x90a3b090dfce4bb5451312e279a1ad9b,
        DATA_ASSIMILATION=0x9aca2372ecb6b20f71462a273b5f653d,
        FILTER=0x7f1c9b7470189c6523f5a21027d8c026,
        OBSERVE=0x9e15dfb1f395ca6c43bcbbe84d08f5d,
    ),
    979497033: dict(
        DATASET=0x673c3a09866605264b85cf2a02dbd867,
        DATA_ASSIMILATION=0xaf65c9ce04b6a0f2a0d174046cb47658,
        FILTER=0x5060eef21df51346048d92a7994c7fbd,
        OBSERVE=0x59cc7d788fcc7b9547879f265fbaf8,
    ),
    97616566: dict(
        DATASET=0x67bd0bf02f6a8cc5b8c4fdf9efbec14b,
        DATA_ASSIMILATION=0x8299e876751c8375c9d324b5c391f8e,
        FILTER=0xa46772daf141c26700925da938e84045,
        OBSERVE=0xeb7e5b92483522fc71d9d84131245ca,
    ),
    715319214: dict(
        DATASET=0x7b8e3af10c73db657611f8b65fa89212,
        DATA_ASSIMILATION=0x831ca330ee71440f56bd51dba31e2409,
        FILTER=0xbf32ea54a1a6760edcb2257e5ada55fb,
        OBSERVE=0x96a9a948f10e938ef23a5df4acba9b0,
    ),
    19704671: dict(
        DATASET=0x314b39e6858250fc639a668a9be3f62f,
        DATA_ASSIMILATION=0x67a54ccb8b42d4fc2698ead1e0dc4f58,
        FILTER=0x44224e2fa5e6bfe9e6b3c1ea912ec7e3,
        OBSERVE=0x3e38566218eb794634d5ac599c999f1,
    )
}

_randbits = set()
for v in RNG_RANDBITS.values():
    for rb in v.values():
        if rb in _randbits:
            raise ValueError(f'RNG_RANDBITS duplicated: {rb}')
        else:
            _randbits.add(rb)
del _randbits


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


def torch_save_once_atomic(obj, save_path: Path):
    import fcntl

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f'{save_path}.lock')

    with lock_path.open('w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if save_path.exists():
            return False

        fd, tmp_path = tempfile.mkstemp(prefix=f'{save_path.name}.', dir=save_path.parent)
        try:
            with os.fdopen(fd, 'wb') as tmp_file:
                torch.save(obj, tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, save_path)
            return True
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def get_torch_rng_generator(base_seed: int, generators: dict, device: torch.device) -> torch.Generator:
    device = torch.device(device)
    device_index = -1 if device.index is None else device.index
    key = (device.type, device_index)
    if key not in generators:
        seed_sequence = np.random.SeedSequence([
            base_seed,
            0 if device.type == 'cpu' else 1,
            device_index,
        ])
        seed_parts = seed_sequence.generate_state(2, dtype=np.uint32)
        seed = (int(seed_parts[0]) << 32) | int(seed_parts[1])
        generator_device = 'cpu' if device.type == 'cpu' else str(device)
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed)
        generators[key] = generator
    return generators[key]
