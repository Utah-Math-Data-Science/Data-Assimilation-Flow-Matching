from pathlib import Path


DIR_ROOT = (Path(__file__).parent/'..'/'..').resolve()
HYDRA_INIT = dict(version_base=None, config_path='../../conf', config_name='conf')


def unpack_batch(batch):
    batch, _, epoch = batch
    batch, batch_idx, _ = batch
    return batch, batch_idx, epoch
