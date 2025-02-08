import logging
import pprint
import sys

import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import lightning.pytorch as pl
from pytorch_lightning.utilities import CombinedLoader

from conf import conf
from dafm import callbacks, datasets, loggers, models, utils


log = logging.getLogger(__file__)


class DataAssimilation(pl.LightningModule):
    def __init__(self, cfg, dataset, model):
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg
        self.dataset = dataset
        self.dataset_iterable = None
        self.model = model

    def configure_optimizers(self):
        return None

    def setup(self, stage):
        if stage == 'fit':
            self.dataset_iterable = iter(self.dataset)

    def train_dataloader(self):
        time_step, time, predicted_states, observation = next(self.dataset_iterable)
        self.optimizer = self.model.get_optimizer(time_step)
        return CombinedLoader({
            epoch: iter(CombinedLoader(dict(
                    time_step=DataLoader([time_step]),
                    time=DataLoader([time]),
                    predicted_states=DataLoader(predicted_states, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle_training_samples),
                    observation=DataLoader([observation]),
            ), mode='max_size_cycle'))
            for epoch in range(self.cfg.epoch_count)
        }, mode='sequential')

    def training_step(self, batch, _):
        batch, batch_idx, epoch = utils.unpack_batch(batch)
        observation = None if batch['time_step'] == 0 else batch['observation']
        self.optimizer.zero_grad()
        loss = self.model.loss(batch['predicted_states'], observation)
        self.manual_backward(loss)
        self.optimizer.step()
        return dict(loss=loss)


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = conf.get_engine()
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        pprint.pp(cfg)
        log.info('Command: python %s', ' '.join(sys.argv))
        log.info('Output directory: %s', cfg.run_dir)

    logger = loggers.CSVLogger(cfg.run_dir, name=None)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.epoch_count,
        check_val_every_n_epoch=cfg.epoch_count,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        callbacks=[
            callbacks.TimeStepProgressBar(cfg),
            callbacks.LogStats(),
            callbacks.SaveTrajectories(cfg.run_dir),
        ],
    )

    model = models.Score(1, 50, 2, False)
    dataset = datasets.PredictedStatesAndObservation(cfg, model)
    data_assimilation = DataAssimilation(cfg, dataset, model)

    trainer.fit(data_assimilation)


def get_run_dir(hydra_init=utils.HYDRA_INIT, commit=True):
    if '-m' in sys.argv:
        raise ValueError("The flag '-m' is not supported. Use GNU parallel instead.")
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = conf.get_engine()
        conf.orm.create_all(engine)
        with conf.sa.orm.Session(engine, expire_on_commit=False) as db:
            cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            if commit and '-c' not in sys.argv:
                db.commit()
                cfg.run_dir.mkdir(exist_ok=True)
            return last_override, str(cfg.run_dir)


if __name__ == '__main__':
    last_override, run_dir = get_run_dir()
    run_dir_override = f'hydra.run.dir={run_dir}'
    if last_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(last_override + 1, run_dir_override)
    main()
