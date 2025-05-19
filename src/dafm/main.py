import logging
import os
import pprint
import sys

from einops import reduce
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
        epoch_count, time_step, time, next_predicted_state, next_observation, ignore_observation = next(self.dataset_iterable)
        self.optimizer = self.model.get_optimizer(time_step, ignore_observation)
        return CombinedLoader({
            epoch: iter(CombinedLoader(dict(
                    time_step=DataLoader([time_step]),
                    time=DataLoader([time]),
                    next_predicted_state=DataLoader(next_predicted_state, batch_size=self.cfg.model.batch_size, shuffle=self.cfg.model.shuffle_training_samples),
                    next_observation=DataLoader([next_observation]),
                    ignore_observation=DataLoader([ignore_observation]),
            ), mode='max_size_cycle'))
            for epoch in range(epoch_count)
        }, mode='sequential')

    def training_step(self, batch, _):
        batch, batch_idx, epoch = utils.unpack_batch(batch)
        self.optimizer.zero_grad()
        next_observation = batch['next_observation'] if not batch['ignore_observation'] else None
        losses = self.model.loss(batch['next_predicted_state'], next_observation, self.dataset.dataset.observe)
        self.manual_backward(losses['loss'])
        self.optimizer.step()
        return losses


@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = conf.get_engine()
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        log.info('Command: python %s', ' '.join(sys.argv))
        log.info(pprint.pformat(cfg))
        log.info('Output directory: %s', cfg.run_dir)

    pl.seed_everything(cfg.rng_seed)
    with pl.utilities.seed.isolate_rng():
        dynamics = datasets.get_dynamics_dataset(cfg.dataset, cfg.device)
    with pl.utilities.seed.isolate_rng():
        model = models.get_model(cfg.model, cfg.dataset.state_dimension, cfg.dataset.observation_noise_std)

    time_step_time_logger = loggers.CSVLogger(cfg.run_dir, name=None, name_metrics_file='time_step_times.csv')

    dataset = datasets.PredictedStatesAndObservation(
        dynamics, model,
        logger=time_step_time_logger,
        data_to_save_callback=lambda time_step, data_to_save: datasets.save_trajectories(
            cfg.dataset, data_to_save,
            cfg.run_dir/f'{cfg.prediction_filename}.{time_step}.parquet'
        )
    )
    data_assimilation = DataAssimilation(cfg, dataset, model)

    logger = loggers.CSVLogger(cfg.run_dir, name=None)

    cbs = [
        callbacks.LogStats(),
        # callbacks.SaveTrajectories(cfg.run_dir/cfg.prediction_filename),
    ]
    enable_progress_bar = False
    if cfg.model.epoch_count > 0:# or cfg.model.epoch_count_sampling > 0:
        enable_progress_bar = True
        cbs.append(callbacks.TimeStepProgressBar(cfg))
    trainer = pl.Trainer(
        # detect_anomaly=True,
        enable_progress_bar=enable_progress_bar,
        accelerator=cfg.device,
        devices=1,
        logger=logger,
        max_epochs=-1,
        check_val_every_n_epoch=None,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        callbacks=cbs,
    )

    try:
        trainer.fit(data_assimilation)
    except StopIteration as e:
        if cfg.model.epoch_count == 0 and cfg.model.epoch_count_sampling == 0:
            pass
        else:
            raise e


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
            if commit:
            # if commit and '-c' not in sys.argv:
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
