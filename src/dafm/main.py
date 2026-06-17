import pprint
import sys
import time

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
import lightning.pytorch as pl
from einops import rearrange, reduce

import conf.conf
import conf.prob_path
import dafm.callbacks
import dafm.datasets
import dafm.filters
import dafm.loggers
import dafm.observe
import dafm.utils

log = dafm.utils.getLoggerByFilename(__file__)


class ObservationCollector(IterableDataset):
    def __init__(self, cfg, rng: np.random.Generator, device, dataset, observe):
        super().__init__()
        self.cfg = cfg
        (
            self.rng_initial_condition,
            self.rng_obs_noise,
        ) = rng.spawn(2)
        self.device = device
        self.dataset = dataset
        self.observe = observe

    def __len__(self):
        return len(self.cfg.setting.observation_time_steps)

    @torch.no_grad()
    def __iter__(self):
        true_initial_condition = self.dataset[getattr(self.cfg.setting.splitter, f'start_{self.cfg.setting.split}')]
        if self.cfg.setting.ensemble_initial_mean_is_true_state:
            loc = true_initial_condition
        else:
            loc = torch.zeros_like(true_initial_condition)
        self.filtering_ensemble = torch.tensor(self.rng_initial_condition.normal(
            loc=loc.numpy(), scale=self.cfg.filter.ensemble_initial_std,
            size=(self.cfg.filter.ensemble_size, *true_initial_condition.shape),
        ), device=self.device, dtype=true_initial_condition.dtype)
        previous_time_step = getattr(self.cfg.setting.splitter, f'start_{self.cfg.setting.split}')
        for time_step in self.cfg.setting.observation_time_steps:
            predictive_ensemble = self.dataset.integrate(previous_time_step, time_step, self.filtering_ensemble)[-1]
            data = self.dataset[time_step][None].to(self.device)
            observation = self.observe(data)
            observation = observation + self.observe.get_sparsity_mask(device=self.device) * torch.tensor(self.rng_obs_noise.normal(
                scale=self.cfg.setting.obs_noise_std, size=observation.shape
            ), device=self.device, dtype=observation.dtype)

            yield dict(
                time_step=time_step,
                gt=data,
                predictive_ensemble=predictive_ensemble,
                observation=observation,
            )

            previous_time_step = time_step


class DataAssimilation(pl.LightningModule):
    def __init__(self, cfg, dataset, filter):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.filter = filter

    def prepare_data(self):
        self.dataset.dataset.prepare_data()

    def setup(self, stage):
        self.dataset.dataset.prepare_data()
        self.dataset.dataset.setup('fit')

    def val_dataloader(self):
        return DataLoader(self.dataset)

    def validation_step(self, data, batch_idx):
        ensemble = data['predictive_ensemble'][0]
        ensemble_shape = ensemble.shape
        observation = data['observation'][0]

        if hasattr(self.cfg.filter, 'prob_path') and isinstance(self.cfg.filter.prob_path, conf.prob_path.FilteringToPredictive):
            self.filter.prob_path.set_previous_filtering(rearrange(self.dataset.filtering_ensemble, 'member ... -> member (...)'))

        """Data assimilation timing start"""
        log_da_start = time.process_time()

        self.filter.initialize(self.cfg.setting, rearrange(self.dataset.filtering_ensemble, 'member ... -> member (...)'), self.dataset.dataset)

        ensemble = self.filter.assimilate(
            data['time_step'],
            rearrange(ensemble, 'member ... -> member (...)'),
            rearrange(observation, '1 ... -> 1 (...)'),
            self.cfg.setting.obs_noise_std,
            dafm.observe.Unflatten(self.dataset.observe),
            # lambda e: rearrange(
            #     self.dataset.observe(e.view(ensemble_shape)),
            #     'member ... -> member (...)'
            # ),
        )

        log_da_end = time.process_time()
        """Data assimilation timing end"""

        self.dataset.filtering_ensemble = ensemble.view(ensemble_shape)
        return dict(
            time_step=data['time_step'],
            filtering_ensemble=ensemble,
            true_state=rearrange(data['gt'], '1 ... -> 1 (...)'),
            observation=observation,
            da_time_s=log_da_end - log_da_start,
        )


hydra_init = dafm.utils.HYDRA_INIT


def run(cfg):
    with conf.conf.Session() as db:
        cfg: conf.conf.Conf = conf.conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        cmd_args = sys.argv
        if cmd_args[-1].startswith('hydra.run.dir='):
            cmd_args = cmd_args[:-1]
        log.info('Command: python %s', ' '.join(cmd_args))
        log.info(pprint.pformat(cfg))
        log.info('Output directory: %s', cfg.run_dir)

    dataset = dafm.datasets.get_dataset(
        cfg.setting.dataset,
        np.random.default_rng(dafm.utils.RNG_RANDBITS[cfg.setting.dataset.rng_seed]['DATASET']),
        cfg.device,
    )

    observe = dafm.observe.build_observe(cfg.setting.observes, cfg_dataset=cfg.setting.dataset, rng=np.random.default_rng(dafm.utils.RNG_RANDBITS[cfg.rng_seed]['OBSERVE']))

    da_dataset = ObservationCollector(cfg, np.random.default_rng(dafm.utils.RNG_RANDBITS[cfg.rng_seed]['DATA_ASSIMILATION']), cfg.device, dataset, observe)
    da_filter = dafm.filters.get_filter(cfg.filter, cfg_dataset=cfg.setting.dataset, rng=np.random.default_rng(dafm.utils.RNG_RANDBITS[cfg.rng_seed]['FILTER']))
    data_assimilation = DataAssimilation(cfg, da_dataset, da_filter)

    logger = dafm.loggers.CSVLogger(cfg.run_dir, name=None)

    callbacks = [
        dafm.callbacks.DAProgressBar(cfg),
        dafm.callbacks.LogDAStats(cfg),
    ]
    trainer = pl.Trainer(
        accelerator=cfg.device,
        devices=1,
        logger=logger,
        check_val_every_n_epoch=None,
        deterministic=not isinstance(cfg.filter, conf.filter.BootstrapParticleFilter),
        callbacks=callbacks,
        log_every_n_steps=1,
        inference_mode=False,
    )

    trainer.validate(data_assimilation)


@hydra.main(**hydra_init)
def main(cfg):
    run(cfg)


if __name__ == '__main__':
    last_override, run_dir = conf.conf.get_run_dir(hydra_init=hydra_init)
    conf.conf.set_run_dir(last_override, run_dir)
    main()
