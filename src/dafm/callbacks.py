import logging
import math

import torch
from einops import rearrange, reduce
import lightning.pytorch as pl
import pandas as pd
import polars

from dafm import utils


log = logging.getLogger(__file__)


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    CHECKPOINT_EQUALS_CHAR = '_'


class TimeStepProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop('v_num', None)
        return items

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module) -> None:
        super().on_train_epoch_start(trainer)
        time_step = pl_module.dataset.time_step
        if self.cfg.model.train_on_initial_predicted_state and time_step == 0 and trainer.current_epoch == 0:
            estimation_message = f'Training on initial predicted state without observation at time step {time_step}/{self.cfg.dataset.time_step_count - self.cfg.dataset.time_step_count_drop_first}'
        else:
            ignore_observations_text = ' without observation' if self.cfg.model.ignore_observations else ''
            estimation_message = f'Training{ignore_observations_text} to estimate state at time step {time_step + 1}/{self.cfg.dataset.time_step_count - self.cfg.dataset.time_step_count_drop_first}'
        self.train_progress_bar.set_description(f'{estimation_message}. Epochs={self.cfg.model.epoch_count}, Batches={math.ceil(self.cfg.dataset.predicted_state_count / self.cfg.model.batch_size)}')


class LogStats(pl.callbacks.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch, batch_idx, epoch = utils.unpack_batch(batch)
        self.log_dict(outputs, on_epoch=True, prog_bar=True, batch_size=batch['next_predicted_state'].shape[0])


class SaveTrajectories(pl.callbacks.Callback):
    def __init__(self, save_dir, prediction_filename):
        self.save_dir = save_dir
        self.prediction_filename = prediction_filename

    def on_train_epoch_start(self, trainer, pl_module):
        self.log('predicted_state_mean', reduce(
            pl_module.dataset.dataset.data['predicted_state'][-1],
            'predicted_state_count dim ->', 'mean'
        ), on_epoch=True, prog_bar=True)

    def on_exception(self, trainer, pl_module, exception):
        self.on_train_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        df = polars.scan_parquet(self.save_dir/f'{self.prediction_filename}.*.parquet')
        df.write_parquet(self.save_dir)
        log.info('Trajectory data saved to %s', self.save_dir)
