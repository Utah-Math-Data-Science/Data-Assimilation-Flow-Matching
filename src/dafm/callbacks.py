import logging
import math

import torch
from einops import rearrange, reduce
import lightning.pytorch as pl
import pandas as pd

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
    def __init__(self, save_path):
        self.save_path = save_path

    def on_train_epoch_start(self, trainer, pl_module):
        self.log('predicted_state_mean', reduce(
            pl_module.dataset.dataset.data['predicted_state'][-1],
            'predicted_state_count dim ->', 'mean'
        ), on_epoch=True, prog_bar=True)

    def on_exception(self, trainer, pl_module, exception):
        self.on_train_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        data = pl_module.dataset.dataset.data.copy()
        data['predicted_state'][-1] = data['predicted_state'][-1].to('cpu')
        data['predicted_state'] = rearrange(
            data['predicted_state'],
            't predicted_state_count dim -> t predicted_state_count dim'
        )
        time_step_count = data['times'].shape[0]
        data['times'] = pd.DataFrame(
            data['times'].cpu().numpy(),
            index=range(time_step_count),
            columns=['times'],
        )
        time_step_count_predicted, predicted_state_count, dim = data['predicted_state'].shape
        if pl_module.dataset.dataset.cfg.save_only_mean_std:
            data['predicted_state_mean'] = reduce(
                data['predicted_state'],
                't predicted_state_count dim -> t dim',
                'mean',
            )
            data['predicted_state_std'] = reduce(
                data['predicted_state'],
                't predicted_state_count dim -> t dim',
                torch.std,
            )
            for stat in ('mean', 'std'):
                data[f'predicted_state_{stat}'] = pd.DataFrame(
                    data[f'predicted_state_{stat}'].cpu().numpy(),
                    index=range(time_step_count_predicted),
                    columns=[
                        f'predicted_state_{stat}_dim_{d}'
                        for d in range(dim)
                    ],
                )
            df = pd.concat([data[k] for k in ('times', 'predicted_state_mean', 'predicted_state_std')], axis=1)
        else:
            data['true_state'] = rearrange(
                data['true_state'],
                't 1 dim -> t dim',
            )
            data['observation'] = rearrange(
                data['observation'],
                't 1 dim -> t dim',
            )
            data['predicted_state'] = rearrange(
                data['predicted_state'],
                't predicted_state_count dim -> t (predicted_state_count dim)'
            )
            data['predicted_state'] = pd.DataFrame(
                data['predicted_state'].cpu().numpy(),
                index=range(time_step_count_predicted),
                columns=[
                    f'predicted_state_{state}_dim_{d}'
                    for state in range(predicted_state_count)
                    for d in range(dim)
                ],
            )
            data['true_state'] = pd.DataFrame(
                data['true_state'].cpu().numpy(),
                index=range(time_step_count),
                columns=[f'true_state_dim_{d}' for d in range(dim)],
            )
            data['observation'] = pd.DataFrame(
                data['observation'].cpu().numpy(),
                index=range(time_step_count),
                columns=[f'observation_dim_{d}' for d in range(data['observation'].shape[1])],
            )
            df = pd.concat([data[k] for k in ('times', 'true_state', 'observation', 'predicted_state')], axis=1)
        df.to_parquet(self.save_path)
        log.info('Trajectory data saved to %s', self.save_path)
