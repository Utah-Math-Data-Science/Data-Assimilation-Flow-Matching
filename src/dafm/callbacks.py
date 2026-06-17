from functools import cache
import math

import numpy as np
import torch
from einops import rearrange, reduce
import lightning.pytorch as pl

import conf.conf
import conf.filter
import dafm.utils


log = dafm.utils.getLoggerByFilename(__file__)


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


class LogDAStats(pl.callbacks.Callback):
    def __init__(self, cfg: conf.conf.Conf):
        super().__init__()
        self.cfg = cfg

    def on_validation_start(self, trainer, pl_module):
        self.should_save_observations = False
        self.observations_save_path = self.cfg.setting.cache_dir/f'{self.cfg.setting.id}_observations.pt'
        if self.cfg.setting.reference_filter is not None:
            self.reference_filter_mean = torch.load(self.cfg.setting.reference_filter.run_dir/'ensemble_mean.pt', weights_only=True)
        if self.cfg.save_ensemble_stats:
            if not self.observations_save_path.exists():
                self.should_save_observations = True
                self.observations = []
            self.ensemble_stats = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        ensemble = outputs['filtering_ensemble']
        true_state = outputs['true_state']
        true_state_norm = reduce(true_state.square(), '1 dim ->', 'mean').sqrt().clamp(min=1e-5)
        metrics = dict(
            time_step=torch.tensor(float(outputs['time_step'])),
            rmse=self.RMSE(ensemble, true_state),
            rmv=self.RMV(ensemble, true_state),
            da_time_s=outputs['da_time_s'],
        )
        metrics['nrmse'] = metrics['rmse'] / true_state_norm
        if ensemble.shape[0] < 1000000:
            metrics['energy_score'] = self.energy_score(ensemble, true_state, vmap_chunk_size=1024)
        if self.cfg.setting.reference_filter is not None:
            reference_mean = self.reference_filter_mean[batch_idx].to(device=ensemble.device)
            metrics['rmse_from_reference_filter'] = self.RMSE(ensemble, reference_mean)
            metrics['nrmse_from_reference_filter'] = metrics['rmse_from_reference_filter'] / true_state_norm
            metrics['rmv_from_reference_filter'] = self.RMV(ensemble, reference_mean)
            if ensemble.shape[0] < 1000000:
                metrics['energy_score_from_reference_filter'] = self.energy_score(ensemble, reference_mean, vmap_chunk_size=1024)
        pl_module.log_dict(metrics, batch_size=1, on_step=True, on_epoch=False, prog_bar=True)
        if self.should_save_observations:
            self.observations.append(outputs['observation'].squeeze(0).cpu())
        if self.cfg.save_ensemble_stats:
            ensemble_stats = dict(
                time_step=outputs['time_step'],
                mean=ensemble.mean(0).cpu(),
            )
            if ensemble.shape[-1] <= 3:
                if isinstance(pl_module.cfg.filter, conf.filter.KalmanFilter):
                    ensemble_stats['covariance'] = pl_module.filter.covariance.cpu()
                else:
                    ensemble_stats['covariance'] = ensemble.mT.cov().cpu()
                if ensemble.shape[0] <= 50:
                    ensemble_stats['all'] = ensemble.cpu()
            self.ensemble_stats.append(ensemble_stats)

    def on_validation_end(self, trainer, pl_module):
        if self.cfg.save_ensemble_stats:
            if len(self.ensemble_stats) == 0:
                log.warning('No ensemble stats saved.')
                return
            keys = self.ensemble_stats[0].keys()
            for k in keys:
                torch.save(
                    rearrange([d[k] for d in self.ensemble_stats], 'time_step ... -> time_step ...'),
                    self.cfg.run_dir/f'ensemble_{k}.pt',
                )
            if self.should_save_observations:
                dafm.utils.torch_save_once_atomic(rearrange(self.observations, 'time_step ... -> time_step ...'), self.observations_save_path)

    def on_exception(self, trainer, pl_module, exception):
        self.on_validation_end(trainer, pl_module)

    @staticmethod
    def RMSE(ensemble, true_state):
        ensemble_mean = reduce(ensemble, 'member dim -> 1 dim', 'mean')
        return reduce((ensemble_mean - true_state).square(), '1 dim ->', 'mean').sqrt()

    @staticmethod
    def RMV(ensemble, true_state):
        """
        Variance of the ensemble in each dimension, averaged and square-rooted.
        """
        return ensemble.var(0).mean().sqrt()

    @classmethod
    def energy_score(cls, ensemble, true_state, vmap_chunk_size=None):
        """
        Reduces to the Continuous Ranked Probability Score when the system has
        one dimension.
        """
        mean_r_from_true = reduce(
            (ensemble - true_state).square(),
            'member dim -> member',
            'sum',
        ).sqrt().mean(0)

        mean_r_between_members = (torch.vmap(
            lambda x: reduce((ensemble - x).square(), 'member dim -> member', 'sum').sqrt().sum(),
            chunk_size=vmap_chunk_size,
        )(ensemble)).sum() / ensemble.shape[0]**2

        # ensemble_size = ensemble.shape[0]
        # member_a, member_b = cls._energy_score_member_pairs(ensemble_size, dtype=torch.long, device=ensemble.device)
        # half_mean_r_between_members = reduce(
        #     (ensemble[member_a] - ensemble[member_b]).square(),
        #     'member dim -> member',
        #     'sum',
        # ).sqrt().sum(0) / ensemble_size**2

        return mean_r_from_true - .5 * mean_r_between_members

    @staticmethod
    @cache
    def _energy_score_member_pairs(ensemble_size, **kwargs):
        """
        Returns an edge list for an adjacency matrix with ones strictly below
        the diagonal.
        """
        member_idx = torch.arange(ensemble_size, **kwargs)
        member_a = member_idx.repeat_interleave(ensemble_size)
        member_b = member_idx.repeat(ensemble_size)
        self_loop_or_symmetric_edge = member_a >= member_b
        member_a = member_a[~self_loop_or_symmetric_edge]
        member_b = member_b[~self_loop_or_symmetric_edge]
        return member_a, member_b


class DAProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, cfg: conf.conf.Conf):
        self.cfg = cfg
        # theme = pl.callbacks.progress.rich_progress.RichProgressBarTheme(
        #     description=f'Assimilation steps (dt_obs={cfg.setting.observe_every_n_time_steps * cfg.setting.dataset.time_step_size:.2f})'
        # )
        # super().__init__(theme=theme)
        super().__init__()

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description(f'Assimilation steps (dt_obs={self.cfg.setting.observe_every_n_time_steps * self.cfg.setting.dataset.time_step_size:.2f})')
        return bar
