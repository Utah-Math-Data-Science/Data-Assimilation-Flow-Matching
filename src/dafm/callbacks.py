import logging

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

    def _get_current_time_step_of_estimation(self, trainer):
        if self.cfg.model.train_on_initial_predicted_state:
            return trainer.current_epoch
        else:
            return trainer.current_epoch + 1

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_) -> None:
        super().on_train_epoch_start(trainer)
        time_step_to_estimate = self._get_current_time_step_of_estimation(trainer)
        if self.cfg.model.train_on_initial_predicted_state and trainer.current_epoch == 0:
            estimation_message = f'Training on initial predicted state without observation at time step {time_step_to_estimate}/{self.cfg.dataset.time_step_count}'
        else:
            ignore_observations_text = ' without observation' if self.cfg.model.ignore_observations else ''
            estimation_message = f'Estimating state{ignore_observations_text} for time step {time_step_to_estimate}/{self.cfg.dataset.time_step_count}'
        self.train_progress_bar.set_description(f'{estimation_message}, training for {self.cfg.model.epoch_count} epochs')


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
        data['predicted_state_mean'] = reduce(
            data['predicted_state'],
            't predicted_state_count dim -> t dim', 'mean'
        )
        del data['predicted_state']
        df = pd.concat([
            pd.Series(rearrange(v, 't dim -> (t dim)').cpu().numpy(), name=k)
            for k, v in data.items()
        ], axis=1)
        df.to_parquet(self.save_path)
        log.info('Trajectory data saved to %s', self.save_path)
