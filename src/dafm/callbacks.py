from collections import defaultdict

from einops import rearrange, reduce
import lightning.pytorch as pl
import pandas as pd

from dafm import utils


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    CHECKPOINT_EQUALS_CHAR = '_'


class TimeStepProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_) -> None:
        super().on_train_epoch_start(trainer)
        self.train_progress_bar.set_description(f'Time step {trainer.current_epoch}/{self.cfg.time_step_count}, training for {self.cfg.epoch_count} epochs')


class LogStats(pl.callbacks.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch, batch_idx, epoch = utils.unpack_batch(batch)
        self.log_dict(outputs, on_epoch=True, prog_bar=True, batch_size=batch['predicted_states'].shape[0])


class SaveTrajectories(pl.callbacks.Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.trajectories = defaultdict(list)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch, batch_idx, epoch = utils.unpack_batch(batch)
        self.batch_time = batch['time'][0]
        assert self.batch_time.shape == (1,)
        self.batch_observation = batch['observation'][0]
        assert self.batch_observation.shape == (1,)

    def on_train_epoch_end(self, trainer, pl_module):
        self.trajectories['time'].append(self.batch_time)
        self.trajectories['true_state'].append(pl_module.dataset.true_state)
        self.trajectories['predicted_state_mean'].append(reduce(
            pl_module.dataset.predicted_states,
            'predicted_state_count dim -> dim', 'mean'
        ))
        # for i, ps in enumerate(pl_module.dataset.predicted_states):
        #     self.trajectories[f'predicted_state_{i}'].append(ps)
        self.trajectories['observation'].append(self.batch_observation)

    def on_exception(self, trainer, pl_module, exception):
        self.on_train_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        df = pd.DataFrame({k: rearrange(v, 't dim -> (t dim)').cpu().numpy() for k, v in self.trajectories.items()})
        df.to_parquet(self.save_dir/'trajectories.parquet')
