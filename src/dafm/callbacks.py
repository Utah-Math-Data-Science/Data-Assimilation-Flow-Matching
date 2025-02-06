import lightning.pytorch as pl


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    CHECKPOINT_EQUALS_CHAR = '_'


class TimeStepProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_) -> None:
        super().on_train_epoch_start(trainer)
        self.train_progress_bar.set_description(f'Time step {trainer.current_epoch}/{self.cfg.time_step_count}, training for {self.cfg.epoch_count} epochs')
