import os
from pathlib import Path
import typing_extensions

import lightning.pytorch as pl


class CSVLogger(pl.loggers.CSVLogger):
    def __init__(
        self,
        save_dir,
        name="lightning_logs",
        version=None,
        prefix="",
        flush_logs_every_n_steps=100,
        name_metrics_file='metrics.csv',
    ):
        super().__init__(
            save_dir=save_dir,
            name=name,
            version=version,
            prefix=prefix,
            flush_logs_every_n_steps=flush_logs_every_n_steps,
        )
        self.experiment.NAME_METRICS_FILE = name_metrics_file
        self.experiment.metrics_file_path = os.path.join(self.experiment.log_dir, self.experiment.NAME_METRICS_FILE)
        Path(self.experiment.metrics_file_path).unlink(missing_ok=True)

    @property
    @typing_extensions.override
    def log_dir(self) -> str:
        return self.root_dir
