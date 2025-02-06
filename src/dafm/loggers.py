import typing_extensions

import lightning.pytorch as pl


class CSVLogger(pl.loggers.CSVLogger):
    @property
    @typing_extensions.override
    def log_dir(self) -> str:
        return self.root_dir
