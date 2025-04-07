import pytest
from omegaconf import OmegaConf
import lightning.pytorch as pl

from fixtures import init_hydra_cfg, engine

from conf import conf
from dafm import datasets, datasets_v1
