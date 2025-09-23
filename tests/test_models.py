import pytest
import torch
import numpy as np
from omegaconf import OmegaConf
import lightning.pytorch as pl
from einops import rearrange

from fixtures import init_hydra_cfg, engine

from conf import conf
from dafm import models, datasets, utils


@pytest.mark.parametrize('model_kind', [
    'BootstrapParticleFilter',
    'EnsembleKalmanFilterPerturbedObservations',
    'EnsembleRandomizedSquareRootFilter',
    'LocalEnsembleTransformKalmanFilter',
])
def test_simple_dataset(engine, model_kind):
    cfg = init_hydra_cfg('conf', [
        'dataset=_Simple',
        'dataset.predicted_state_count=2',
        'dataset.time_step_count=2',
        f'model={model_kind}',
        'model.epoch_count=1',
        'model.epoch_count_sampling=0',
   ])
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, cfg)
        rng = np.random.default_rng(utils.RNG_RANDBITS[cfg.rng_seed])
        dynamics = datasets.get_dynamics_dataset(cfg.dataset, rng, cfg.device, delete_true_state=True)
        pl.seed_everything(cfg.rng_seed)
        with pl.utilities.seed.isolate_rng():
            model = models.get_model(cfg.model, cfg.dataset.state_dimension, 0.1, dynamics)

        dataset = datasets.PredictedStatesAndObservation(dynamics, model)

        for epoch_count, time_step, time, next_predicted_state, next_observation, ignore_observation in dataset:
            pass
