import pytest
import torch
from omegaconf import OmegaConf
import lightning.pytorch as pl
from einops import rearrange

from fixtures import init_hydra_cfg, engine

from conf import conf
from dafm import models, datasets


@pytest.mark.parametrize('observe_every_n_time_steps', [
    1, 2, 3
])
def test_simple_dataset(engine, observe_every_n_time_steps):
    cfg = init_hydra_cfg('conf', [
        'dataset=_Simple',
        'dataset.predicted_state_count=2',
        'dataset.time_step_count=250',
        f'dataset.observe_every_n_time_steps={observe_every_n_time_steps}',
        'model=ScoreMatchingMarginalBao2024EnSF',
        'model.epoch_count=1',
        'model.epoch_count_sampling=0',
        'model.sampling_time_step_count=2',
   ])
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, cfg)
        pl.seed_everything(cfg.rng_seed)
        with pl.utilities.seed.isolate_rng():
            dynamics = datasets.get_dynamics_dataset(cfg.dataset, cfg.device)
        with pl.utilities.seed.isolate_rng():
            model = models.get_model(cfg.model, cfg.dataset.state_dimension, 0.1)

        def data_to_save_callback(time_step, data_to_save):
            for k, v in info.items():
                v.extend(data_to_save[k])

        dataset = datasets.PredictedStatesAndObservation(dynamics, model, data_to_save_callback=data_to_save_callback)

        info = dict(
            time_step=[],
            times=[], predicted_state=[],
        )

        assert (dynamics.true_state[0] == dynamics.current_predicted_state).all()
        for epoch_count, time_step, time, next_predicted_state, next_observation, ignore_observation in dataset:
            assert ignore_observation == (time_step % cfg.dataset.observe_every_n_time_steps != 0)
            assert (next_observation == dynamics.true_state[time_step + 1]).all()
        assert (torch.arange(cfg.dataset.time_step_count + 1) == torch.tensor(info['time_step'])).all()
