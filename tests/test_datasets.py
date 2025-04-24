import pytest
from omegaconf import OmegaConf
import lightning.pytorch as pl

from fixtures import init_hydra_cfg, engine

from conf import conf
from dafm import models, datasets


@pytest.mark.parametrize('observe_every_n_time_steps', [
    1, 2, 3
])
def test_simple_dataset(engine, observe_every_n_time_steps):
    cfg = init_hydra_cfg('conf', [
        'dataset=_Simple',
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
        dataset = datasets.PredictedStatesAndObservation(dynamics, model)

        assert (dynamics.data['true_state'][0] == dynamics.data['predicted_state'][0]).all()
        for (epoch_count, time_step, time, next_predicted_state, next_observation, ignore_observation), _ in zip(dataset, range(5)):
            assert ignore_observation == (time_step % cfg.dataset.observe_every_n_time_steps != 0)
            assert (next_observation == dynamics.data['true_state'][time_step + 1]).all()
            print(time)
