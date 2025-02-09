from collections import defaultdict
import logging

from einops import rearrange, reduce
import torch
from torch.utils.data.dataset import IterableDataset

import conf.datasets


log = logging.getLogger(__file__)


def euler_maruyama(dt, t, x, f, noise):
    return x + dt * f(t, x) + noise * dt**(1/2)


class DoubleWell:
    state_dimension = 1

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng

        self.data = defaultdict(list)
        self.data['times'] = self.times(cfg, rng)

        for k, state in zip(('true_state', 'predicted_state'), self.initialize_states(cfg, rng)):
            self.data[k].append(state)

        self.times = self.data['times'][:-1]
        true_state_noise = torch.randn((self.times.shape[0], *self.data['true_state'][0].shape), device=rng.device, generator=rng) * cfg.model_std
        for time_step, t in enumerate(self.times):
            state = self.data['true_state'][-1]
            if time_step > 0 and time_step % 20 == 0:
                state = -state
            next_state = euler_maruyama(
                cfg.time_step_size, t, state, self.dynamics, true_state_noise[time_step]
            )
            self.data['true_state'].append(next_state)
        self.data['true_state'] = rearrange(
            self.data['true_state'],
            't dim -> t dim'
        )

        observation_noise = torch.randn((self.data['times'].shape[0], *self.data['true_state'][0].shape), device=rng.device, generator=rng) * cfg.observation_std
        self.data['observation'] = self.data['true_state'] + observation_noise

        self.predicted_state_noise = torch.randn((self.times.shape[0], *self.data['predicted_state'][0].shape), device=rng.device, generator=rng)

    @staticmethod
    def times(cfg, rng):
        return cfg.time_step_size * torch.arange(cfg.time_step_count, device=rng.device)[:, None]

    @staticmethod
    def initialize_states(cfg, rng):
        true_state = -1 + torch.randn(1, device=rng.device, generator=rng) * cfg.true_state_initial_condition_std
        predicted_state = true_state + torch.randn((cfg.predicted_state_count, 1), device=rng.device, generator=rng) * cfg.predicted_state_initial_condition_std
        return true_state, predicted_state

    @staticmethod
    def dynamics(t, x):
        return -4 * x * (x**2 - 1)

    def predict(self, time_step, t, sampled_state):
        next_predicted_state = euler_maruyama(
            self.cfg.time_step_size, t, sampled_state, self.dynamics, self.predicted_state_noise[time_step]
        )
        return next_predicted_state

    def __iter__(self):
        for time_step, t in enumerate(self.times):
            yield time_step, t, self.data['predicted_state'][time_step], self.data['observation'][time_step + 1]
        self.data['predicted_state'] = rearrange(
            self.data['predicted_state'],
            't predicted_state_count dim -> t predicted_state_count dim'
        )


class PredictedStatesAndObservation(IterableDataset):
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def __iter__(self):
        # this does not train the model on the initial conditions
        for time_step, t, predicted_state, next_observation in self.dataset:
            next_predicted_state = self.dataset.predict(time_step, t, predicted_state)
            log.info('next_predicted_state mean: %s', reduce(
                next_predicted_state,
                'predicted_state_count dim ->', 'mean'
            ).item())
            yield time_step, t, next_predicted_state, next_observation
            sampled_state = self.model.sample(next_predicted_state, next_observation)
            log.info('sampled_state mean: %s', reduce(
                sampled_state,
                'predicted_state_count dim ->', 'mean'
            ).item())
            self.dataset.data['predicted_state'].append(sampled_state)


def get_dynamics_dataset(cfg, rng):
    if isinstance(cfg, conf.datasets.DoubleWell):
        return DoubleWell(cfg, rng)
    else:
        raise ValueError(f'Unknown dynamics dataset: {cfg}')
