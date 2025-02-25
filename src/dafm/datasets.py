from collections import defaultdict
import logging

from einops import rearrange, reduce, unpack
import torch
from torch.utils.data.dataset import IterableDataset

import conf.datasets


log = logging.getLogger(__file__)


def euler_maruyama(dt, t, x, f, noise):
    return x + dt * f(t, x) + noise * dt**(1/2)


class DoubleWell:
    state_dimension = 1

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.data = defaultdict(list)
        self.data['times'] = self.times(cfg, device)

        for k, state in zip(('true_state', 'predicted_state'), self.initialize_states(cfg, device)):
            self.data[k].append(state)

        self.times = self.data['times'][:-1]
        true_state_noise = torch.randn((self.times.shape[0], *self.data['true_state'][0].shape), device=device) * cfg.model_std
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
            't 1 dim -> t 1 dim'
        )

        observation_noise = torch.randn((self.data['times'].shape[0], *self.data['true_state'][0].shape), device=device) * cfg.observation_std
        self.data['observation'] = self.data['true_state'] + observation_noise

        self.predicted_state_noise = torch.randn((self.times.shape[0], *self.data['predicted_state'][0].shape), device=device) * cfg.predicted_state_model_std

    @staticmethod
    def times(cfg, device):
        # number of time steps after the initial condition at time step zero
        return cfg.time_step_size * torch.arange(cfg.time_step_count + 1, device=device)[:, None]

    def initialize_states(self, cfg, device):
        true_state = -1 + torch.randn((1, self.state_dimension), device=device) * cfg.true_state_initial_condition_std
        predicted_state = true_state + torch.randn((cfg.predicted_state_count, self.state_dimension), device=device) * cfg.predicted_state_initial_condition_std
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


class Lorenz63:
    state_dimension = 3

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.data = defaultdict(list)
        time_step_indices, times_from_zero = self.times(cfg, device)
        self.data['true_state'].append(self.initialize_true_state(cfg, device))

        times_after_zero = times_from_zero[:-1]
        true_state_noise = torch.randn((times_after_zero.shape[0], *self.data['true_state'][0].shape), device=device) * cfg.model_std
        for time_step, t in enumerate(times_after_zero):
            state = self.data['true_state'][-1]
            next_state = euler_maruyama(
                cfg.time_step_size, t, state, self.dynamics, true_state_noise[time_step]
            )
            self.data['true_state'].append(next_state)
        self.data['true_state'] = rearrange(
            self.data['true_state'],
            't 1 dim -> t 1 dim'
        )

        times_to_keep = time_step_indices >= self.cfg.time_step_count_drop_first
        self.data['times'] = times_from_zero[times_to_keep]
        self.times = self.data['times'][:-1]
        self.data['true_state'] = self.data['true_state'][times_to_keep]
        self.data['predicted_state'].append(
            self.data['true_state'][0] + torch.randn((cfg.predicted_state_count, self.state_dimension), device=device) * cfg.predicted_state_initial_condition_std
        )

        observation_noise = torch.randn((self.data['times'].shape[0], *self.data['true_state'][0].shape), device=device) * cfg.observation_std
        self.data['observation'] = self.data['true_state'] + observation_noise

        self.predicted_state_noise = torch.randn((self.times.shape[0], *self.data['predicted_state'][0].shape), device=device) * cfg.predicted_state_model_std

    @staticmethod
    def times(cfg, device):
        time_step_indices = torch.arange(cfg.time_step_count + 1, device=device)
        return time_step_indices, cfg.time_step_size * time_step_indices[:, None]

    def initialize_true_state(self, cfg, device):
        true_state = torch.randn((1, self.state_dimension), device=device)
        return true_state

    def dynamics(self, t, x):
        x = x * self.cfg.rescaling
        x0, x1, x2 = unpack(x, self.state_dimension * [[]], 'state_count *')
        dot_x = rearrange([
            self.cfg.sigma * (x1 - x0),
            x0 * (self.cfg.rho - x2) - x1,
            x0 * x1 - self.cfg.beta * x2,
        ], 'dim state_count -> state_count dim')
        return dot_x / self.cfg.rescaling

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
        if self.model.cfg.train_on_initial_predicted_state:
            for _, (time_step, t, predicted_state, next_observation) in zip(range(1), self.dataset):
                yield time_step, t, predicted_state, next_observation, True
                if self.model.cfg.resample_initial_predicted_state:
                    sampled_state = self.model.sample(predicted_state, None)
                    self.dataset.data['predicted_state'][0] = sampled_state
        for time_step, t, predicted_state, next_observation in self.dataset:
            next_predicted_state = self.dataset.predict(time_step, t, predicted_state)
            log.info('next_predicted_state mean: %s', reduce(
                next_predicted_state,
                'predicted_state_count dim ->', 'mean'
            ).item())
            yield time_step, t, next_predicted_state, next_observation, False
            sampled_state = self.model.sample(next_predicted_state, next_observation)
            log.info('sampled_state mean: %s', reduce(
                sampled_state,
                'predicted_state_count dim ->', 'mean'
            ).item())
            self.dataset.data['predicted_state'].append(sampled_state)


def get_dynamics_dataset(cfg, device):
    if isinstance(cfg, conf.datasets.DoubleWell):
        return DoubleWell(cfg, device)
    elif isinstance(cfg, conf.datasets.Lorenz63):
        return Lorenz63(cfg, device)
    else:
        raise ValueError(f'Unknown dynamics dataset: {cfg}')
