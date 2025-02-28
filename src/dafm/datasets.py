from collections import defaultdict
import logging

from einops import rearrange, reduce, unpack
import torch
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm

import conf.datasets
import dafm.observe


log = logging.getLogger(__file__)


def euler_maruyama(dt, t, x, f, noise):
    return x + dt * f(t, x) + noise * dt**(1/2)


class Dataset:
    def __init__(self, cfg, observe, device):
        self.cfg = cfg
        self.observe = observe
        self.device = device

        self.data = defaultdict(list)
        time_step_indices, times_from_zero = self.time_steps_and_times(cfg, device)
        times_to_keep = time_step_indices >= self.cfg.time_step_count_drop_first
        self.data['times'] = times_from_zero[times_to_keep]
        self.times = self.data['times'][:-1]
        self.data['true_state'].append(self.initialize_true_state(cfg, device))

        predicted_state_initial_condition_noise = torch.randn((cfg.predicted_state_count, cfg.state_dimension), device=device) * cfg.predicted_state_initial_condition_std
        true_state_noise = torch.randn((times_from_zero.shape[0] - 1, *self.data['true_state'][0].shape), device=device) * cfg.model_std
        observation_noise = torch.randn((self.data['times'].shape[0], *self.data['true_state'][0].shape), device=device) * cfg.observation_std
        self.predicted_state_noise = torch.randn((self.times.shape[0], cfg.predicted_state_count, cfg.state_dimension), device=device) * cfg.predicted_state_model_std

        for time_step, t in enumerate(times_from_zero[:-1]):
            self.data['true_state'].append(
                self.true_state_step(time_step, t, self.data['true_state'][-1], true_state_noise[time_step])
            )
        self.data['true_state'] = rearrange(
            self.data['true_state'],
            't 1 dim -> t 1 dim'
        )

        self.data['true_state'] = self.data['true_state'][times_to_keep]
        self.data['predicted_state'].append(
            self.data['true_state'][0] + predicted_state_initial_condition_noise
        )

        self.data['observation'] = self.observe(self.data['true_state']) + observation_noise

    @staticmethod
    def time_steps_and_times(cfg, device):
        time_step_indices = torch.arange(cfg.time_step_count + 1, device=device)
        return time_step_indices, cfg.time_step_size * time_step_indices[:, None]

    def initialize_true_state(self, cfg, device):
        raise NotImplementedError()

    def dynamics(self, t, x):
        raise NotImplementedError()

    def true_state_step(self, time_step, t, true_state, model_noise):
        raise NotImplementedError()

    def predict(self, time_step, t, sampled_state):
        next_predicted_state = euler_maruyama(
            self.cfg.time_step_size, t, sampled_state, self.dynamics, self.predicted_state_noise[time_step]
        )
        return next_predicted_state

    def __iter__(self):
        for time_step, t in enumerate(self.times):
            ignore_observation = time_step % self.cfg.observe_every_n_time_steps != 0
            yield time_step, t, self.data['predicted_state'][time_step], self.data['observation'][time_step + 1], ignore_observation
        self.data['predicted_state'] = rearrange(
            self.data['predicted_state'],
            't predicted_state_count dim -> t predicted_state_count dim'
        )


class DoubleWell(Dataset):
    def initialize_true_state(self, cfg, device):
        true_state = -1 + torch.randn((1, self.cfg.state_dimension), device=device) * cfg.true_state_initial_condition_std
        return true_state

    def dynamics(self, t, x):
        return -4 * x * (x**2 - 1)

    def true_state_step(self, time_step, t, true_state, model_noise):
        if time_step > 0 and time_step % 20 == 0:
            true_state = -true_state
        next_true_state = euler_maruyama(
            self.cfg.time_step_size, t, true_state, self.dynamics, model_noise
        )
        return next_true_state


class Lorenz63(Dataset):
    def initialize_true_state(self, cfg, device):
        true_state = torch.randn((1, self.cfg.state_dimension), device=device)
        return true_state

    def dynamics(self, t, x):
        x = x * self.cfg.rescaling
        x0, x1, x2 = unpack(x, self.cfg.state_dimension * [[]], 'state_count *')
        dot_x = rearrange([
            self.cfg.sigma * (x1 - x0),
            x0 * (self.cfg.rho - x2) - x1,
            x0 * x1 - self.cfg.beta * x2,
        ], 'dim state_count -> state_count dim')
        return dot_x / self.cfg.rescaling

    def true_state_step(self, time_step, t, true_state, model_noise):
        next_true_state = euler_maruyama(
            self.cfg.time_step_size, t, true_state, self.dynamics, model_noise
        )
        return next_true_state


class Lorenz96(Dataset):
    def initialize_true_state(self, cfg, device):
        true_state = cfg.true_state_initial_condition_mean + torch.randn((1, self.cfg.state_dimension), device=device) * cfg.true_state_initial_condition_std
        # true_state = torch.ones((1, self.cfg.state_dimension), device=device) * self.cfg.forcing
        return true_state

    def dynamics(self, t, x):
        x_p1 = x.roll(-1, -1)
        x_m2 = x.roll(2, -1)
        x_m1 = x.roll(1, -1)
        return (x_p1 - x_m2) * x_m1 - x + self.cfg.forcing

    def true_state_step(self, time_step, t, true_state, model_noise):
        if time_step > 0 and time_step % 30 == 0:
            true_state = true_state + torch.randn_like(true_state) * 3.
        next_true_state = euler_maruyama(
            self.cfg.time_step_size, t, true_state, self.dynamics, model_noise
        )
        return next_true_state


class PredictedStatesAndObservation(IterableDataset):
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        self.time_step = None  # set in iter

    def __iter__(self):
        if self.model.cfg.train_on_initial_predicted_state:
            for _, (time_step, t, predicted_state, next_observation, _) in zip(range(1), self.dataset):
                self.time_step = time_step
                yield time_step, t, predicted_state, next_observation, True
                if self.model.cfg.resample_initial_predicted_state:
                    sampled_state = self.model.sample(predicted_state, None, self.dataset.observe)
                    self.dataset.data['predicted_state'][0] = sampled_state
        for time_step, t, predicted_state, next_observation, ignore_observation in tqdm(self.dataset, total=self.dataset.cfg.time_step_count, desc='Estimating state at time step', initial=1):
            self.time_step = time_step
            next_predicted_state = self.dataset.predict(time_step, t, predicted_state)
            # log.info('next_predicted_state mean: %s', reduce(
            #     next_predicted_state,
            #     'predicted_state_count dim ->', 'mean'
            # ).item())
            if not ignore_observation or self.model.cfg.train_when_ignoring_observation:
                yield time_step, t, next_predicted_state, next_observation, ignore_observation
            if self.model.cfg.resample_predicted_state_when_ignoring_observation:
                sampled_state = self.model.sample(next_predicted_state, next_observation)
            else:
                sampled_state = next_predicted_state
            sampled_state = self.model.sample(next_predicted_state, next_observation, self.dataset.observe)
            # log.info('sampled_state mean: %s', reduce(
            #     sampled_state,
            #     'predicted_state_count dim ->', 'mean'
            # ).item())
            self.dataset.data['predicted_state'].append(sampled_state)


def get_dynamics_dataset(cfg, device):
    observe = dafm.observe.get_observe(cfg)
    if isinstance(cfg, conf.datasets.DoubleWell):
        return DoubleWell(cfg, observe, device)
    elif isinstance(cfg, conf.datasets.Lorenz63):
        return Lorenz63(cfg, observe, device)
    elif isinstance(cfg, conf.datasets.Lorenz96):
        return Lorenz96(cfg, observe, device)
    else:
        raise ValueError(f'Unknown dynamics dataset: {cfg}')
