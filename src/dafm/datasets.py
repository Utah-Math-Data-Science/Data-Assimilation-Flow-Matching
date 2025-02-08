import logging

import torch
from torch.utils.data.dataset import IterableDataset

import conf.datasets


log = logging.getLogger(__file__)


class DoubleWell:
    state_dimension = 1

    def __init__(self, cfg):
        self.cfg = cfg

    def times(self):
        return self.cfg.time_step_size * torch.arange(self.cfg.time_step_count)[:, None]

    def initialize_states(self):
        true_state = -1 + torch.randn(1) * self.cfg.true_state_initial_condition_std
        predicted_states = true_state + torch.randn((self.cfg.predicted_state_count, 1)) * self.cfg.predicted_state_initial_condition_std
        return true_state, predicted_states

    def dynamics(self, time_step, t, x, is_predicted_state=True):
        if time_step > 0 and (time_step + 1) % 20 == 0:
            x = -x
        # why is the noise coefficient 1 here when is_predicted_state is True?
        noise_coefficient = 1 if is_predicted_state else self.cfg.model_std
        return x - 4 * x * (x**2 - 1) * self.cfg.time_step_size + noise_coefficient * torch.randn(x.shape, device=x.device) * (self.cfg.time_step_size)**(1/2)

    def observe(self, true_state):
        return true_state + torch.randn_like(true_state) * self.cfg.observation_std


class PredictedStatesAndObservation(IterableDataset):
    def __init__(self, dataset, model, device):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.times = dataset.times().to(device)

    def __iter__(self):
        self.true_state, self.predicted_states = map(lambda x: x.to(self.device), self.dataset.initialize_states())
        for time_step, t in enumerate(self.times):
            observation = self.dataset.observe(self.true_state)
            yield time_step, t, self.predicted_states, observation
            self.true_state = self.dataset.dynamics(time_step, t, self.true_state, is_predicted_state=False)
            # why no observation of the initial conditions in the first sampling?
            # because we are only allowed to guess the initial conditions, and then hone in our predictions?
            current_states = self.model.sample(self.predicted_states, None if time_step == 0 else observation)
            self.predicted_states = self.dataset.dynamics(time_step, t, current_states)


def get_dynamics_dataset(cfg):
    if isinstance(cfg, conf.datasets.DoubleWell):
        return DoubleWell(cfg)
    else:
        raise ValueError(f'Unknown dynamics dataset: {cfg}')
