import torch
from torch.utils.data.dataset import IterableDataset


class PredictedStatesAndObservation(IterableDataset):
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.predicted_state_count = 1000
        self.predicted_states = None
        self.true_state = None
        self.dt = torch.tensor(0.1, device=cfg.device)
        self.times = self.dt * torch.arange(cfg.time_step_count, device=cfg.device)[:, None]
        self.model_noise_std = 0.2
        self.observation_noise_std = 0.1
        self.true_initial_condition_noise_std = 0.02
        self.predicted_initial_condition_noise_std = 0.2

    def model_dynamics(self, time_step, t, x, noise_coefficient):
        if time_step > 0 and (time_step + 1) % 20 == 0:
            x = -x
        return x - 4 * x * (x**2 - 1) * self.dt + noise_coefficient * torch.randn(x.shape, device=self.cfg.device) * torch.sqrt(self.dt)

    def make_observation(self, true_state):
        return true_state + torch.randn(true_state.shape, device=self.cfg.device) * self.observation_noise_std

    def __iter__(self):
        self.true_state = -1 + torch.randn((1,), device=self.cfg.device) * self.true_initial_condition_noise_std
        self.predicted_states = self.true_state + torch.randn((self.predicted_state_count, 1), device=self.cfg.device) * self.predicted_initial_condition_noise_std
        for time_step, t in enumerate(self.times):
            observation = self.make_observation(self.true_state)

            yield time_step, t, self.predicted_states, observation

            self.true_state = self.model_dynamics(time_step, t, self.true_state, self.model_noise_std)
            # why no observation of the initial conditions in the first sampling?
            # because we are only allowed to guess the initial conditions, and then hone in our predictions?
            current_states = self.model.sample(self.predicted_states, None if time_step == 0 else observation)
            # why is the noise coefficient 1 here?
            self.predicted_states = self.model_dynamics(time_step, t, current_states, 1.)
