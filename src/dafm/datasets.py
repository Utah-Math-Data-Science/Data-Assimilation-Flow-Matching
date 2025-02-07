import torch
from torch.utils.data.dataset import IterableDataset


class PredictedStatesAndObservation(IterableDataset):
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self.predicted_states = None
        self.true_state = None
        self.dt = torch.tensor(0.1, device=cfg.device)
        self.times = self.dt * torch.arange(cfg.time_step_count, device=cfg.device)[:, None]

    def model_dynamics(self, time_step, t, x):
        if time_step > 0 and (time_step + 1) % 20 == 0:
            x = -x
        return x - 4 * x * (x**2 - 1) * self.dt + 0.2 * torch.randn(x.shape, device=self.cfg.device) * torch.sqrt(self.dt)

    def make_observation(self, true_state):
        return true_state + torch.randn(true_state.shape, device=self.cfg.device) * 0.2

    def __iter__(self):
        self.true_state = -1 + torch.randn((1,), device=self.cfg.device) * .02
        self.predicted_states = -1 + torch.randn((10, 1), device=self.cfg.device) * .02
        for time_step, t in enumerate(self.times):
            yield t, self.predicted_states, self.make_observation(self.true_state)
            current_states = self.model.sample(self.predicted_states)
            self.true_state = self.model_dynamics(time_step, t, self.true_state)
            self.predicted_states = self.model_dynamics(time_step, t, self.predicted_states)
