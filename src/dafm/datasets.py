from collections import defaultdict
import logging
import time

from einops import rearrange, reduce, unpack
import torch
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
from torchdiffeq import odeint

import conf.datasets
import conf.inflation_scale
import dafm.observe


log = logging.getLogger(__file__)


def euler_maruyama(dt, t, x, f, noise):
    return x + dt * f(t, x) + noise * dt**(1/2)


class Dataset:
    def __init__(self, cfg, observe, state_perturbation, device):
        self.cfg = cfg
        self.observe = observe
        self.device = device
        self.store_trajectory_on_cpu = cfg.state_dimension > cfg.trajectory_stored_on_gpu_max_state_dimension
        if self.store_trajectory_on_cpu:
            device = 'cpu'

        self.data = defaultdict(list)
        time_step_indices, times_from_zero = self.time_steps_and_times(cfg, device)
        times_to_keep = time_step_indices >= self.cfg.time_step_count_drop_first
        self.data['times'] = times_from_zero[times_to_keep]
        self.data['true_state'].append(self.initialize_true_state(cfg, device))

        predicted_state_initial_condition_noise = torch.randn((cfg.predicted_state_count, cfg.state_dimension), device=device) * cfg.predicted_state_initial_condition_std
        true_state_noise = torch.randn((times_from_zero.shape[0] - 1, *self.data['true_state'][0].shape), device=device) * cfg.model_noise_std
        observation_noise = torch.randn((self.data['times'].shape[0], *self.observe(self.data['true_state'][0]).shape), device=device) * cfg.observation_noise_std
        self.predicted_state_noise = torch.randn((self.data['times'][:-1].shape[0], cfg.predicted_state_count, cfg.state_dimension), device=device) * cfg.predicted_state_model_noise_std

        for time_step, t_now_and_next in enumerate(times_from_zero.unfold(0, 2, 1)):
            true_state = self.data['true_state'][-1]
            true_state = state_perturbation(time_step, true_state)
            self.data['true_state'].append(
                self._step_state(time_step, t_now_and_next, true_state, true_state_noise[time_step])
            )

        self.data['true_state'] = rearrange(
            self.data['true_state'],
            't 1 dim -> t 1 dim'
        )

        self.data['true_state'] = self.data['true_state'][times_to_keep]

        predicted_state_initial_condition = predicted_state_initial_condition_noise
        if cfg.predicted_state_initial_condition_add_true_state:
            predicted_state_initial_condition += self.data['true_state'][0]
        self.data['predicted_state'].append(predicted_state_initial_condition)

        self.data['observation'] = self.observe(self.data['true_state']) + observation_noise

    @staticmethod
    def time_steps_and_times(cfg, device):
        time_step_indices = torch.arange(cfg.time_step_count + 1, device=device)
        return time_step_indices, cfg.time_step_size * time_step_indices[:, None]

    def initialize_true_state(self, cfg, device):
        raise NotImplementedError()

    def dynamics(self, t, x):
        raise NotImplementedError()

    def _step_state(self, time_step, t_now_and_next, state, model_noise):
        if self.cfg.integrator is conf.datasets.Integrator.RUNGE_KUTTA_4:
            if self.cfg.model_noise_std > 0 or self.cfg.predicted_state_model_noise_std > 0:
                raise ValueError(
                    f'{self.cfg.integrator.name} is not a stochastic differential equation integrator.'
                    'Please choose a different integrator (e.g., set dataset.integrator=EULER_MARUYAMA) or set dataset.model_noise_std=0 and dataset.predicted_state_model_noise_std=0.'
                )
            next_state = odeint(
                self.dynamics, state, rearrange(t_now_and_next, '1 times -> times'),
                method='rk4', options=dict(step_size=self.cfg.time_step_size),
            )[1]
        elif self.cfg.integrator is conf.datasets.Integrator.EULER_MARUYAMA:
            next_state = euler_maruyama(
            self.cfg.time_step_size, t_now_and_next[:, :1], state, self.dynamics, model_noise
            )
        else:
            raise ValueError(f'Unknown integrator: {self.cfg.integrator}')
        return next_state

    def predict(self, time_step, t_now_and_next, sampled_state):
        return self._step_state(time_step, t_now_and_next, sampled_state, self.predicted_state_noise[time_step])

    def __iter__(self):
        for time_step, t_now_and_next in enumerate(self.data['times'].unfold(0, 2, 1)):
            ignore_observation = time_step % self.cfg.observe_every_n_time_steps != 0
            yield time_step, t_now_and_next, self.data['predicted_state'][time_step], self.data['observation'][time_step + 1], ignore_observation
        self.data['predicted_state'] = rearrange(
            self.data['predicted_state'],
            't predicted_state_count dim -> t predicted_state_count dim'
        )


class DoubleWell(Dataset):
    def initialize_true_state(self, cfg, device):
        true_state = -1 + torch.randn((1, self.cfg.state_dimension), device=device) * cfg.true_state_initial_condition_std
        return true_state

    def dynamics(self, t, x):
        return -4 * x * (x.square() - 1)


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


class Simple(Dataset):
    def initialize_true_state(self, cfg, device):
        true_state = torch.zeros((1, self.cfg.state_dimension), device=device)
        return true_state

    def dynamics(self, t, x):
        return 1


class PredictedStatesAndObservation(IterableDataset):
    def __init__(self, dataset, model, logger=None):
        self.dataset = dataset
        self.model = model
        self.logger = logger
        self.time_step = None  # set in iter

    def __iter__(self):
        if self.model.cfg.train_on_initial_predicted_state:
            time_step, t_now_and_next, predicted_state, next_observation, _ = next(iter(self.dataset))
            self.time_step = time_step
            t_now_and_next, predicted_state, next_observation = map(
                lambda x: x.to(self.dataset.device),
                (t_now_and_next, predicted_state, next_observation)
            )
            yield self.model.cfg.epoch_count, time_step, t_now_and_next, predicted_state, next_observation, True
            if self.model.cfg.resample_initial_predicted_state:
                for done, sample_time_step, sample_time, sampled_state in self.model.sampling_steps(predicted_state, next_observation, self.dataset.observe):
                    if self.model.cfg.epoch_count_sampling > 0 and not done:
                        yield self.model.cfg.epoch_count_sampling, sample_time_step, sample_time, sampled_state, next_observation, True
                if self.dataset.store_trajectory_on_cpu:
                    sampled_state = sampled_state.to('cpu')
                self.dataset.data['predicted_state'][0] = sampled_state
        for time_step, t_now_and_next, predicted_state, next_observation, ignore_observation in tqdm(
           self.dataset,
           total=self.dataset.cfg.time_step_count - self.dataset.cfg.time_step_count_drop_first,
           initial=1,
           desc='Estimating state at time step',
        ):
            log_time_step_time_start = time.process_time()

            self.time_step = time_step
            next_predicted_state = self.dataset.predict(time_step, t_now_and_next, predicted_state)
            t_now_and_next, next_predicted_state, next_observation = map(
                lambda x: x.to(self.dataset.device),
                (t_now_and_next, next_predicted_state, next_observation)
            )
            if self.model.cfg.epoch_count > 0 and (not ignore_observation or self.model.cfg.train_when_ignoring_observation):
                yield self.model.cfg.epoch_count, time_step, t_now_and_next, next_predicted_state, next_observation, ignore_observation
            if not ignore_observation or self.model.cfg.resample_predicted_state_when_ignoring_observation:
                for done, sample_time_step, sample_time, sampled_state in self.model.sampling_steps(next_predicted_state, next_observation, self.dataset.observe):
                    if self.model.cfg.epoch_count_sampling > 0 and not done:
                        yield self.model.cfg.epoch_count_sampling, sample_time_step, sample_time, sampled_state, next_observation, ignore_observation
            else:
                sampled_state = next_predicted_state
            if self.model.cfg.use_state_perturbation:
                sampled_state = sampled_state + torch.randn_like(sampled_state) * self.model.cfg.state_perturbation_std
            if not isinstance(self.model.cfg.inflation_scale, conf.inflation_scale.NoScaling):
                sampled_state_mean = reduce(
                    sampled_state,
                    'predicted_state_count dim -> 1 dim',
                    'mean',
                )
                r2_from_mean = reduce(
                    (sampled_state - sampled_state_mean).square(),
                    'predicted_state_count dim -> predicted_state_count 1',
                    'sum',
                )
                sampled_state = (
                    sampled_state_mean + self.model.inflation_scale(r2_from_mean) * (sampled_state - sampled_state_mean)
                )
            if self.dataset.store_trajectory_on_cpu:
                sampled_state = sampled_state.to('cpu')
            self.dataset.data['predicted_state'].append(sampled_state)

            log_time_step_time_end = time.process_time()
            if self.logger is not None:
                self.logger.log_metrics(dict(
                    time_s=log_time_step_time_end - log_time_step_time_start,
                ), step=time_step + 1)


def get_state_perturbation(state_perturbation):
    if state_perturbation is conf.datasets.StatePerturbation.IDENTITY:
        return lambda _, x: x
    elif state_perturbation is conf.datasets.StatePerturbation.BAO_ET_AL_DOUBLE_WELL:
        def _perturb(time_step, state):
            if time_step > 0 and time_step % 20 == 0:
                state = -state
            return state
        return _perturb
    elif state_perturbation is conf.datasets.StatePerturbation.BAO_ET_AL_LORENZ_96:
        def _perturb(time_step, state):
            if time_step == 30 or time_step == 60:
                state = state + torch.randn_like(state) * 3.
            return state
        return _perturb
    else:
        raise ValueError(f'Unknown state perturbation: {state_perturbation}')


def get_dynamics_dataset(cfg, device):
    observe = dafm.observe.get_observe(cfg)
    state_perturbation = get_state_perturbation(cfg.state_perturbation)
    if isinstance(cfg, conf.datasets.DoubleWell):
        return DoubleWell(cfg, observe, state_perturbation, device)
    elif isinstance(cfg, conf.datasets.Lorenz63):
        return Lorenz63(cfg, observe, state_perturbation, device)
    elif isinstance(cfg, conf.datasets.Lorenz96):
        return Lorenz96(cfg, observe, state_perturbation, device)
    elif isinstance(cfg, conf.datasets.Simple):
        return Simple(cfg, observe, state_perturbation, device)
    else:
        raise ValueError(f'Unknown dynamics dataset: {cfg}')
