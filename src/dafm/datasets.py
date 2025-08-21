from collections import defaultdict
import logging
import time

from einops import rearrange, reduce, unpack
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
from torchdiffeq import odeint
import dapper.mods.KS
import polars

import conf.datasets
import conf.inflation_scale
import conf.diffusion_path
import dafm.observe


log = logging.getLogger(__file__)


def euler_maruyama(dt, t, x, f, noise):
    return x + dt * f(t, x) + noise * dt**(1/2)


class Dataset:
    def __init__(self, cfg, rng, observe, state_perturbation, device, delete_true_state=False):
        self.cfg = cfg
        self.observe = observe
        self.device = device
        self.state_perturbation = state_perturbation

        (
            self.rng_initial_state,
            self.rng_model_noise,
            self.rng_predicted_state_model_noise,
            self.rng_observation_noise,
        ) = rng.spawn(4)

        true_state = self.initialize_true_state(cfg, device)
        for time_steps in zip(range(cfg.time_step_count_drop_first), range(1, cfg.time_step_count_drop_first + 1)):
            times = [ts * cfg.time_step_size for ts in time_steps]
            true_state = state_perturbation(time_steps[0], true_state)
            noise = self.get_noise(true_state.shape, self.rng_model_noise, cfg.model_noise_std, device=device)
            true_state = self._step_state(time_steps[0], times, true_state, noise)
        self.true_state = true_state

        predicted_state_initial_condition = self.get_noise(
            (cfg.predicted_state_count, *true_state.shape[1:]),
            self.rng_predicted_state_model_noise,
            cfg.predicted_state_initial_condition_std,
            device=true_state.device
        )
        if cfg.predicted_state_initial_condition_add_true_state:
            predicted_state_initial_condition += true_state
        self.current_predicted_state = predicted_state_initial_condition

    def get_noise(self, shape, rng, std, **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = torch.float32
        if std > 0:
            noise = torch.tensor(rng.normal(scale=std, size=shape), **kwargs)
        else:
            noise = 0.
        return noise

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
                self.dynamics, state, torch.tensor(t_now_and_next, device=state.device),
                method='rk4', options=dict(step_size=self.cfg.time_step_size),
            )[1]
        elif self.cfg.integrator is conf.datasets.Integrator.EULER_MARUYAMA:
            next_state = euler_maruyama(
            self.cfg.time_step_size, t_now_and_next[0], state, self.dynamics, model_noise
            )
        else:
            raise ValueError(f'Unknown integrator: {self.cfg.integrator}')
        return next_state

    def predict(self, time_step, t_now_and_next, sampled_state, noise):
        return self._step_state(time_step, t_now_and_next, sampled_state, noise)

    def __iter__(self):
        for time_steps in zip(
            range(self.cfg.time_step_count_drop_first, self.cfg.time_step_count),
            range(self.cfg.time_step_count_drop_first + 1, self.cfg.time_step_count + 1),
        ):
            times = [ts * self.cfg.time_step_size for ts in time_steps]
            self.true_state = self.state_perturbation(time_steps[0], self.true_state)
            model_noise = self.get_noise(self.true_state.shape, self.rng_model_noise, self.cfg.model_noise_std, device=self.device)
            self.true_state = self._step_state(time_steps[0], times, self.true_state, model_noise)
            ignore_observation = (time_steps[0] - self.cfg.time_step_count_drop_first) % self.cfg.observe_every_n_time_steps != 0
            observation = self.observe(self.true_state)
            observation_noise = self.get_noise(observation.shape, self.rng_observation_noise, self.cfg.observation_noise_std, device=self.true_state.device)
            yield time_steps[0], times, self.current_predicted_state, observation + observation_noise, ignore_observation


class DoubleWell(Dataset):
    def initialize_true_state(self, cfg, device):
        true_state = -1 + self.get_noise((1, self.cfg.state_dimension), self.rng_initial_state, cfg.true_state_initial_condition_std, device=device)
        return true_state

    def dynamics(self, t, x):
        return -4 * x * (x.square() - 1)


class Lorenz63(Dataset):
    def initialize_true_state(self, cfg, device):
        true_state = self.get_noise((1, self.cfg.state_dimension), self.rng_initial_state, 1., device=device)
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
        true_state = cfg.true_state_initial_condition_mean + self.get_noise((1, self.cfg.state_dimension), self.rng_initial_state, cfg.true_state_initial_condition_std, device=device)
        # true_state = torch.ones((1, self.cfg.state_dimension), device=device) * self.cfg.forcing
        return true_state

    def dynamics(self, t, x):
        x_p1 = x.roll(-1, -1)
        x_m2 = x.roll(2, -1)
        x_m1 = x.roll(1, -1)
        return (x_p1 - x_m2) * x_m1 - x + self.cfg.forcing


class NavierStokes(Dataset):
    def initialize_true_state(self, cfg, device):
        # linspace excluding endpoint
        horizontal_axis = torch.linspace(0, cfg.grid_width, cfg.grid_horizontal_count + 1, device=device)[:-1]
        vertical_axis = torch.linspace(0, cfg.grid_height, cfg.grid_vertical_count + 1, device=device)[:-1]
        self.grid_horizontal_spacing = (horizontal_axis[1] - horizontal_axis[0]).item()
        self.grid_vertical_spacing = (vertical_axis[1] - vertical_axis[0]).item()

        horizontal_grid, vertical_grid = torch.meshgrid(horizontal_axis, vertical_axis, indexing='ij')

        horizontal_velocity = self.sample_squared_exponential_gaussian_process(cfg, device, horizontal_grid.dtype)
        vertical_velocity = self.sample_squared_exponential_gaussian_process(cfg, device, horizontal_grid.dtype)
        b0 = self.divergence(horizontal_velocity, vertical_velocity) / cfg.time_step_size
        pressure = torch.zeros((cfg.grid_horizontal_count, cfg.grid_vertical_count), device=device)
        pressure  = self.pressure_poisson(pressure, b0)
        dhorizontal_pressure, dvertical_pressure = self.gradient(pressure)
        horizontal_velocity = horizontal_velocity - cfg.time_step_size * dhorizontal_pressure
        vertical_velocity = vertical_velocity - cfg.time_step_size * dvertical_pressure

        x = rearrange(
            [pressure, horizontal_velocity, vertical_velocity],
            'value_count grid_horizontal_count grid_vertical_count -> 1 (value_count grid_horizontal_count grid_vertical_count)'
        )

        self.horizontal_forcing = cfg.forcing_amplitude * torch.sin(2 * torch.pi * cfg.vertical_mode_number / cfg.grid_height * vertical_grid)
        self.vertical_forcing = torch.zeros_like(self.horizontal_forcing)

        return x

    def sample_squared_exponential_gaussian_process(self, cfg, device, dtype):
        w = self.rng_initial_state.normal(size=(cfg.grid_horizontal_count, cfg.grid_vertical_count))
        w_hat = np.fft.fft2(w)
        kx = np.fft.fftfreq(cfg.grid_horizontal_count, d=self.grid_horizontal_spacing)
        ky = np.fft.fftfreq(cfg.grid_vertical_count, d=self.grid_vertical_spacing)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        S = np.exp(-2 * np.pi**2 * cfg.gaussian_process_length_scale**2 * (KX**2 + KY**2))
        f = np.fft.ifft2(w_hat * np.sqrt(S)).real
        out = cfg.gaussian_process_std * (f - f.mean()) / f.std()
        return torch.tensor(out, device=device, dtype=dtype)

    def _step_state(self, time_step, t_now_and_next, state, model_noise):
        """
        Chorin's projection method.
        """
        pressure, horizontal_velocity, vertical_velocity = rearrange(
            state,
            'predicted_state_count (value_count grid_horizontal_count grid_vertical_count) -> value_count predicted_state_count grid_horizontal_count grid_vertical_count',
            value_count=3, grid_horizontal_count=self.cfg.grid_horizontal_count, grid_vertical_count=self.cfg.grid_vertical_count,
        )

        # advection
        horizontal_advection = self.advect(horizontal_velocity, vertical_velocity, horizontal_velocity)
        vertical_advection = self.advect(horizontal_velocity, vertical_velocity, vertical_velocity)

        # intermediate velocity + forcing
        horizontal_velocity_predictive = horizontal_velocity + self.cfg.time_step_size * (
            self.cfg.viscosity * self.laplacian(horizontal_velocity)
            - horizontal_advection + self.horizontal_forcing
        )
        vertical_velocity_predictive = vertical_velocity + self.cfg.time_step_size * (
            self.cfg.viscosity * self.laplacian(vertical_velocity)
            - vertical_advection + self.vertical_forcing
        )

        # pressure correction
        b = self.divergence(horizontal_velocity_predictive, vertical_velocity_predictive) / self.cfg.time_step_size
        pressure = self.pressure_poisson(pressure, b)

        # project to incompressible
        dhorizontal_pressure, dvertical_pressure = self.gradient(pressure)
        horizontal_velocity = horizontal_velocity_predictive - self.cfg.time_step_size * dhorizontal_pressure
        vertical_velocity = vertical_velocity_predictive - self.cfg.time_step_size * dvertical_pressure

        state = rearrange(
            [pressure, horizontal_velocity, vertical_velocity],
            'value_count predicted_state_count grid_horizontal_count grid_vertical_count -> predicted_state_count (value_count grid_horizontal_count grid_vertical_count)'
        )

        return state

    def laplacian(self, f):
        return (
            (f.roll(-1, -2) - 2 * f + f.roll(1, -2)) / self.grid_horizontal_spacing**2
            + (f.roll(-1, -1) - 2 * f + f.roll(1, -1)) / self.grid_vertical_spacing**2
        )

    def divergence(self, horizontal_f, vertical_f):
        return (
            (horizontal_f.roll(-1, -2) - horizontal_f.roll(1, -2)) / (2 * self.grid_horizontal_spacing)
            + (vertical_f.roll(-1, -1) - vertical_f.roll(1, -1)) / (2 * self.grid_vertical_spacing)
        )

    def gradient(self, f):
        dfdx = (f.roll(-1, -2) - f.roll(1, -2)) / (2 * self.grid_horizontal_spacing)
        dfdy = (f.roll(-1, -1) - f.roll(1, -1)) / (2 * self.grid_vertical_spacing)
        return dfdx, dfdy

    def pressure_poisson(self, pressure, b):
        for _ in range(self.cfg.pressure_poisson_solve_iteration_count):
            pressure = (
                (pressure.roll(1, -2) + pressure.roll(-1, -2)) * self.grid_vertical_spacing**2
                + (pressure.roll(1, -1) + pressure.roll(-1, -1)) * self.grid_horizontal_spacing**2
                - b * self.grid_horizontal_spacing**2 * self.grid_vertical_spacing**2
            ) / (
                2 * (self.grid_horizontal_spacing**2 + self.grid_vertical_spacing**2)
            )
        return pressure

    def advect(self, u, v, f):
        dfdx = (f.roll(-1, -2) - f.roll(1, -2)) / (2 * self.grid_horizontal_spacing)
        dfdy = (f.roll(-1, -1) - f.roll(1, -1)) / (2 * self.grid_vertical_spacing)
        return u * dfdx + v * dfdy


class KuramotoSivashinsky(Dataset):
    def __init__(self, cfg, rng, observe, state_perturbation, device, delete_true_state=False):
        if cfg.floating_point_precision == 64:
            self.dtype = torch.float64
        elif cfg.floating_point_precision == 32:
            self.dtype = torch.float32
        else:
            raise ValueError(f'Unknown floating point precision (should be 32 or 64): {self.cfg.floating_point_precision}')
        self.solve = self.etd_rk4_wrapper(cfg, device)
        super().__init__(cfg, rng, observe, state_perturbation, device, delete_true_state=delete_true_state)

    def initialize_true_state(self, cfg, device):
        x0 = torch.tensor(dapper.mods.KS.Model(
            dt=cfg.time_step_size,
            DL=cfg.domain_pi_multiple,
            Nx=cfg.state_dimension,
        ).x0, device=device, dtype=self.dtype)
        x0 = rearrange(x0, 'state_dimension -> 1 state_dimension')
        return x0 + self.get_noise(x0.shape, self.rng_initial_state, 1., device=device)

    def _step_state(self, time_step, t_now_and_next, state, model_noise):
        return self.solve(None, state)

    def etd_rk4_wrapper(self, cfg, device):
        """ Returns an ETD-RK4 integrator for the KS equation. Currently very specific, need
        to adjust this to fit into the same framework as the ODE integrators

        Directly ported from https://github.com/nansencenter/DAPPER/blob/master/dapper/mods/KS/core.py
        which is adapted from kursiv.m of Kassam and Trefethen, 2002, doi.org/10.1137/S1064827502410633.
        """
        kk = np.append(np.arange(0, cfg.state_dimension / 2), 0) * 2 / cfg.domain_pi_multiple  # wave nums for rfft
        h = cfg.time_step_size

        # Operators
        L = kk ** 2 - kk ** 4  # Linear operator for K-S eqn: F[ - u_xx - u_xxxx]

        # Precompute ETDRK4 scalar quantities
        E = torch.tensor(np.exp(h * L), device=device, dtype=self.dtype).unsqueeze(0)  # Integrating factor, eval at dt
        E2 = torch.tensor(np.exp(h * L / 2), device=device, dtype=self.dtype).unsqueeze(0)  # Integrating factor, eval at dt/2

        # Roots of unity are used to discretize a circular contour...
        nRoots = 16
        roots = np.exp(1j * np.pi * (0.5 + np.arange(nRoots)) / nRoots)
        # ... the associated integral then reduces to the mean,
        # g(CL).mean(axis=-1) ~= g(L), whose computation is more stable.
        CL = h * L[:, None] + roots  # Contour for (each element of) L
        # E * exact_integral of integrating factor:
        Q = torch.tensor(h * ((np.exp(CL / 2) - 1) / CL).mean(axis=-1).real, device=device).unsqueeze(0)
        # RK4 coefficients (modified by Cox-Matthews):
        f1 = torch.tensor(h * ((-4 - CL + np.exp(CL) * (4 - 3 * CL + CL ** 2)) / CL ** 3).mean(axis=-1).real, device=device).unsqueeze(0)
        f2 = torch.tensor(h * ((2 + CL + np.exp(CL) * (-2 + CL)) / CL ** 3).mean(axis=-1).real, device=device).unsqueeze(0)
        f3 = torch.tensor(h * ((-4 - 3 * CL - CL ** 2 + np.exp(CL) * (4 - CL)) / CL ** 3).mean(axis=-1).real, device=device).unsqueeze(0)

        D = 1j * torch.tensor(kk, device=device, dtype=self.dtype)  # Differentiation to compute:  F[ u_x ]

        def NL(v):
            return -.5 * D * torch.fft.rfft(torch.fft.irfft(v, dim=-1) ** 2, dim=-1)

        def inner(t, v):
            v = torch.fft.rfft(v, dim=-1)
            N1 = NL(v)
            v1 = E2 * v + Q * N1

            N2a = NL(v1)
            v2a = E2 * v + Q * N2a

            N2b = NL(v2a)
            v2b = E2 * v1 + Q * (2 * N2b - N1)

            N3 = NL(v2b)
            v = E * v + N1 * f1 + 2 * (N2a + N2b) * f2 + N3 * f3
            return torch.fft.irfft(v, dim=-1)

        return inner


class Simple(Dataset):
    def initialize_true_state(self, cfg, device):
        true_state = torch.zeros((1, self.cfg.state_dimension), device=device)
        return true_state

    def dynamics(self, t, x):
        return 1


class PredictedStatesAndObservation(IterableDataset):
    def __init__(self, dataset, model, logger=None, data_to_save_callback=lambda *_: None):
        self.dataset = dataset
        self.model = model
        self.logger = logger
        self.time_step = None  # set in iter
        self.data_to_save_callback = data_to_save_callback

    def __iter__(self):
        if self.model.cfg.train_on_initial_predicted_state:
            time_step, t_now_and_next, predicted_state, next_observation, _ = next(iter(self.dataset))
            self.time_step = time_step
            if isinstance(self.model.cfg.diffusion_path, conf.diffusion_path.PreviousPosteriorToPredictive):
                self.model.diffusion_path.set_previous_posterior(predicted_state)
            yield self.model.cfg.epoch_count, time_step, t_now_and_next, predicted_state, next_observation, True
            if self.model.cfg.resample_initial_predicted_state:
                for done, sample_time_step, sample_time, sampled_state in self.model.sampling_steps(predicted_state, next_observation, self.dataset.observe):
                    if self.model.cfg.epoch_count_sampling > 0 and not done:
                        yield self.model.cfg.epoch_count_sampling, sample_time_step, sample_time, sampled_state, next_observation, True
                self.dataset.current_predicted_state= sampled_state
        data_to_save = dict(
            time_step=[],
            times=[], predicted_state=[],
        )
        for time_step, t_now_and_next, predicted_state, next_observation, ignore_observation in tqdm(
           self.dataset,
           total=self.dataset.cfg.time_step_count - self.dataset.cfg.time_step_count_drop_first,
           initial=1,
           desc='Estimating state at time step',
        ):
            data_to_save['time_step'].append(time_step)
            data_to_save['times'].append(torch.tensor([t_now_and_next[0]]))
            data_to_save['predicted_state'].append(predicted_state.cpu())

            self.time_step = time_step
            if hasattr(self.model.cfg, 'diffusion_path') and isinstance(self.model.cfg.diffusion_path, conf.diffusion_path.PreviousPosteriorToPredictive):
                self.model.diffusion_path.set_previous_posterior(predicted_state.to(self.dataset.device))

            # tensors that may need to be moved to a device
            # keep these out of the timing
            predicted_state_model_noise = self.dataset.get_noise(predicted_state.shape, self.dataset.rng_predicted_state_model_noise, self.dataset.cfg.predicted_state_model_noise_std, device=self.dataset.device)
            if (not ignore_observation or self.model.cfg.resample_predicted_state_when_ignoring_observation) and self.dataset.cfg.use_predicted_state_perturbation:
                predicted_state_perturbation_before_sampling = self.dataset.get_noise(predicted_state.shape, self.dataset.rng_predicted_state_model_noise, self.dataset.cfg.predicted_state_perturbation_std, device=predicted_state.device)
            if not ignore_observation and self.model.cfg.use_state_perturbation:
                predicted_state_perturbation_after_sampling = self.dataset.get_noise(predicted_state.shape, self.dataset.rng_predicted_state_model_noise, self.model.cfg.state_perturbation_std, device=predicted_state.device)

            log_time_step_time_start = time.process_time()

            next_predicted_state = self.dataset.predict(time_step, t_now_and_next, predicted_state, predicted_state_model_noise)

            if self.model.cfg.epoch_count > 0 and (not ignore_observation or self.model.cfg.train_when_ignoring_observation):
                yield self.model.cfg.epoch_count, time_step, t_now_and_next, next_predicted_state, next_observation, ignore_observation

            if not ignore_observation or self.model.cfg.resample_predicted_state_when_ignoring_observation:
                if self.dataset.cfg.use_predicted_state_perturbation:
                    next_predicted_state = next_predicted_state + predicted_state_perturbation_before_sampling
                for done, sample_time_step, sample_time, sampled_state in self.model.sampling_steps(next_predicted_state, next_observation, self.dataset.observe):
                    if self.model.cfg.epoch_count_sampling > 0 and not done:
                        yield self.model.cfg.epoch_count_sampling, sample_time_step, sample_time, sampled_state, next_observation, ignore_observation
            else:
                sampled_state = next_predicted_state

            if not ignore_observation:
                if self.model.cfg.use_state_perturbation:
                    sampled_state = sampled_state + predicted_state_perturbation_after_sampling
                if not isinstance(self.model.cfg.inflation_scale, conf.inflation_scale.NoScaling):
                    sampled_state_mean = reduce(
                        sampled_state,
                        'predicted_state_count dim -> 1 dim',
                        'mean',
                    )
                    sampled_state_centered = sampled_state - sampled_state_mean
                    sampled_state = (
                        sampled_state_mean + self.model.inflation_scale(sampled_state_centered) * sampled_state_centered
                    )
            self.dataset.current_predicted_state = sampled_state

            log_time_step_time_end = time.process_time()

            if self.logger is not None:
                self.logger.log_metrics(dict(
                    time_s=log_time_step_time_end - log_time_step_time_start,
                    rmse=reduce((sampled_state.mean(0, keepdims=True) - self.dataset.true_state).square(), '1 dim ->', 'mean').sqrt(),
                    crps=continuous_ranked_probability_score(sampled_state, self.dataset.true_state),
                ), step=time_step + 1)

            is_last_time_step = time_step == self.dataset.cfg.time_step_count - 1
            if (
                self.dataset.cfg.save_data_every_n_time_steps is not None and time_step > 0 and time_step % self.dataset.cfg.save_data_every_n_time_steps == 0
                or
                is_last_time_step
            ):
                if is_last_time_step:
                    data_to_save['time_step'].append(time_step + 1)
                    data_to_save['times'].append(torch.tensor([t_now_and_next[1]]))
                    data_to_save['predicted_state'].append(sampled_state.cpu())
                    self.data_to_save_callback(time_step + 1, data_to_save)
                else:
                    self.data_to_save_callback(time_step, data_to_save)

                data_to_save = dict(
                    time_step=[],
                    times=[], predicted_state=[],
                )

        if self.logger is not None:
            self.logger.save()


def continuous_ranked_probability_score(predicted_state, true_state):
    mean_r_from_true = reduce(
        (predicted_state - true_state).square(),
        'predicted_state_count dim -> predicted_state_count',
        'sum',
    ).sqrt().mean(0)

    predicted_state_count = predicted_state.shape[0]
    predicted_state_idx = torch.arange(predicted_state_count, dtype=torch.long, device=predicted_state.device)
    predicted_state_a = predicted_state_idx.repeat_interleave(predicted_state_count)
    predicted_state_b = predicted_state_idx.repeat(predicted_state_count)
    self_loop_or_symmetric_edge = predicted_state_a >= predicted_state_b
    predicted_state_a = predicted_state_a[~self_loop_or_symmetric_edge]
    predicted_state_b = predicted_state_b[~self_loop_or_symmetric_edge]
    half_mean_r_between_predicted_states = reduce(
        (predicted_state[predicted_state_a] - predicted_state[predicted_state_b]).square(),
        'predicted_state_count dim -> predicted_state_count',
        'sum',
    ).sqrt().sum(0) / predicted_state_count**2

    return mean_r_from_true  - half_mean_r_between_predicted_states


def save_trajectories(cfg, data, save_dir):
    data['times'] = rearrange(data['times'], 'time_step 1 -> time_step').cpu().numpy()
    data['times'] = polars.DataFrame(
        data['times'],
        schema=['times'],
        orient='row',
    )
    data['predicted_state'] = rearrange(
        data['predicted_state'],
        't predicted_state_count dim -> t predicted_state_count dim'
    )
    time_step_count_predicted, predicted_state_count, dim = data['predicted_state'].shape
    if cfg.save_only_mean_std:
        data['predicted_state_mean'] = reduce(
            data['predicted_state'],
            't predicted_state_count dim -> t dim',
            'mean',
        )
        data['predicted_state_std'] = reduce(
            data['predicted_state'],
            't predicted_state_count dim -> t dim',
            torch.std,
        )
        for stat in ('mean', 'std'):
            data[f'predicted_state_{stat}'] = polars.DataFrame(
                data[f'predicted_state_{stat}'].cpu().numpy(),
                schema=[
                    f'predicted_state_{stat}_dim_{d}'
                    for d in range(dim)
                ],
                orient='row',
            )
        df = polars.concat([data[k] for k in ('times', 'predicted_state_mean', 'predicted_state_std')], how='horizontal')
    else:
        data['true_state'] = rearrange(
            data['true_state'],
            't 1 dim -> t dim',
        )
        data['observation'] = rearrange(
            data['observation'],
            't 1 dim -> t dim',
        )
        data['predicted_state'] = rearrange(
            data['predicted_state'],
            't predicted_state_count dim -> t (predicted_state_count dim)'
        )
        data['predicted_state'] = polars.DataFrame(
            data['predicted_state'].cpu().numpy(),
            schema=[
                f'predicted_state_{state}_dim_{d}'
                for state in range(predicted_state_count)
                for d in range(dim)
            ],
            orient='row',
        )
        data['true_state'] = polars.DataFrame(
            data['true_state'].cpu().numpy(),
            schema=[f'true_state_dim_{d}' for d in range(dim)],
            orient='row',
        )
        data['observation'] = polars.DataFrame(
            data['observation'].cpu().numpy(),
            schema=[f'observation_dim_{d}' for d in range(data['observation'].shape[1])],
            orient='row',
        )
        df = polars.concat([data[k] for k in ('times', 'true_state', 'observation', 'predicted_state')], how='horizontal')
    df.write_parquet(save_dir)
    log.info('Trajectory data saved to %s', save_dir)


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


def get_dynamics_dataset(cfg, rng, device, delete_true_state=False):
    observe = dafm.observe.get_observe(cfg)
    state_perturbation = get_state_perturbation(cfg.state_perturbation)
    if isinstance(cfg, conf.datasets.DoubleWell):
        return DoubleWell(cfg, rng, observe, state_perturbation, device, delete_true_state=delete_true_state)
    elif isinstance(cfg, conf.datasets.Lorenz63):
        return Lorenz63(cfg, rng, observe, state_perturbation, device, delete_true_state=delete_true_state)
    elif isinstance(cfg, conf.datasets.Lorenz96):
        return Lorenz96(cfg, rng, observe, state_perturbation, device, delete_true_state=delete_true_state)
    elif isinstance(cfg, conf.datasets.NavierStokes):
        return NavierStokes(cfg, rng, observe, state_perturbation, device, delete_true_state=delete_true_state)
    elif isinstance(cfg, conf.datasets.KuramotoSivashinsky):
        return KuramotoSivashinsky(cfg, rng, observe, state_perturbation, device, delete_true_state=delete_true_state)
    elif isinstance(cfg, conf.datasets.Simple):
        return Simple(cfg, rng, observe, state_perturbation, device, delete_true_state=delete_true_state)
    else:
        raise ValueError(f'Unknown dynamics dataset: {cfg}')
