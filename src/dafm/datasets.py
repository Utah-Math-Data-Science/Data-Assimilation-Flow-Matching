import copy
from collections import deque

import numpy as np
import torch
from torchdiffeq import odeint
import lightning.pytorch as pl
import dapper.mods.KS
from einops import rearrange, unpack
from tqdm import tqdm

import conf.dataset
import conf.conf
import dafm.utils


log = dafm.utils.getLoggerByFilename(__file__)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitems__(self, idx):
        return self.tensor[idx]


class DynamicalSystem(pl.LightningDataModule):
    def __init__(
        self,
        cfg: conf.dataset.DynamicalSystemImpl, rng: np.random.Generator,
        device,
        batch_size=1,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        (
            self.rng_dataset_shuffle,
            self.rng_initial_state,
        ) = rng.spawn(2)
        self.batch_size = batch_size
        self.states = None

    def should_prepare_data(self):
        should_prepare = True
        use_existing = False

        if self.cfg.processed_filepath.exists():
            use_existing = True
            with conf.conf.Session() as db:
                max_time_step_count = (
                    db.query(conf.dataset.DynamicalSystemImpl.time_step_count)
                    .where(
                        conf.dataset.DynamicalSystemImpl.processed_code == self.cfg.processed_code,
                    )
                    .order_by(conf.dataset.DynamicalSystemImpl.time_step_count.desc())
                    .first()
                )[0]
            should_prepare = self.cfg.recompute_trajectory or self.cfg.time_step_count > max_time_step_count

        return should_prepare, use_existing

    def prepare_data(self):
        should_prepare, use_existing = self.should_prepare_data()
        if not should_prepare:
            return
        log.info(f'Computing {type(self).__name__}({self.cfg.processed_code})')
        saved_states = self.load_states().to(self.device) if use_existing else None

        states = deque()
        if saved_states is None:
            state = self.initialize_true_state(self.cfg, self.device)
            states.append(state)
            time_step_start = len(states)
        else:
            state = saved_states[-1:]
            time_step_start = len(saved_states)

        states.extend(self.integrate(time_step_start, self.cfg.time_step_count + 1, state)[1:])

        if saved_states is not None:
            states.appendleft(saved_states)

        try:
            states = torch.cat(list(states))
        except torch.OutOfMemoryError:
            states = torch.cat([s.cpu() for s in states])

        assert states.shape[0] == self.cfg.time_step_count + 1, f'{states.shape=}'
        self.save_states(states)

    def integrate(self, t0, t1, state0):
        state = state0
        states = [state0]
        for t in range(t0, t1):
            state = self.step_state(
                t * self.cfg.time_step_size, (t + 1) * self.cfg.time_step_size, state
            )
            states.append(state)
        return states

    def unflatten_integrate(self, t0, t1, state0):
        return [
            state.reshape(state0.shape)
            for state in self.integrate(
                t0, t1, state0.view(-1, self.cfg.channels, *self.cfg.spatial_dims)
            )
        ]

    def load_states(self):
        return torch.load(self.cfg.processed_filepath, weights_only=True)

    def save_states(self, states):
        dafm.utils.torch_save_once_atomic(states.cpu(), self.cfg.processed_filepath)

    def initialize_true_state(self, cfg, device):
        raise NotImplementedError()

    def setup(self, stage, force_state_reload=False):
        if self.states is None or force_state_reload:
            self.states = self.load_states()
            if self.states.shape[0] < self.cfg.time_step_count + 1:
                raise RuntimeError(
                    f"Expected the saved dynamical system trajectory to have at least {self.cfg.time_step_count} time steps plus the initial condition, "
                    f"but it has {self.states.shape[0] - 1} time steps plus the initial condition. "
                    f"The trajectory needs to be recomputed up to {self.cfg.time_step_count} time steps. "
                    f"To recompute the trajectory, please set setting.dataset.recompute_trajectory=true or delete this file: {self.cfg.processed_filepath}"
                )

    def __getitem__(self, idx):
        return self.states[idx]

    def __getitems__(self, idx):
        return self.states[idx]

    def subset(self, idx):
        states = self.states[idx]
        new_self = copy.copy(self)
        new_self.states = states
        return new_self

    def __repr__(self):
        class_name = type(self).__name__
        return f"{class_name}({len(self.states) if self.states is not None else ''})"

    def step_state(self, t0, t1, state0):
        raise NotImplementedError()

    def get_noise(self, shape, rng, std, **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = torch.float32
        if std > 0:
            noise = torch.tensor(rng.normal(scale=std, size=shape), **kwargs)
        else:
            noise = 0.
        return noise

    def linearize(self, x):
        raise NotImplementedError()


class UnitTestDataset(DynamicalSystem):
    """
    Dataset for testing the dataset loading and splitting logic without saving
    data to disk.
    """
    def should_prepare_data(self):
        """
        Redefine this in the unit tests.
        """
        return True, True

    def load_states(self):
        if hasattr(self, 'states'):
            return self.states
        return torch.arange(self.cfg.time_step_count // 2)[:, None, None, None].expand(
            (self.cfg.time_step_count // 2, self.cfg.channels, *self.cfg.spatial_dims)
        ) * 1.0

    def save_states(self, states):
        self.states = states

    def initialize_true_state(self, cfg, device):
        return torch.arange(1)[:, None, None, None].expand(
            (1, self.cfg.channels, *self.cfg.spatial_dims)
        ) * 1.0

    def step_state(self, t0, t1, state0):
        return state0 + 1


class Rotation2D(DynamicalSystem):
    def __init__(self, cfg, rng: np.random.Generator, device):
        (
            rng_rotation,
            rng_super,
        ) = rng.spawn(2)
        super().__init__(cfg, rng_super, device)
        rotation_angle = torch.tensor(2 * np.pi * rng.random(), dtype=torch.float32)
        self.rotation = torch.tensor([
            [rotation_angle.cos(), -rotation_angle.sin()],
            [rotation_angle.sin(), rotation_angle.cos()],
        ], device=device)

    def initialize_true_state(self, cfg, device):
        return torch.tensor([cfg.system.radius, 0], device=device)[None, None]

    def step_state(self, t0, t1, state0):
        return state0 @ self.rotation.T

    def linearize(self, x):
        return self.rotation


class NavierStokes2DBackwardFacingStepGLED(DynamicalSystem):
    def prepare_data(self):
        if self.cfg.processed_filepath.exists():
            return
        self.cfg.processed_filepath.symlink_to(self.cfg.system.gled_filepath)

    def load_states(self):
        return torch.load(self.cfg.processed_filepath).squeeze(0)


class NavierStokes2DPeriodicBoundary(DynamicalSystem):
    def __init__(self, cfg, rng, device):
        super().__init__(cfg, rng, device)
        horizontal_axis = torch.linspace(0, cfg.system.grid_width, cfg.system.grid_horizontal_count + 1, device=device)[:-1]
        vertical_axis = torch.linspace(0, cfg.system.grid_height, cfg.system.grid_vertical_count + 1, device=device)[:-1]
        self.grid_horizontal_spacing = (horizontal_axis[1] - horizontal_axis[0]).item()
        self.grid_vertical_spacing = (vertical_axis[1] - vertical_axis[0]).item()

        horizontal_grid, vertical_grid = torch.meshgrid(horizontal_axis, vertical_axis, indexing='ij')

        self.horizontal_forcing = cfg.system.forcing_amplitude * torch.sin(2 * torch.pi * cfg.system.vertical_mode_number / cfg.system.grid_height * vertical_grid)
        self.vertical_forcing = torch.zeros_like(self.horizontal_forcing)

    def initialize_true_state(self, cfg, device):
        # linspace excluding endpoint
        horizontal_axis = torch.linspace(0, cfg.system.grid_width, cfg.system.grid_horizontal_count + 1, device=device)[:-1]
        vertical_axis = torch.linspace(0, cfg.system.grid_height, cfg.system.grid_vertical_count + 1, device=device)[:-1]

        horizontal_grid, vertical_grid = torch.meshgrid(horizontal_axis, vertical_axis, indexing='ij')

        horizontal_velocity = self.sample_squared_exponential_gaussian_process(cfg, device, horizontal_grid.dtype)
        vertical_velocity = self.sample_squared_exponential_gaussian_process(cfg, device, horizontal_grid.dtype)
        b0 = self.divergence(horizontal_velocity, vertical_velocity) / cfg.time_step_size
        pressure = torch.zeros((cfg.system.grid_horizontal_count, cfg.system.grid_vertical_count), device=device)
        pressure  = self.pressure_poisson(pressure, b0)
        dhorizontal_pressure, dvertical_pressure = self.gradient(pressure)
        horizontal_velocity = horizontal_velocity - cfg.time_step_size * dhorizontal_pressure
        vertical_velocity = vertical_velocity - cfg.time_step_size * dvertical_pressure

        x = rearrange(
            [pressure, horizontal_velocity, vertical_velocity],
            'value_count grid_horizontal_count grid_vertical_count -> 1 value_count grid_horizontal_count grid_vertical_count'
        )

        self.horizontal_forcing = cfg.system.forcing_amplitude * torch.sin(2 * torch.pi * cfg.system.vertical_mode_number / cfg.system.grid_height * vertical_grid)
        self.vertical_forcing = torch.zeros_like(self.horizontal_forcing)

        return x

    def sample_squared_exponential_gaussian_process(self, cfg, device, dtype):
        w = self.rng_initial_state.normal(size=(cfg.system.grid_horizontal_count, cfg.system.grid_vertical_count))
        w_hat = np.fft.fft2(w)
        kx = np.fft.fftfreq(cfg.system.grid_horizontal_count, d=self.grid_horizontal_spacing)
        ky = np.fft.fftfreq(cfg.system.grid_vertical_count, d=self.grid_vertical_spacing)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        S = np.exp(-2 * np.pi**2 * cfg.system.gaussian_process_length_scale**2 * (KX**2 + KY**2))
        f = np.fft.ifft2(w_hat * np.sqrt(S)).real
        out = cfg.system.gaussian_process_std * (f - f.mean()) / f.std()
        return torch.tensor(out, device=device, dtype=dtype)

    def step_state(self, t0, t1, state0):
        """
        Chorin's projection method.
        """
        pressure, horizontal_velocity, vertical_velocity = rearrange(
            state0,
            'member value_count grid_horizontal_count grid_vertical_count -> value_count member grid_horizontal_count grid_vertical_count',
            value_count=3, grid_horizontal_count=self.cfg.system.grid_horizontal_count, grid_vertical_count=self.cfg.system.grid_vertical_count,
        )

        # advection
        horizontal_advection = self.advect(horizontal_velocity, vertical_velocity, horizontal_velocity)
        vertical_advection = self.advect(horizontal_velocity, vertical_velocity, vertical_velocity)

        # intermediate velocity + forcing
        horizontal_velocity_predictive = horizontal_velocity + self.cfg.time_step_size * (
            self.cfg.system.viscosity * self.laplacian(horizontal_velocity)
            - horizontal_advection + self.horizontal_forcing
        )
        vertical_velocity_predictive = vertical_velocity + self.cfg.time_step_size * (
            self.cfg.system.viscosity * self.laplacian(vertical_velocity)
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
            'value_count member grid_horizontal_count grid_vertical_count -> member value_count grid_horizontal_count grid_vertical_count'
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
        for _ in range(self.cfg.system.pressure_poisson_solve_iteration_count):
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


class Lorenz63(DynamicalSystem):
    def initialize_true_state(self, cfg, device):
        true_state = self.get_noise((1, self.cfg.channels, *self.cfg.spatial_dims), self.rng_initial_state, 1., device=device)
        return true_state

    def dynamics(self, t, x):
        x = x * self.cfg.system.rescaling
        x1, x2, x3 = unpack(x, self.cfg.spatial_dims[0] * [[]], 'batch channel *')
        dot_x = rearrange([
            self.cfg.system.sigma * (x2 - x1),
            x1 * (self.cfg.system.rho - x3) - x2,
            x1 * x2 - self.cfg.system.beta * x3,
        ], 'dim batch channel -> batch channel dim')
        return dot_x / self.cfg.system.rescaling

    def step_state(self, t0, t1, state0):
        return odeint(
            self.dynamics, state0, torch.tensor([t0, t1], device=state0.device),
            method='rk4', options=dict(step_size=self.cfg.time_step_size),
        )[1]


class Lorenz96(DynamicalSystem):
    def initialize_true_state(self, cfg, device):
        true_state = cfg.system.true_state_initial_condition_mean + self.get_noise((self.cfg.channels, *self.cfg.spatial_dims), self.rng_initial_state, cfg.system.true_state_initial_condition_std, device=device)
        # true_state = torch.ones((1, self.cfg.system.dimension), device=device) * self.cfg.system.forcing
        return true_state

    def dynamics(self, t, x):
        x_p1 = x.roll(-1, -1)
        x_m2 = x.roll(2, -1)
        x_m1 = x.roll(1, -1)
        return (x_p1 - x_m2) * x_m1 - x + self.cfg.system.forcing

    def step_state(self, t0, t1, state0):
        return odeint(
            self.dynamics, state0, torch.tensor([t0, t1], device=state0.device),
            method='rk4', options=dict(step_size=self.cfg.time_step_size),
        )[1]


class KuramotoSivashinsky(DynamicalSystem):
    def __init__(self, cfg, rng, device):
        super().__init__(cfg, rng, device)
        self.dtype = torch.float32
        self.solve = self.etd_rk4_wrapper(cfg, device)

    def initialize_true_state(self, cfg, device):
        x0 = torch.tensor(dapper.mods.KS.Model(
            dt=cfg.time_step_size,
            DL=cfg.system.domain_pi_multiple,
            Nx=cfg.system.dimension,
        ).x0, device=device, dtype=self.dtype)
        x0 = rearrange(x0, 'dim -> 1 dim')
        return x0 + self.get_noise(x0.shape, self.rng_initial_state, 1., device=device)

    def step_state(self, t0, t1, state0):
        return self.solve(None, state0)

    def etd_rk4_wrapper(self, cfg, device):
        """ Returns an ETD-RK4 integrator for the KS equation. Currently very specific, need
        to adjust this to fit into the same framework as the ODE integrators

        Directly ported from https://github.com/nansencenter/DAPPER/blob/master/dapper/mods/KS/core.py
        which is adapted from kursiv.m of Kassam and Trefethen, 2002, doi.org/10.1137/S1064827502410633.
        """
        kk = np.append(np.arange(0, cfg.system.dimension / 2), 0) * 2 / cfg.system.domain_pi_multiple  # wave nums for rfft
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
        Q = torch.tensor(h * ((np.exp(CL / 2) - 1) / CL).mean(axis=-1).real, device=device, dtype=self.dtype).unsqueeze(0)
        # RK4 coefficients (modified by Cox-Matthews):
        f1 = torch.tensor(h * ((-4 - CL + np.exp(CL) * (4 - 3 * CL + CL ** 2)) / CL ** 3).mean(axis=-1).real, device=device, dtype=self.dtype).unsqueeze(0)
        f2 = torch.tensor(h * ((2 + CL + np.exp(CL) * (-2 + CL)) / CL ** 3).mean(axis=-1).real, device=device, dtype=self.dtype).unsqueeze(0)
        f3 = torch.tensor(h * ((-4 - 3 * CL - CL ** 2 + np.exp(CL) * (4 - CL)) / CL ** 3).mean(axis=-1).real, device=device, dtype=self.dtype).unsqueeze(0)

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


def get_dataset(cfg: conf.dataset.DynamicalSystemImpl, rng, device):
    match cfg.system:
        case conf.dataset.Lorenz63():
            return Lorenz63(cfg, rng, device)
        case conf.dataset.Lorenz96():
            return Lorenz96(cfg, rng, device)
        case conf.dataset.KuramotoSivashinsky():
            return KuramotoSivashinsky(cfg, rng, device)
        case conf.dataset.NavierStokes2DPeriodicBoundary():
            return NavierStokes2DPeriodicBoundary(cfg, rng, device)
        case conf.dataset.NavierStokes2DBackwardFacingStepGLED():
            return NavierStokes2DBackwardFacingStepGLED(cfg, rng, device)
        case conf.dataset.UnitTestDataset():
            return UnitTestDataset(cfg, rng, device)
        case conf.dataset.Rotation2D():
            return Rotation2D(cfg, rng, device)
        case _:
            raise ValueError(f'Unknown dataset: {cfg}')
