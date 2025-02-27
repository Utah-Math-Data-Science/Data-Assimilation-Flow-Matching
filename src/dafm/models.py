import logging

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

import conf.models
import conf.diffusion_path


log = logging.getLogger(__file__)


class Identity:
    def sample(self, current_states):
        return current_states


class GaussianFourierProjection(nn.Module):
    def __init__(self, frequency_count, frequency_std=30.):
        super().__init__()
        # initialize a random distribution of frequencies that are fixed in training
        self.frequencies = nn.Parameter(
            torch.randn((1, frequency_count)) * frequency_std,
            requires_grad=False
        )

    def forward(self, time):
        angle = 2 * torch.pi * self.frequencies * time
        return rearrange(
            [angle.sin(), angle.cos()],
            'component batch frequency_count -> batch (component frequency_count)'
        )


class ResidualBlock(nn.Module):
    def __init__(self, dimension, use_batch_norm):
        super().__init__()
        self.layer = nn.Linear(dimension, dimension)
        self.activation = nn.SiLU()
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(dimension)

    def forward(self, x):
        x = x + self.activation(self.layer(x))
        if self.use_batch_norm:
            x = self.batch_norm(x)
        return x


class Model(nn.Module):
    def __init__(self, cfg, state_dimension, observation_std):
        super().__init__()
        self.cfg = cfg
        self.state_dimension = state_dimension
        self.observation_std = observation_std
        if cfg.embedding_dimension % 2 != 0:
            raise ValueError(
                'The embedding dimension must be even because it is twice the number of frequencies for the Gaussian Fourier projection of the time embedding.'
                f' The embedding dimension specified is {cfg.embedding_dimension}.'
            )

        self.embed_time = nn.Sequential(
            GaussianFourierProjection(frequency_count=cfg.embedding_dimension // 2),
            nn.Linear(cfg.embedding_dimension, cfg.embedding_dimension),
            nn.SiLU(),
        )
        self.embed_state = nn.Linear(state_dimension, cfg.embedding_dimension)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(cfg.embedding_dimension, cfg.use_batch_norm)
            for _ in range(cfg.residual_block_count)
        ])
        self.unembed_state = nn.Linear(cfg.embedding_dimension, state_dimension)


    def forward(self, time, state):
        embedded_time = self.embed_time(time)

        embedded_state = self.embed_state(state)
        for residual_block in self.residual_blocks:
            embedded_state = embedded_time + residual_block(embedded_state)
        unembedded_state =  self.unembed_state(embedded_state)

        return unembedded_state

    def get_optimizer(self, time_step):
        raise NotImplementedError()


class ScoreMatching(Model):
    def forward(self, time, state):
        return super().forward(time, state) / self.sigma(time)

    def get_optimizer(self, time_step, ignore_observation):
        if self.cfg.train_on_initial_predicted_state and time_step == 0 and ignore_observation:
            lr = 5e-3
        else:
            lr = 1e-2
        return torch.optim.Adam(self.parameters(), lr=lr)

    def sigma(self, t):
        """
        Eqn.(30) of [1]_ modified to be continuous as t -> 0.
        But, we divide by 2*log(sigma_max/sigma_min)? Why?

        References
        ----------
        .. [1] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
           Score-Based Generative Modeling through Stochastic Differential Equations
           (No. arXiv:2011.13456). arXiv. http://arxiv.org/abs/2011.13456
        """
        return self.cfg.diffusion_path.sigma_min * (
            ((self.cfg.diffusion_path.sigma_max / self.cfg.diffusion_path.sigma_min)**(2 * t) - 1)
            / 2
            / np.log(self.cfg.diffusion_path.sigma_max / self.cfg.diffusion_path.sigma_min)
        )**(1/2)

    def loss(self, state, observation):
        # diffusion_time = torch.rand((state.shape[0], 1), device=state.device) * (1 - 1e-5) + 1e-5
        diffusion_time = torch.rand((state.shape[0], 1), device=state.device) * (1 - self.cfg.diffusion_path.time_min) + self.cfg.diffusion_path.time_min
        noise = torch.randn_like(state)
        std = self.sigma(diffusion_time)
        noised_state = state + noise * std
        predicted_score = self(diffusion_time, noised_state)
        return reduce(
            (predicted_score * std + noise)**2,
            'batch dim -> batch', 'sum'
        ).mean()

    def observation_likelihood_score_damping(self, t):
        return (1 - 2 * t).clamp(min=0)

    @torch.no_grad
    def sample(self, current_states, observation, time_step_count=None):
        time_step_count = time_step_count or self.cfg.sampling_time_step_count
        diffusion_times = torch.linspace(1., self.cfg.diffusion_path.time_min, time_step_count, device=current_states.device)
        minus_time_step_size = diffusion_times[1] - diffusion_times[0]
        state = torch.randn_like(current_states) * self.sigma(1)
        for t in diffusion_times[:-1]:
            score = self(t, state)
            if observation is None or self.cfg.ignore_observations:
                observation_score = 0.
            else:
                # this seems backwards; should it be (observation - state)? because we are given state?
                observation_score = -(state - observation) / self.observation_std**2
            score = score + observation_score * self.observation_likelihood_score_damping(t)

            # why use mean? like RMSE?
            score_norm = reduce(score**2, 'batch dim -> batch', 'mean').sqrt()
            score_norm_too_large = score_norm > self.cfg.sampling_max_score_norm
            score[score_norm_too_large] = score[score_norm_too_large] * rearrange(
                self.cfg.sampling_max_score_norm / score_norm[score_norm_too_large],
                'batch -> batch 1'
            )

            # this is sqrt(d/dt (self.sigma(t))^2)
            g = self.cfg.diffusion_path.sigma_min * (self.cfg.diffusion_path.sigma_max / self.cfg.diffusion_path.sigma_min)**t
            if self.cfg.sampler is conf.models.Sampler.EULER:
                state = state - g**2 * score * minus_time_step_size
                state_out = state
            elif self.cfg.sampler is conf.models.Sampler.EULER_MARUYAMA:
                state_drift = state - g**2 * score * minus_time_step_size
                state = state_drift + g * torch.randn_like(state) * minus_time_step_size.abs().sqrt()
                state_out = state_drift
            else:
                raise ValueError(f'Unsupported sampler for {self.__class__.__name__}: {self.cfg.sampler}')

        # no noise in final step
        # maybe this is done to handle the discontinuity of the variance exploding path as t -> 0.
        # just use the mean of the noise distribution for the last step
        return state_out


class FlowMatching(Model):
    def forward(self, time, state):
        return super().forward(time, state)

    def get_optimizer(self, time_step, ignore_observation):
        if self.cfg.train_on_initial_predicted_state and time_step == 0 and ignore_observation:
            lr = 5e-3
        else:
            lr = 1e-2
        return torch.optim.Adam(self.parameters(), lr=lr)

    def sigma(self, t):
        """
        Eqn.(30) of [1]_ modified to be continuous as t -> 0.
        But, we divide by 2*log(sigma_max/sigma_min)? Why?

        References
        ----------
        .. [1] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
           Score-Based Generative Modeling through Stochastic Differential Equations
           (No. arXiv:2011.13456). arXiv. http://arxiv.org/abs/2011.13456
        """
        return self.cfg.diffusion_path.sigma_min * (
            ((self.cfg.diffusion_path.sigma_max / self.cfg.diffusion_path.sigma_min)**(2 * t) - 1)
            / 2
            / np.log(self.cfg.diffusion_path.sigma_max / self.cfg.diffusion_path.sigma_min)
        )**(1/2)

    def dsigma(self, t):
        return self.cfg.diffusion_path.sigma_min * (
            self.cfg.diffusion_path.sigma_max / self.cfg.diffusion_path.sigma_min
        )**(2 * t) / (
            ((self.cfg.diffusion_path.sigma_max / self.cfg.diffusion_path.sigma_min)**(2 * t) - 1)
            * 2
            / np.log(self.cfg.diffusion_path.sigma_max / self.cfg.diffusion_path.sigma_min)
        )**(1/2)

    def loss(self, state, observation):
        batch_size = self.cfg.loss_sample_count
        if batch_size != 1 and batch_size != state.shape[0]:
            # model.batch_size <= predicted_state_count
            raise ValueError(f'model.loss_sample_count can only be 1 or model.batch_size={state.shape[0]}, not {batch_size}.')
        noise = rearrange(torch.randn_like(state[:batch_size]), 'batch dim -> batch 1 dim')
        state = rearrange(state, 'predicted_state_count dim -> 1 predicted_state_count dim')

        if isinstance(self.cfg.diffusion_path, conf.diffusion_path.ConditionalOptimalTransport):
            diffusion_time = torch.rand(batch_size, device=state.device)
            noise_flowed_to_t = (
                state * rearrange(diffusion_time, 'batch -> 1 batch 1')
                + noise * rearrange(1 - diffusion_time, 'batch -> batch 1 1')
            )
            target_velocity = state - noise
            diffusion_path_weighting = 1.
        elif isinstance(self.cfg.diffusion_path, conf.diffusion_path.VarianceExploding):
            diffusion_time = torch.rand((batch_size, 1, 1), device=state.device) * (1 - self.cfg.diffusion_path.time_min) + self.cfg.diffusion_path.time_min
            std = self.sigma(1 - diffusion_time)
            noise_flowed_to_t = state + noise * std
            dt_std = self.dsigma(1 - diffusion_time)
            dx_target_velocity = -dt_std / std
            target_velocity = dx_target_velocity * (noise_flowed_to_t - state)
            diffusion_path_weighting = 1 / dt_std**2
            diffusion_time = rearrange(diffusion_time, 'batch 1 1 -> batch')
        else:
            raise ValueError(f'Unknown diffusion path: {self.cfg.diffusion_path}')

        predicted_velocity = rearrange(
            self(
                repeat(diffusion_time, 'batch -> (repeat batch) 1', repeat=batch_size),
                rearrange(noise_flowed_to_t, 'batch predicted_state_count dim -> (batch predicted_state_count) dim')
            ),
            '(batch predicted_state_count) dim -> batch predicted_state_count dim',
            batch=batch_size
        )

        if observation is None or self.cfg.ignore_observations:
            weighting = 1.
            return reduce(
                weighting * diffusion_path_weighting * (predicted_velocity - target_velocity)**2,
                'batch predicted_state_count dim -> batch predicted_state_count', 'sum'
            ).mean()
        else:
            weighting_argument = -0.5 * reduce(
                (observation - state)**2,
                '1 predicted_state_count dim -> 1 predicted_state_count 1', 'sum'
            ) / self.observation_std**2
            if self.cfg.softmax_loss_weighting:
                weighting = torch.softmax(weighting_argument, dim=1)
            else:
                weighting = torch.exp(weighting_argument)
            return reduce(
                weighting * diffusion_path_weighting * (predicted_velocity - target_velocity)**2,
                'batch predicted_state_count dim -> batch predicted_state_count', 'sum'
            ).sum(1).mean()

    def observation_likelihood_vector_field_damping(self, t):
        return t
        return 1.

    def observation_likelihood_score_damping(self, t):
        return (1 - 2 * t).clamp(min=0)

    @torch.no_grad
    def sample(self, current_states, observation, time_step_count=None):
        time_step_count = time_step_count or self.cfg.sampling_time_step_count
        if isinstance(self.cfg.diffusion_path, conf.diffusion_path.ConditionalOptimalTransport):
            diffusion_times = torch.linspace(0., 1., time_step_count, device=current_states.device)
            state = torch.randn_like(current_states)
        elif isinstance(self.cfg.diffusion_path, conf.diffusion_path.VarianceExploding):
            diffusion_times = torch.linspace(0., 1. - self.cfg.diffusion_path.time_min, time_step_count, device=current_states.device)
            state = torch.randn_like(current_states) * self.sigma(1)
        else:
            raise ValueError(f'Unknown diffusion path: {self.cfg.diffusion_path}')
        time_step_size = diffusion_times[1] - diffusion_times[0]

        if (
            self.cfg.sampling_use_observation_likelihood
            and observation is not None
            and not self.cfg.ignore_observations
        ):
            dot_state = lambda t, x: (
                self(t, x)
                + self.observation_likelihood_vector_field_damping(t) * (
                    (observation - x) / (1 - t)
                )
            ) / 2
        else:
            dot_state = self
        for t_now, t_next in zip(diffusion_times, diffusion_times[1:]):
            if self.cfg.sampler is conf.models.Sampler.EULER:
                state = state + time_step_size * dot_state(t_now, state)
                state_out = state
            elif self.cfg.sampler is conf.models.Sampler.EULER_MARUYAMA:
                if not isinstance(self.cfg.diffusion_path, conf.diffusion_path.VarianceExploding):
                    raise ValueError(
                        f'The Euler-Maruyama sampler is only supported with the variance exploding diffusion path, not {self.cfg.diffusion_path.__class__.__name__}.'
                        ' Please use a different sampler (e.g., set model.sampler=EULER), or use a diffusion diffusion path (e.g., set model/diffusion_path=ConditionalOptimalTransport).'
                    )
                # this is sqrt(d/dt (self.sigma(t))^2) evaluated at 1 - t_now
                g = self.cfg.diffusion_path.sigma_min * (self.cfg.diffusion_path.sigma_max / self.cfg.diffusion_path.sigma_min)**(1 - t_now)
                if self.cfg.sampling_use_observation_likelihood_score:
                    score = dot_state(t_now, state) / g**2
                    if observation is None or self.cfg.ignore_observations:
                        observation_score = 0.
                    else:
                        # this seems backwards; should it be (observation - state)? because we are given state?
                        observation_score = -(state - observation) / self.observation_std**2
                    score = score + observation_score * self.observation_likelihood_score_damping(1 - t_now)
                    state_drift = state + g**2 * score * time_step_size
                else:
                    state_drift = state + time_step_size * dot_state(t_now, state)
                state = state_drift + g * torch.randn_like(state) * time_step_size.sqrt()
                state_out = state_drift
            elif self.cfg.sampler is conf.models.Sampler.HEUN:
                if self.cfg.sampling_use_observation_likelihood:
                    raise ValueError(
                        'Using the observation likelihood vector field does not work with the Heun sampler yet.'
                        ' Please use a different sampler (e.g., set model.sampler=EULER), or set model.sampleing_use_observation_likelihood=false.'
                    )
                xdot_now = dot_state(t_now, state)
                temp = state + time_step_size * xdot_now
                state = state + time_step_size * (xdot_now + dot_state(t_next, temp)) / 2
                state_out = state
            else:
                raise ValueError(f'Unsupported sampler for {self.__class__.__name__}: {self.cfg.sampler}')

        return state_out


def get_model(cfg, state_dimension, observation_std):
    if isinstance(cfg, conf.models.ScoreMatching):
        return ScoreMatching(cfg, state_dimension, observation_std)
    elif isinstance(cfg, conf.models.FlowMatching):
        return FlowMatching(cfg, state_dimension, observation_std)
    else:
        raise ValueError(f'Unknown model: {cfg}')
