import logging

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat


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
    def __init__(self, state_dimension, embedding_dimension, hidden_residual_blocks, use_batch_norm):
        super().__init__()

        if embedding_dimension % 2 != 0:
            raise ValueError(
                'The embedding dimension must be even because it is twice the number of frequencies for the Gaussian Fourier projection of the time embedding.'
                f' The embedding dimension specified is {embedding_dimension}.'
            )

        self.embed_time = nn.Sequential(
            GaussianFourierProjection(frequency_count=embedding_dimension // 2),
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.SiLU(),
        )
        self.embed_state = nn.Linear(state_dimension, embedding_dimension)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(embedding_dimension, use_batch_norm)
            for _ in range(hidden_residual_blocks)
        ])
        self.unembed_state = nn.Linear(embedding_dimension, state_dimension)


    def forward(self, time, state):
        embedded_time = self.embed_time(time)

        embedded_state = self.embed_state(state)
        for residual_block in self.residual_blocks:
            embedded_state = embedded_time + residual_block(embedded_state)
        unembedded_state =  self.unembed_state(embedded_state)

        return unembedded_state

    def get_optimizer(self, time_step):
        raise NotImplementedError()


class Score(Model):
    sigma_max = 25
    eps = 1e-3

    def forward(self, time, state):
        return super().forward(time, state) / self.sigma(time, self.sigma_max)

    def get_optimizer(self, time_step):
        lr = 0.005 if time_step == 0 else 0.01
        return torch.optim.Adam(self.parameters(), lr=lr)

    def sigma(self, t, sigma):
        return torch.sqrt(
            (sigma**(2 * t) - 1)
            / 2
            / np.log(sigma)
        )

    def loss(self, state, observation):
        diffusion_time = torch.rand((state.shape[0], 1), device=state.device) * (1 - self.eps) + self.eps
        noise = torch.randn_like(state)
        std = self.sigma(diffusion_time, self.sigma_max)
        noised_state = state + noise * std
        predicted_score = self(diffusion_time, noised_state)
        return reduce(
            (predicted_score * std + noise)**2,
            'batch dim -> batch', 'sum'
        ).mean()

    def observation_likelihood_score_damping(self, t):
        return (1 - 2 * t).clamp(min=0)

    @torch.no_grad
    def sample(self, current_states, observation, time_step_count=600):
        diffusion_times = torch.linspace(1., self.eps, time_step_count, device=current_states.device)
        time_step_size = diffusion_times[0] - diffusion_times[1]
        state = torch.randn_like(current_states) * self.sigma(torch.ones(1, device=current_states.device), self.sigma_max)
        for t in diffusion_times[:-1]:
            score = self(t, state)
            # why use mean? like RMSE?
            score_norm = reduce(score**2, 'batch dim -> batch', 'mean').sqrt()
            norm_max = 50
            score_rescaling = torch.ones_like(score)
            score_rescaling[score_norm > norm_max] = norm_max / score_rescaling[score_norm > norm_max]
            score = score * score_rescaling

            if observation is None:
                observation_score = 0.
            else:
                observation_noise_std = 0.1
                # this seems backwards; should it be (observation - state)? because we are given state?
                observation_score = -(state - observation) / observation_noise_std**2

            score = score + observation_score * self.observation_likelihood_score_damping(t)

            g = self.sigma_max**t
            state_drift = state + g**2 * score * time_step_size
            state = state_drift + g * torch.randn_like(state) * time_step_size.sqrt()

        # no noise in final step
        log.info(reduce(state_drift, 'batch dim -> dim', 'mean'))
        return state_drift


class FlowMatching(Model):
    def forward(self, time, state):
        return super().forward(time, state)

    def get_optimizer(self, time_step):
        lr = 0.005 if time_step == 0 else 0.01
        return torch.optim.Adam(self.parameters(), lr=lr)

    def loss(self, state, observation):
        batch_size = state.shape[0]
        diffusion_time = torch.rand((batch_size, 1), device=state.device)
        noise = torch.randn_like(state)
        target_velocity = state[None] - noise[:, None]
        noise_flowed_to_t = rearrange(
            (diffusion_time * state)[None] + ((1 - diffusion_time) * noise)[:, None],
            'batch predicted_state_count dim -> (batch predicted_state_count) dim'
        )
        diffusion_time = repeat(diffusion_time, 'batch dim -> (repeat batch) dim', repeat=batch_size)
        predicted_velocity = rearrange(
            self(diffusion_time, noise_flowed_to_t),
            '(batch predicted_state_count) dim -> batch predicted_state_count dim',
            batch=batch_size
        )
        if observation is None:
            weighting = 1.
        else:
            observation_noise_std = 0.1
            weighting = torch.exp(
                - 0.5 * reduce(
                    (observation - state)**2,
                    'predicted_state_count dim -> 1 predicted_state_count 1', 'sum'
                ) / observation_noise_std**2
            )
        return reduce(
            weighting * (predicted_velocity - target_velocity)**2,
            'batch predicted_state_count dim -> batch predicted_state_count', 'sum'
        ).mean()

    @torch.no_grad
    def sample(self, current_states, observation, time_step_count=600):
        diffusion_times = torch.linspace(0., 1., time_step_count, device=current_states.device)
        time_step_size = diffusion_times[1] - diffusion_times[0]
        state = torch.randn_like(current_states)
        for t_now, t_next in zip(diffusion_times, diffusion_times[1:]):
            # state = state + time_step_size * self(t_now, state)
            xdot_now = self(t_now, state)
            temp = state + time_step_size * xdot_now
            state = state + time_step_size * (xdot_now + self(t_next, temp)) / 2
        log.info(reduce(state, 'batch dim -> dim', 'mean'))

        return state
