import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce


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


class Score(Model):
    sigma_max = 25
    eps = 1e-5

    def forward(self, time, state):
        return super().forward(time, state) / self.sigma(time, self.sigma_max)

    def sigma(self, t, sigma):
        return torch.sqrt(
            (sigma**(2 * t) - 1)
            / 2
            / np.log(sigma)
        )

    def loss(self, state):
        diffusion_time = torch.rand((state.shape[0], 1), device=state.device) * (1 - self.eps) + self.eps
        noise = torch.randn_like(state)
        std = self.sigma(diffusion_time, self.sigma_max)
        noised_state = state + noise * std
        predicted_score = self(diffusion_time, noised_state)
        return reduce(
            (predicted_score * std + noise)**2,
            'batch dim -> batch', 'sum'
        ).mean()

    @torch.no_grad
    def sample(self, current_states, time_step_count=1000):
        times = torch.linspace(1., self.eps, time_step_count, device=current_states.device)
        time_step_size = times[1] - times[0]
        state = torch.randn_like(current_states) * self.sigma(torch.ones(1, device=current_states.device), self.sigma_max)
        for t in times:
            score = self(t, state)

            # why use mean? like RMSE?
            score_norm = reduce(score**2, 'batch dim -> batch', 'mean').sqrt()
            norm_max = 50
            score_rescaling = torch.ones_like(score)
            score_rescaling[score_norm > norm_max] = norm_max / score_rescaling[score_norm > norm_max]
            score = score * score_rescaling

            g = self.sigma_max**t
            state_drift = state + g**2 * score * time_step_size
            state = state_drift + g * torch.randn_like(state) * time_step_size.sqrt()

        # no noise in final step
        return state_drift

