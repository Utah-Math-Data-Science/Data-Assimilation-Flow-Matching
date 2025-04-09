import logging

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

import conf.models
import conf.diffusion_path
import dafm.diffusion_path
from dafm import flow_matching_guidance, utils


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
    def __init__(self, cfg, state_dimension, observation_std, diffusion_path):
        super().__init__()
        self.cfg = cfg
        self.state_dimension = state_dimension
        self.observation_std = observation_std
        self.diffusion_path = diffusion_path
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
            lr = self.cfg.learning_rate_when_training_on_initial_predicted_state
        else:
            lr = self.cfg.learning_rate
        return torch.optim.Adam(self.parameters(), lr=lr)

    def loss(self, state, observation, observe):
        diffusion_time = self.diffusion_path.sample_time((state.shape[0], 1), device=state.device)
        noise = torch.randn_like(state)
        std = self.std(diffusion_time, state)
        noised_state = state + noise * std
        predicted_score = self(diffusion_time, noised_state)
        return dict(
            loss=reduce(
                (predicted_score * std + noise).square(),
                'batch dim -> batch', 'sum'
            ).mean(),
        )

    def observation_likelihood_score_damping(self, t):
        return (1 - 2 * t).clamp(min=0)

    @torch.no_grad
    def sample(self, current_states, observation, observe, time_step_count=None):
        time_step_count = time_step_count or self.cfg.sampling_time_step_count
        diffusion_time = self.diffusion_path.linspace_time(time_step_count, device=current_states.device)
        minus_time_step_size = diffusion_time[1] - diffusion_time[0]
        minus_time_step_size_abs_sqrt = minus_time_step_size.abs().sqrt()
        xt = torch.randn_like(current_states) * self.std(1)
        for t in diffusion_time[:-1]:
            score = self(t, xt)
            if observation is None or self.cfg.ignore_observations:
                observation_score = 0.
            else:
                with torch.enable_grad():
                    xt_grad = xt.detach().requires_grad_()
                    log_observation_likelihood = -.5 * reduce(
                        (observe(xt_grad) - observation).square(),
                        'predicted_state_count dim ->',
                        'sum'
                    )
                    observation_score, *_ = torch.autograd.grad(
                        outputs=log_observation_likelihood,
                        inputs=xt_grad,
                    )
                # this seems backwards; should it be (observation - state)? because we are given state?
                # observation_score = -(observe(state) - observation) / self.observation_std.square()
            score = score + observation_score * self.observation_likelihood_score_damping(t)

            # why use mean? like RMSE?
            score_norm = reduce(score.square(), 'batch dim -> batch', 'mean').sqrt()
            score_norm_too_large = score_norm > self.cfg.sampling_max_score_norm
            score[score_norm_too_large] = score[score_norm_too_large] * rearrange(
                self.cfg.sampling_max_score_norm / score_norm[score_norm_too_large],
                'batch -> batch 1'
            )

            g = self.diffusion_path.g(t)
            if self.cfg.sampler is conf.models.Sampler.EULER:
                xt = xt - g.square() * score * minus_time_step_size
                state_out = xt
            elif self.cfg.sampler is conf.models.Sampler.EULER_MARUYAMA:
                state_drift = xt - g.square() * score * minus_time_step_size
                xt = state_drift + g * torch.randn_like(xt) * minus_time_step_size_abs_sqrt
                state_out = state_drift
            else:
                raise ValueError(f'Unsupported sampler for {self.__class__.__name__}: {self.cfg.sampler}')

        # no noise in final step
        # maybe this is done to handle the discontinuity of the variance exploding path as t -> 0.
        # just use the mean of the noise distribution for the last step
        return state_out


class FlowMatching(Model):
    def __init__(self, cfg, state_dimension, observation_std, diffusion_path, guidance):
        super().__init__(cfg, state_dimension, observation_std, diffusion_path)
        self.guidance = guidance

    def forward(self, time, state):
        return super().forward(time, state)

    def get_optimizer(self, time_step, ignore_observation):
        if self.cfg.train_on_initial_predicted_state and time_step == 0 and ignore_observation:
            lr = self.cfg.learning_rate_when_training_on_initial_predicted_state
        else:
            lr = self.cfg.learning_rate
        return torch.optim.Adam(self.parameters(), lr=lr)

    @staticmethod
    def diffusion_path_context(cfg, diffusion_path, time_shape, noise, state):
        path_time = diffusion_path.sample_time(time_shape, device=state.device)

        mean = diffusion_path.mean(path_time, state)
        std = diffusion_path.std(path_time, state)
        dt_std = diffusion_path.dt_std(path_time, state)
        noise_flowed_to_t = mean + noise * std
        if isinstance(cfg.diffusion_path, conf.diffusion_path.ConditionalOptimalTransport):
            flow_matching_time = path_time
            target_velocity = state - noise
        elif isinstance(cfg.diffusion_path, conf.diffusion_path.VarianceExploding):
            flow_matching_time = 1 - path_time
            dt_std = -dt_std  # change of variables
            target_velocity = dt_std * noise
        else:
            raise ValueError(f'Unknown diffusion path: {cfg.diffusion_path}')

        diffusion_path_weighting = 1 / dt_std.square()

        return dict(
            flow_matching_time=flow_matching_time,
            noise_flowed_to_t=noise_flowed_to_t,
            target_velocity=target_velocity,
            mean=mean, std=std, dt_std=dt_std,
            diffusion_path_weighting=diffusion_path_weighting,
        )

    @staticmethod
    def divergence_matching_loss(cfg, forward, flow_matching_time, noise_flowed_to_t, target_velocity, mean, std, dt_std):
        predicted_state_count, dim = noise_flowed_to_t.shape
        dx_target_velocity = dt_std / std
        hutchinson_noise = torch.randn((predicted_state_count, dim), device=noise_flowed_to_t.device)
        predicted_velocity, predicted_velocity_jvp = torch.autograd.functional.jvp(
            lambda xt: forward(flow_matching_time, xt),
            noise_flowed_to_t,
            hutchinson_noise,
            create_graph=True,
        )
        predicted_divergence = utils.inner_product(hutchinson_noise, predicted_velocity_jvp)
        dx_log_pt = -(noise_flowed_to_t - mean) / std.square()
        if cfg.divergence_matching_use_hutchinson_trace_for_target_divergence:
            target_divergence = (
                utils.inner_product(hutchinson_noise * dx_target_velocity, hutchinson_noise)
                + utils.inner_product(hutchinson_noise, target_velocity - predicted_velocity) * utils.inner_product(dx_log_pt, hutchinson_noise)
            )
        else:
            target_divergence = (
                dx_target_velocity.reshape(cfg.loss_expectation_sample_count, -1).sum(-1, keepdim=True)
                + utils.inner_product(target_velocity, dx_log_pt)
                - utils.inner_product(predicted_velocity, dx_log_pt)
            )

        divergence_matching_weighting = 1 / dx_target_velocity.abs() / (predicted_state_count * dim)
        divergence_matching_loss = cfg.divergence_matching_loss_coefficient * (
            divergence_matching_weighting * (target_divergence - predicted_divergence).abs()
        ).mean()

        return predicted_velocity, divergence_matching_loss

    def loss(self, state, observation, observe):
        predicted_state_count, dim = state.shape
        if self.cfg.use_expectation_of_sum:
            time_noise_samples_per_expectation_sample = 1
        else:
            time_noise_samples_per_expectation_sample = predicted_state_count

        noise = torch.randn((self.cfg.loss_expectation_sample_count, time_noise_samples_per_expectation_sample, dim), device=state.device)
        state = rearrange(state, 'predicted_state_count dim -> 1 predicted_state_count dim')

        path_context = self.diffusion_path_context(self.cfg, self.diffusion_path, (self.cfg.loss_expectation_sample_count, time_noise_samples_per_expectation_sample, 1), noise, state)
        if self.cfg.use_expectation_of_sum:
            path_context['path_time'] = repeat(path_context['path_time'], 'loss_expectation_sample_count 1 1 -> loss_expectation_sample_count predicted_state_count 1', predicted_state_count=predicted_state_count)
        path_context = {
            k: rearrange(x, 'loss_expectation_sample_count predicted_state_count dim -> (loss_expectation_sample_count predicted_state_count) dim')
            for k, x in path_context.items()
        }

        if self.cfg.use_divergence_matching:
            predicted_velocity, divergence_matching_loss = self.divergence_matching_loss(
                self.cfg,
                self,
                path_context['flow_matching_time'],
                path_context['noise_flowed_to_t'],
                path_context['target_velocity'],
                path_context['mean'],
                path_context['std'],
                path_context['dt_std'],
            )
        else:
            predicted_velocity = self(path_context['flow_matching_time'], path_context['noise_flowed_to_t'])
            divergence_matching_loss = 0.

        predicted_velocity = rearrange(
            predicted_velocity,
            '(loss_expectation_sample_count predicted_state_count) dim -> loss_expectation_sample_count predicted_state_count dim',
            loss_expectation_sample_count=self.cfg.loss_expectation_sample_count,
        )

        if isinstance(self.cfg.guidance, flow_matching_guidance.No) or observation is None or self.cfg.ignore_observations:
            weighting = 1 / predicted_state_count
        else:
            observation_likelihood_distribution = torch.distributions.Independent(
                torch.distributions.Normal(observe(state), self.observation_std),
                1,
            )
            log_observation_likelihood = rearrange(
                observation_likelihood_distribution.log_prob(observation),
                '1 predicted_state_count -> 1 predicted_state_count 1',
            )
            weighting = log_observation_likelihood.softmax(1)

        flow_loss = reduce(
            weighting * path_context['diffusion_path_weighting'] * (predicted_velocity - path_context['target_velocity']).square(),
            'loss_expectation_sample_count predicted_state_count dim -> loss_expectation_sample_count', 'sum'
        ).mean()

        return dict(
            loss=flow_loss + divergence_matching_loss,
            flow_loss=flow_loss,
            divergence_matching_loss=divergence_matching_loss,
        )

    def observation_likelihood_score_damping(self, t):
        return (1 - 2 * t).clamp(min=0)

    @torch.no_grad
    def sample(self, current_states, observation, observe, time_step_count=None):
        time_step_count = time_step_count or self.cfg.sampling_time_step_count
        path_time = self.diffusion_path.linspace_time(time_step_count, device=current_states.device)
        if isinstance(self.cfg.diffusion_path, conf.diffusion_path.ConditionalOptimalTransport):
            flow_matching_time = path_time
            xt = torch.randn_like(current_states)
        elif isinstance(self.cfg.diffusion_path, conf.diffusion_path.VarianceExploding):
            flow_matching_time = 1 - path_time
            xt = torch.randn_like(current_states) * self.diffusion_path.std(1)
        else:
            raise ValueError(f'Unknown diffusion path: {self.cfg.diffusion_path}')

        if (
            observation is not None
            and not self.cfg.ignore_observations
        ):
            noise = xt
            def velocity(t, x):
                dot_state_unguided = self(t, x)
                return dot_state_unguided + self.guidance(
                    t, x, noise, current_states, dot_state_unguided,
                    energy_function=lambda x1_predicted: reduce(
                        (observe(x1_predicted) - observation).square(),
                        'predicted_state_count dim -> predicted_state_count 1',
                        'sum'
                    )
                )
        else:
            velocity = self

        time_step_size = flow_matching_time[1] - flow_matching_time[0]
        time_step_size_sqrt = time_step_size.sqrt()
        for t_now, t_next in zip(flow_matching_time, flow_matching_time[1:]):
            if self.cfg.sampler is conf.models.Sampler.EULER:
                xt = xt + time_step_size * velocity(t_now, xt)
                state_out = xt
            elif self.cfg.sampler is conf.models.Sampler.EULER_MARUYAMA:
                if not isinstance(self.cfg.diffusion_path, conf.diffusion_path.VarianceExploding):
                    raise ValueError(
                        f'The Euler-Maruyama sampler is only supported with the variance exploding diffusion path, not {self.cfg.diffusion_path.__class__.__name__}.'
                        ' Please use a different sampler (e.g., set model.sampler=EULER), or use a diffusion diffusion path (e.g., set model/diffusion_path=ConditionalOptimalTransport).'
                    )
                g = self.diffusion_path.g(1 - t_now)
                state_drift = xt + time_step_size * velocity(t_now, xt)
                xt = state_drift + g * torch.randn_like(xt) * time_step_size_sqrt
                state_out = state_drift
            elif self.cfg.sampler is conf.models.Sampler.HEUN:
                xdot_now = velocity(t_now, xt)
                temp = xt + time_step_size * xdot_now
                xt = xt + time_step_size * (xdot_now + velocity(t_next, temp)) / 2
                state_out = xt
            else:
                raise ValueError(f'Unsupported sampler for {self.__class__.__name__}: {self.cfg.sampler}')

        return state_out


class FlowMatchingMarginal(nn.Module):
    def __init__(self, cfg, diffusion_path, guidance):
        super().__init__()
        self.cfg = cfg
        self.diffusion_path = diffusion_path
        self.guidance = guidance

    def get_optimizer(self, time_step, ignore_observation):
        if self.cfg.train_conditional_vector_field_weights:
            if self.cfg.train_on_initial_predicted_state and time_step == 0 and ignore_observation:
                lr = self.cfg.learning_rate_when_training_on_initial_predicted_state
            else:
                lr = self.cfg.learning_rate
            return torch.optim.Adam(self.parameters(), lr=lr)
        else:
            return None

    def forward(self, time, xt, x0, x1):
        if self.cfg.resample_noise_when_estimating_vector_field:
            x0 = torch.randn_like(x1)
        if isinstance(self.cfg.diffusion_path, conf.diffusion_path.ConditionalOptimalTransport):
            mean = x1 * time
            std = 1 - (1 - self.cfg.diffusion_path.sigma_min) * time
            if self.cfg.use_velocity_of_conditional_flow_map:
                conditional_velocities = rearrange(
                    x1 - x0,
                    'particle_count dim -> particle_count 1 dim',
                )
            else:
                conditional_velocities = (
                    rearrange(x1, 'particle_count dim -> particle_count 1 dim')
                    - (1 - self.cfg.diffusion_path.sigma_min) * rearrange(xt, 'predicted_state_count dim -> 1 predicted_state_count dim')
                ) / std
        elif isinstance(self.cfg.diffusion_path, conf.diffusion_path.VarianceExploding):
            raise NotImplementedError()
            time = time * (1 - self.cfg.diffusion_path.time_min) + self.cfg.diffusion_path.time_min

        conditional_distribution = torch.distributions.Independent(
            torch.distributions.Normal(
                loc=rearrange(mean, 'particle_count dim -> particle_count 1 dim'),
                scale=std
            ),
            1,
        )
        log_pt_given_x1 = rearrange(
            conditional_distribution.log_prob(
                rearrange(xt, 'predicted_state_count dim -> 1 predicted_state_count dim')
            ),
            'particle_count predicted_state_count -> particle_count predicted_state_count 1',
        )
        weights = log_pt_given_x1.softmax(0) * xt.shape[0]
        return reduce(
            weights * conditional_velocities,
            'particle_count predicted_state_count dim -> predicted_state_count dim',
            'mean'
        )

    def loss(self, flow_matching_time, noise, predicted_state, observation, observe):
        path_context = FlowMatching.diffusion_path_context(self.cfg, self.diffusion_path, predicted_state.shape, noise, predicted_state)
        if self.cfg.use_divergence_matching:
            predicted_velocity, divergence_matching_loss = FlowMatching.divergence_matching_loss(
                self.cfg, lambda flow_matching_time, noise_flowed_to_t: self(flow_matching_time, noise_flowed_to_t, noise, predicted_state),
                path_context['flow_matching_time'],
                path_context['noise_flowed_to_t'],
                path_context['target_velocity'],
                path_context['mean'],
                path_context['std'],
                path_context['dt_std'],
            )
        else:
            predicted_velocity = self(flow_matching_time, path_context['noise_flowed_to_t'], noise, predicted_state)

        flow_loss = reduce(
            path_context['diffusion_path_weighting'] * (predicted_velocity - path_context['target_velocity']).square(),
            'predicted_state_count dim -> predicted_state_count', 'sum'
        ).mean()

        return dict(
            loss=flow_loss + divergence_matching_loss,
            flow_loss=flow_loss,
            divergence_matching_loss=divergence_matching_loss,
        )

    @torch.no_grad
    def sample(self, current_states, observation, observe, time_step_count=None):
        time_step_count = time_step_count or self.cfg.sampling_time_step_count
        if isinstance(self.cfg.diffusion_path, conf.diffusion_path.ConditionalOptimalTransport):
            diffusion_time = torch.linspace(0., 1., time_step_count, device=current_states.device)
            xt = torch.randn_like(current_states)
        elif isinstance(self.cfg.diffusion_path, conf.diffusion_path.VarianceExploding):
            diffusion_time = torch.linspace(0., 1. - self.cfg.diffusion_path.time_min, time_step_count, device=current_states.device)
            xt = torch.randn_like(current_states) * self.sigma(1)
        else:
            raise ValueError(f'Unknown diffusion path: {self.cfg.diffusion_path}')
        time_step_size = diffusion_time[1] - diffusion_time[0]

        noise = xt
        if (
            observation is not None
            and not self.cfg.ignore_observations
        ):
            def dot_state(t, x):
                dot_state_unguided = self(t, x, noise, current_states)
                return dot_state_unguided + self.guidance(
                    t, x, noise, current_states, dot_state_unguided,
                    energy_function=lambda x1_predicted: reduce(
                        (observe(x1_predicted) - observation).square(),
                        'predicted_state_count dim -> predicted_state_count 1',
                        'sum'
                    )
                )
        else:
            dot_state = lambda t, x: self(t, x, noise, current_states)

        for t_now, t_next in zip(diffusion_time, diffusion_time[1:]):
            if self.cfg.sampler is conf.models.Sampler.EULER:
                xt = xt + time_step_size * dot_state(t_now, xt)
                state_out = xt
            elif self.cfg.sampler is conf.models.Sampler.EULER_MARUYAMA:
                if not isinstance(self.cfg.diffusion_path, conf.diffusion_path.VarianceExploding):
                    raise ValueError(
                        f'The Euler-Maruyama sampler is only supported with the variance exploding diffusion path, not {self.cfg.diffusion_path.__class__.__name__}.'
                        ' Please use a different sampler (e.g., set model.sampler=EULER), or use a diffusion diffusion path (e.g., set model/diffusion_path=ConditionalOptimalTransport).'
                    )
                # this is sqrt(d/dt (self.sigma(t))^2) evaluated at 1 - t_now
                g = self.cfg.diffusion_path.sigma_min * (self.cfg.diffusion_path.sigma_max / self.cfg.diffusion_path.sigma_min)**(1 - t_now)
                if self.cfg.sampling_use_observation_likelihood_score:
                    score = dot_state(t_now, xt) / g.square()
                    if observation is None or self.cfg.ignore_observations:
                        observation_score = 0.
                    else:
                        raise NotImplementedError('Need to use automatic differentiation to get the score')
                        # this seems backwards; should it be (observation - xt)? because we are given xt?
                        observation_score = -(xt - observation) / self.observation_std.square()
                    score = score + observation_score * self.observation_likelihood_score_damping(1 - t_now)
                    state_drift = xt + g.square() * score * time_step_size
                else:
                    state_drift = xt + time_step_size * dot_state(t_now, xt)
                xt = state_drift + g * torch.randn_like(xt) * time_step_size.sqrt()
                state_out = state_drift
            elif self.cfg.sampler is conf.models.Sampler.HEUN:
                xdot_now = dot_state(t_now, xt)
                temp = xt + time_step_size * xdot_now
                xt = xt + time_step_size * (xdot_now + dot_state(t_next, temp)) / 2
                state_out = xt
            else:
                raise ValueError(f'Unsupported sampler for {self.__class__.__name__}: {self.cfg.sampler}')

        return state_out


def get_model(cfg, state_dimension, observation_std):
    if isinstance(cfg, conf.models.ScoreMatching):
        diffusion_path = dafm.diffusion_path.get_diffusion_path(cfg.diffusion_path)
        return ScoreMatching(cfg, state_dimension, observation_std, diffusion_path)
    elif isinstance(cfg, conf.models.FlowMatching):
        diffusion_path = dafm.diffusion_path.get_diffusion_path(cfg.diffusion_path)
        guidance = flow_matching_guidance.get_guidance(cfg.guidance)
        return FlowMatching(cfg, state_dimension, observation_std, diffusion_path, guidance)
    elif isinstance(cfg, conf.models.FlowMatchingMarginal):
        diffusion_path = dafm.diffusion_path.get_diffusion_path(cfg.diffusion_path)
        guidance = flow_matching_guidance.get_guidance(cfg.guidance)
        return FlowMatchingMarginal(cfg, diffusion_path, guidance)
    else:
        raise ValueError(f'Unknown model: {cfg}')
