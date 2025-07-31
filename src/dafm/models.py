from functools import partial, cache
import logging

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

import conf.models
import conf.diffusion_path
import conf.inflation_scale
import dafm.diffusion_path
import dafm.inflation_scale
from dafm import flow_matching_guidance, models_classical, models_classical_iterative, utils


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
    def __init__(self, cfg, state_dimension, observation_noise_std, diffusion_path, inflation_scale):
        super().__init__()
        self.cfg = cfg
        self.state_dimension = state_dimension
        self.observation_noise_std = observation_noise_std
        self.diffusion_path = diffusion_path
        self.inflation_scale = inflation_scale
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
        return super().forward(time, state) / self.diffusion_path.std(time, None)

    def get_optimizer(self, time_step, ignore_observation):
        if self.cfg.train_on_initial_predicted_state and time_step == 0 and ignore_observation:
            lr = self.cfg.learning_rate_when_training_on_initial_predicted_state
        else:
            lr = self.cfg.learning_rate
        return torch.optim.Adam(self.parameters(), lr=lr)

    def loss(self, state, observation, observe):
        diffusion_time = self.diffusion_path.sample_time((state.shape[0], 1), device=state.device)
        noise = torch.randn_like(state)
        std = self.diffusion_path.std(diffusion_time, state)
        noise_flowed_to_t = state + noise * std
        predicted_score = self(diffusion_time, noise_flowed_to_t)
        return dict(
            loss=reduce(
                (predicted_score * std + noise).square(),
                'batch dim -> batch', 'sum'
            ).mean(),
        )

    def observation_likelihood_score_damping(self, t):
        return (1 - 2 * t).clamp(min=0)

    @classmethod
    def _sampling_steps(cls, cfg, diffusion_path, forward, observation_noise_std, observation_likelihood_score_damping, data, observation, observe, time_step_count=None):
        time_step_count = time_step_count or cfg.sampling_time_step_count
        path_time = diffusion_path.linspace_time(time_step_count, device=data.device)
        minus_time_step_size = path_time[1] - path_time[0]
        minus_time_step_size_abs_sqrt = minus_time_step_size.abs().sqrt()
        noise = diffusion_path.sample_noise(path_time[0], data)
        xt = noise
        for time_step, t_now_and_next in enumerate(path_time.unfold(0, 2, 1)):
            t_now = t_now_and_next[0]
            done = False
            yield done, time_step, t_now, xt
            score = forward(t_now, xt)
            if observation is None or cfg.ignore_observations:
                observation_score = 0.
            else:
                with torch.enable_grad():
                    xt_grad = xt.detach().requires_grad_()
                    observation_likelihood_distribution = utils.Independent(
                        utils.Normal(observation, observation_noise_std),
                        1,
                    )
                    log_observation_likelihood = observation_likelihood_distribution.log_prob_unnormalized(observe(xt_grad))
                    observation_score, *_ = torch.autograd.grad(
                        outputs=log_observation_likelihood.sum(),
                        inputs=xt_grad,
                    )
            score = score + observation_score * observation_likelihood_score_damping(t_now)

            if cfg.sampling_score_norm is conf.models.LNorm.RMS:
                score_norm = reduce(score.square(), 'batch dim -> batch', 'mean').sqrt()
                score_norm_too_large = score_norm > cfg.sampling_max_score_norm
                score[score_norm_too_large] = score[score_norm_too_large] * rearrange(
                    cfg.sampling_max_score_norm / score_norm[score_norm_too_large],
                    'batch -> batch 1'
                )
            elif cfg.sampling_score_norm is conf.models.LNorm.LInfty:
                score = score.clamp(min=-cfg.sampling_max_score_norm, max=cfg.sampling_max_score_norm)
            else:
                raise ValueError(f'Unknown sampling score norm: {cfg.sampling_score_norm}')

            g = diffusion_path.g(t_now)
            if cfg.sampler is conf.models.Sampler.EULER:
                xt = xt + minus_time_step_size * (diffusion_path.f(t_now, xt) - g.square() * score)
                state_out = xt
            elif cfg.sampler is conf.models.Sampler.EULER_MARUYAMA:
                state_drift = xt + minus_time_step_size * (diffusion_path.f(t_now, xt) - g.square() * score)
                xt = state_drift + minus_time_step_size_abs_sqrt * g * torch.randn_like(xt)
                state_out = xt
            else:
                raise ValueError(f'Unsupported sampler for {cfg.__name__}: {cfg.sampler}')

        # no noise in final step
        # maybe this is done to handle the discontinuity of the variance exploding path as t -> 0.
        # just use the mean of the noise distribution for the last step
        done = True
        yield done, time_step, t_now, state_out

    @torch.no_grad
    def sampling_steps(self, current_states, observation, observe, time_step_count=None):
        for done, time_step, t_now_and_next, xt in self._sampling_steps(
            self.cfg,
            self.diffusion_path,
            self,
            self.observation_noise_std,
            self.observation_likelihood_score_damping,
            current_states, observation, observe,
            time_step_count=time_step_count,
        ):
            yield done, time_step, t_now_and_next, xt


class ScoreMatchingMarginal(nn.Module):
    def __init__(self, cfg, observation_noise_std, diffusion_path, inflation_scale):
        super().__init__()
        self.cfg = cfg
        self.observation_noise_std = observation_noise_std
        self.diffusion_path = diffusion_path
        self.inflation_scale = inflation_scale

    def get_optimizer(self, time_step, ignore_observation):
        if self.cfg.train_on_initial_predicted_state and time_step == 0 and ignore_observation:
            lr = self.cfg.learning_rate_when_training_on_initial_predicted_state
        else:
            lr = self.cfg.learning_rate
        return torch.optim.Adam(self.parameters(), lr=lr)

    def score(self, mean, std, time, x):
        return -(x - mean) / std.square()

    def _loss(self, mean, std, state, observation, observe):
        raise NotImplementedError()

    def _forward(self, mean, std, time, x):
        return reduce(
            self.weights * self.score(
                rearrange(mean, 'particle_count dim -> particle_count 1 dim'),
                std,
                time,
                rearrange(x, 'predicted_state_count dim -> 1 predicted_state_count dim'),
            ),
            'particle_count predicted_state_count dim -> predicted_state_count dim',
            'sum',
        )

    def observation_likelihood_score_damping(self, t):
        return 1 - t

    @torch.no_grad
    def sampling_steps(self, data, observation, observe, time_step_count=None):
        for done, time_step, time, xt in ScoreMatching._sampling_steps(
            self.cfg,
            self.diffusion_path,
            self,
            self.observation_noise_std,
            self.observation_likelihood_score_damping,
            data, observation, observe,
            time_step_count=time_step_count,
        ):
            if done:
                yield done, time_step, time, xt
                break
            mean = self.diffusion_path.mean(time, data[:self.cfg.particle_count])
            std = self.diffusion_path.std(time, data[:self.cfg.particle_count])
            conditional_distribution = utils.Independent(
                utils.Normal(
                    loc=rearrange(mean, 'particle_count dim -> particle_count 1 dim'),
                    scale=std,
                ),
                1,
            )
            log_pt_given_x1 = rearrange(
                conditional_distribution.log_prob_unnormalized(
                    rearrange(xt, 'predicted_state_count dim -> 1 predicted_state_count dim')
                ),
                'particle_count predicted_state_count -> particle_count predicted_state_count 1',
            )
            weights = log_pt_given_x1.softmax(0)
            if self.cfg.epoch_count_sampling > 0:
                self.loss = partial(self._loss, mean, std, time)
                self.weights = nn.Parameter(weights)
            else:
                self.weights = weights
            self.forward = partial(self._forward, mean, std)
            yield done, time_step, time, xt


class FlowMatching(Model):
    def __init__(self, cfg, state_dimension, observation_noise_std, diffusion_path, inflation_scale, guidance):
        super().__init__(cfg, state_dimension, observation_noise_std, diffusion_path, inflation_scale)
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
    def diffusion_path_context(cfg, diffusion_path, path_time, noise, data):
        mean = diffusion_path.mean(path_time, data)
        std = diffusion_path.std(path_time, data)
        dt_std = diffusion_path.dt_std(path_time, data)
        noise_flowed_to_t = mean + noise * std
        if isinstance(cfg.diffusion_path, conf.diffusion_path.ConditionalOptimalTransport):
            target_velocity = data - noise
        elif isinstance(cfg.diffusion_path, conf.diffusion_path.VarianceExploding):
            target_velocity = dt_std * noise
        else:
            raise ValueError(f'Unknown diffusion path: {cfg.diffusion_path}')

        diffusion_path_weighting = 1 / dt_std.square()

        return dict(
            path_time=path_time,
            noise_flowed_to_t=noise_flowed_to_t,
            target_velocity=target_velocity,
            mean=mean, std=std, dt_std=dt_std,
            diffusion_path_weighting=diffusion_path_weighting,
        )

    @staticmethod
    def divergence_matching_loss(cfg, mean, std, dt_std, path_time, forward, noise_flowed_to_t, target_velocity):
        predicted_state_count, dim = noise_flowed_to_t.shape
        dx_target_velocity = dt_std / std
        hutchinson_noise = torch.randn((predicted_state_count, dim), device=noise_flowed_to_t.device)
        predicted_velocity, predicted_velocity_jvp = torch.autograd.functional.jvp(
            lambda xt: forward(path_time, xt),
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

        path_time = self.diffusion_path.sample_time((self.cfg.loss_expectation_sample_count, time_noise_samples_per_expectation_sample, 1), device=state.device)
        path_context = self.diffusion_path_context(self.cfg, self.diffusion_path, path_time, noise, state)
        if self.cfg.use_expectation_of_sum:
            path_time = repeat(path_time, 'loss_expectation_sample_count 1 1 -> loss_expectation_sample_count predicted_state_count 1', predicted_state_count=predicted_state_count)
        path_context['path_time'] = path_time
        path_context = {
            k: rearrange(x, 'loss_expectation_sample_count predicted_state_count dim -> (loss_expectation_sample_count predicted_state_count) dim')
            for k, x in path_context.items()
        }

        if self.cfg.use_divergence_matching:
            predicted_velocity, divergence_matching_loss = self.divergence_matching_loss(
                self.cfg,
                path_context['mean'],
                path_context['std'],
                path_context['dt_std'],
                path_context['path_time'],
                self,
                path_context['noise_flowed_to_t'],
                path_context['target_velocity'],
            )
        else:
            predicted_velocity = self(path_context['path_time'], path_context['noise_flowed_to_t'])
            divergence_matching_loss = 0.

        predicted_velocity = rearrange(
            predicted_velocity,
            '(loss_expectation_sample_count predicted_state_count) dim -> loss_expectation_sample_count predicted_state_count dim',
            loss_expectation_sample_count=self.cfg.loss_expectation_sample_count,
        )

        if isinstance(self.cfg.guidance, flow_matching_guidance.No) or observation is None or self.cfg.ignore_observations:
            weighting = 1 / predicted_state_count
        else:
            observation_likelihood_distribution = utils.Independent(
                utils.Normal(observe(state), self.observation_noise_std),
                1,
            )
            log_observation_likelihood = rearrange(
                observation_likelihood_distribution.log_prob_unnormalized(observation),
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

    @classmethod
    def _sampling_steps(cls, cfg, diffusion_path, forward, guidance, observation_noise_std, data, observation, observe, time_step_count=None):
        time_step_count = time_step_count or cfg.sampling_time_step_count
        path_time = diffusion_path.linspace_time(time_step_count, device=data.device)
        noise = diffusion_path.sample_noise(path_time[0], data)
        xt = noise
        if isinstance(cfg.guidance, flow_matching_guidance.No) or observation is None or cfg.ignore_observations:
            velocity = forward
        else:
            observation_likelihood_distribution = utils.Independent(
                utils.Normal(observation, observation_noise_std),
                1,
            )
            if cfg.guidance.use_approximate_conditional_velocity_for_unguided_velocity:
                if not isinstance(cfg.diffusion_path, conf.diffusion_path.ConditionalOptimalTransport):
                    raise ValueError(f'use_approximate_conditional_velocity_for_unguided_velocity not supported for diffusion path {cfg.diffusion_path.__class__.__name__}')
                data = rearrange(data, 'data dim -> data 1 dim')
                noise = rearrange(noise, 'noise dim -> 1 noise dim')
                def velocity(t, x):
                    if cfg.guidance.approximate_conditional_velocity_scale_data_by_time:
                        conditional_velocity_candidates = t * data - noise
                    else:
                        conditional_velocity_candidates = data - noise
                    expected_conditional_velocity_given_xt = rearrange(
                        x,
                        'noise dim -> 1 noise dim',
                    )
                    best_approximate_conditional_velocity_idx = reduce(
                        (conditional_velocity_candidates - expected_conditional_velocity_given_xt).square(),
                        'data noise dim -> data noise 1',
                        'sum',
                    ).argmin(0, keepdim=True).expand(1, data.shape[0], data.shape[2])
                    return conditional_velocity_candidates.gather(0, best_approximate_conditional_velocity_idx).squeeze(0)
            else:
                def velocity(t, x):
                    dot_state_unguided = forward(t, x)
                    return dot_state_unguided + guidance(
                        noise, data,
                        t, x,
                        dot_state_unguided,
                        energy_function=lambda x1_predicted: rearrange(
                            -observation_likelihood_distribution.log_prob_unnormalized(observe(x1_predicted)),
                            'predicted_state_count -> predicted_state_count 1',
                        )
                    )

        time_step_size = path_time[1] - path_time[0]
        time_step_size_sqrt = time_step_size.sqrt()
        for time_step, t_now_and_next in enumerate(path_time.unfold(0, 2, 1)):
            t_now, t_next = t_now_and_next
            if cfg.sampler is conf.models.Sampler.EULER:
                done = False
                yield done, time_step, t_now, noise, xt
                xt = xt + time_step_size * velocity(t_now, xt)
                state_out = xt
            elif cfg.sampler is conf.models.Sampler.EULER_MARUYAMA:
                raise NotImplementedError()
                if not isinstance(cfg.diffusion_path, conf.diffusion_path.VarianceExploding):
                    raise ValueError(
                        f'The Euler-Maruyama sampler is only supported with the variance exploding diffusion path, not {cfg.diffusion_path.__class__.__name__}.'
                        ' Please use a different sampler (e.g., set model.sampler=EULER), or use a diffusion diffusion path (e.g., set model/diffusion_path=ConditionalOptimalTransport).'
                    )
                g = diffusion_path.g(1 - t_now)
                done = False
                yield done, time_step, t_now, noise, xt
                state_drift = xt + time_step_size * velocity(t_now, xt)
                xt = state_drift + g * torch.randn_like(xt) * time_step_size_sqrt
                state_out = state_drift
            elif cfg.sampler is conf.models.Sampler.HEUN:
                done = False
                yield done, time_step, t_now, noise, xt
                xdot_now = velocity(t_now, xt)
                temp = xt + time_step_size * xdot_now
                done = False
                yield done, time_step, t_next, noise, temp
                xt = xt + time_step_size * (xdot_now + velocity(t_next, temp)) / 2
                state_out = xt
            else:
                raise ValueError(f'Unsupported sampler for {cfg.__name__}: {cfg.sampler}')

        done = True
        yield done, time_step, t_now, noise, state_out

    @torch.no_grad
    def sampling_steps(self, current_states, observation, observe, time_step_count=None):
        for done, time_step, t_now_and_next, noise, xt in FlowMatching._sampling_steps(
            self.cfg,
            self.diffusion_path,
            self,
            self.guidance,
            self.observation_noise_std,
            current_states, observation, observe,
            time_step_count=time_step_count
        ):
            yield done, time_step, t_now_and_next, xt


class FlowMatchingMarginal(nn.Module):
    def __init__(self, cfg, observation_noise_std, diffusion_path, inflation_scale, guidance):
        super().__init__()
        self.cfg = cfg
        self.observation_noise_std = observation_noise_std
        self.diffusion_path = diffusion_path
        self.inflation_scale = inflation_scale
        self.guidance = guidance

    def get_optimizer(self, time_step, ignore_observation):
        if self.cfg.train_on_initial_predicted_state and time_step == 0 and ignore_observation:
            lr = self.cfg.learning_rate_when_training_on_initial_predicted_state
        else:
            lr = self.cfg.learning_rate
        return torch.optim.Adam(self.parameters(), lr=lr)

    def conditional_velocity(self, mean, dt_mean, std, dt_std, time, x):
        return dt_std / std * (x - mean) + dt_mean

    def _forward(self, mean, dt_mean, std, dt_std, time, x):
        return reduce(
            self.weights * self.conditional_velocity(
                rearrange(mean, 'particle_count dim -> particle_count 1 dim'),
                rearrange(dt_mean, 'particle_count dim -> particle_count 1 dim'),
                std, dt_std,
                time,
                rearrange(x, 'predicted_state_count dim -> 1 predicted_state_count dim'),
            ),
            'particle_count predicted_state_count dim -> predicted_state_count dim',
            'mean'
        )

    def _loss(self, mean, dt_mean, std, dt_std, time, xt, observation, observe):
        target_velocity = self.conditional_velocity(mean, dt_mean, std, dt_std, time, xt)
        if self.cfg.use_divergence_matching:
            predicted_velocity, divergence_matching_loss = FlowMatching.divergence_matching_loss(
                self.cfg,
                mean,
                std,
                dt_std,
                time,
                lambda t, noise_flowed_to_t: self(t, noise_flowed_to_t),
                xt,
                target_velocity,
            )
        else:
            predicted_velocity = self(time, xt)
            divergence_matching_loss = 0.

        diffusion_path_weighting = 1 / dt_std.square()

        flow_loss = reduce(
            diffusion_path_weighting * (predicted_velocity - target_velocity).square(),
            'predicted_state_count dim -> predicted_state_count', 'sum'
        ).mean()

        return dict(
            loss=flow_loss + divergence_matching_loss,
            flow_loss=flow_loss,
            divergence_matching_loss=divergence_matching_loss,
        )

    @torch.no_grad
    def sampling_steps(self, data, observation, observe, time_step_count=None):
        predicted_state_count = data.shape[0]
        for done, time_step, time, noise, xt in FlowMatching._sampling_steps(
            self.cfg,
            self.diffusion_path,
            self,
            self.guidance,
            self.observation_noise_std,
            data, observation, observe,
            time_step_count=time_step_count
        ):
            if done:
                yield done, time_step, time, xt
                break
            mean = self.diffusion_path.mean(time, data)
            dt_mean = self.diffusion_path.dt_mean(time, data)
            std = self.diffusion_path.std(time, data)
            dt_std = self.diffusion_path.dt_std(time, data)
            conditional_distribution = utils.Independent(
                utils.Normal(
                    loc=rearrange(mean, 'particle_count dim -> particle_count 1 dim'),
                    scale=std,
                ),
                1,
            )
            log_pt_given_x1 = rearrange(
                conditional_distribution.log_prob_unnormalized(
                    rearrange(xt, 'predicted_state_count dim -> 1 predicted_state_count dim')
                ),
                'particle_count predicted_state_count -> particle_count predicted_state_count 1',
            )
            weights = log_pt_given_x1.softmax(0) * predicted_state_count
            if self.cfg.epoch_count_sampling > 0:
                self.loss = partial(self._loss, mean, dt_mean, std, dt_std, time)
                self.weights = nn.Parameter(weights)
            else:
                self.weights = weights
            self.forward = partial(self._forward, mean, dt_mean, std, dt_std)
            yield done, time_step, time, xt


class FlowMatchingGaussianTarget(nn.Module):
    def __init__(self, cfg, observation_noise_std, diffusion_path, inflation_scale, guidance):
        super().__init__()
        self.cfg = cfg
        self.observation_noise_std = observation_noise_std
        self.diffusion_path = diffusion_path
        self.inflation_scale = inflation_scale
        self.guidance = guidance

    def forward(self, time, x):
        target_mean = reduce(x, 'predicted_state_count dim -> dim', 'mean')
        target_covariance = torch.cov(x.T)

        path_std = self.diffusion_path.std(time, target_mean)
        dim = x.shape[1]
        identity = torch.eye(dim, device=x.device)
        covariance = (
            path_std**2 * identity
            + time**2 * target_covariance
        )
        dt_covariance = (
            2 * path_std * self.diffusion_path.dt_std(time, target_mean) * identity
            + 2 * time * target_covariance
        )

        if dim <= 1:
            dt_std_div_inv_std = 0.5 * covariance / dt_covariance
            predicted_velocity = (
                (x - self.diffusion_path.mean(time, target_mean)) * dt_std_div_inv_std
                + self.diffusion_path.dt_mean(time, target_mean)
            )
        else:
            dt_std_div_inv_std = 0.5 * torch.linalg.solve(covariance, dt_covariance, left=False)
            predicted_velocity = (
                (x - self.diffusion_path.mean(time, target_mean)) @ dt_std_div_inv_std.T
                + self.diffusion_path.dt_mean(time, target_mean)
            )

        return predicted_velocity

    @torch.no_grad
    def sampling_steps(self, data, observation, observe, time_step_count=None):
        for done, time_step, time, noise, xt in FlowMatching._sampling_steps(
            self.cfg,
            self.diffusion_path,
            self,
            self.guidance,
            self.observation_noise_std,
            data, observation, observe,
            time_step_count=time_step_count
        ):
            yield done, time_step, time, xt


class Classical(nn.Module):
    def __init__(self, cfg, observation_noise_std, inflation_scale, dynamics):
        super().__init__()
        self.cfg = cfg
        self.dynamics = dynamics
        self.observation_noise_std = observation_noise_std
        self.inflation_scale = inflation_scale

    @staticmethod
    @cache
    def coords(dimension, device=None, dtype=None):
        return torch.arange(dimension, device=device, dtype=dtype).unsqueeze(1)

    @staticmethod
    @cache
    def domain_lengths(dimension, device=None, dtype=None):
        return torch.tensor([dimension], device=device, dtype=dtype)

    @staticmethod
    @cache
    def L_thing(coords_a, coords_b, loc_radius_gc, domain_lengths=None):
        return models_classical.gaspari_cohn_correlation(
            models_classical.pairwise_distances_torch(coords_a, coords_b, domain_lengths=domain_lengths),
            loc_radius_gc
        )


class BootstrapParticleFilter(Classical):
    @torch.no_grad
    def sampling_steps(self, data, observation, observe, time_step_count=None):
        sampled_state = models_classical.bootstrap_particle_filter_analysis(
            particles_forecast=data,
            observation_y=rearrange(observation, '1 dim -> dim'),
            observation_operator=observe,
            sigma_y=self.observation_noise_std,
            resampling_method="multinomial",
        )
        yield True, 0, None, sampled_state


class EnsembleKalmanFilterPerturbedObservations(Classical):
    @torch.no_grad
    def sampling_steps(self, data, observation, observe, time_step_count=None):
        coords_state = self.coords(data.shape[1], device=data.device)
        coords_observation = self.coords(observation.shape[1], device=data.device)
        domain_lengths = self.domain_lengths(data.shape[1], device=data.device)
        sampled_state = models_classical.ensemble_kalman_filter_analysis(
            ensemble_f=data,
            observation_y=rearrange(observation, '1 dim -> dim'),
            observation_operator_ens=observe,
            sigma_y=self.observation_noise_std,
            method="EnKF-PertObs",
            localization_matrix_Lxy=self.L_thing(
                coords_state, coords_observation, self.cfg.loc_radius_gc,
                domain_lengths=domain_lengths,
            ),
            localization_matrix_Lyy=self.L_thing(
                coords_observation, coords_observation, self.cfg.loc_radius_gc,
                domain_lengths=domain_lengths,
            ),
            do_inflation=False,  # inflation is done in src/dafm/datasets.py
        )[0]
        yield True, 0, None, sampled_state


class EnsembleKalmanFilterPerturbedObservationsIterative(Classical):
    @torch.no_grad
    def sampling_steps(self, data, observation, observe, time_step_count=None):
        sampled_state = models_classical_iterative.ensemble_kalman_filter_analysis(
            ensemble_f=data[None],
            observation_y=rearrange(observation, '1 dim -> dim'),
            observation_operator_ens=observe,
            sigma_y=self.observation_noise_std,
            method='iEnKS-PertObs',
            inflation_factor=1.,  # inflation is done in src/dafm/datasets.py
            ienks_lag=1,
            ienks_niter=10,
            ienks_wtol=1e-5,
            model_args=dict(
                # passing None to the propagator because the experiments being run
                # happen to not need these arguments
                propagator=lambda rhs_func, state, dt: self.dynamics._step_state(None, None, state, None),
                rhs=None,
                dt=self.dynamics.cfg.time_step_size,
                steps_between_analyses=self.dynamics.cfg.observe_every_n_time_steps,
            ),
        )[0]
        yield True, 0, None, sampled_state.squeeze(0)


class EnsembleRandomizedSquareRootFilter(Classical):
    @torch.no_grad
    def sampling_steps(self, data, observation, observe, time_step_count=None):
        sampled_state = models_classical.ensemble_kalman_filter_analysis(
            ensemble_f=data,
            observation_y=rearrange(observation, '1 dim -> dim'),
            observation_operator_ens=observe,
            sigma_y=self.observation_noise_std,
            method='ERSF',
            do_inflation=False,  # inflation is done in src/datasets.py
        )[0]
        yield True, 0, None, sampled_state


class LocalEnsembleTransformKalmanFilter(Classical):
    @torch.no_grad
    def sampling_steps(self, data, observation, observe, time_step_count=None):
        coords_state = self.coords(data.shape[1], device=data.device, dtype=data.dtype)
        coords_observation = self.coords(observation.shape[1], device=data.device, dtype=data.dtype)
        domain_lengths = self.domain_lengths(data.shape[1], device=data.device, dtype=data.dtype)
        sampled_state = models_classical.ensemble_kalman_filter_analysis(
            ensemble_f=data,
            observation_y=rearrange(observation, '1 dim -> dim'),
            observation_operator_ens=observe,
            sigma_y=self.observation_noise_std,
            method='LETKF',
            localization_radius_letkf=self.cfg.loc_radius_gc,
            coords_state_letkf=coords_state,
            coords_obs_letkf=coords_observation,
            domain_lengths_letkf=domain_lengths,
            do_inflation=False,  # inflation is done in src/datasets.py
            inflation_factor=self.cfg.inflation_scale.constant,
        )[0]
        yield True, 0, None, sampled_state


def get_model(cfg, state_dimension, observation_noise_std, dynamics):
    inflation_scale = dafm.inflation_scale.get_inflation_scale(cfg.inflation_scale)
    if isinstance(cfg, conf.models.ScoreMatching):
        diffusion_path = dafm.diffusion_path.get_diffusion_path(cfg.diffusion_path, target_distribution_at_time_1=False)
        return ScoreMatching(cfg, state_dimension, observation_noise_std, diffusion_path, inflation_scale)
    elif isinstance(cfg, conf.models.ScoreMatchingMarginal):
        diffusion_path = dafm.diffusion_path.get_diffusion_path(cfg.diffusion_path, target_distribution_at_time_1=False)
        return ScoreMatchingMarginal(cfg, observation_noise_std, diffusion_path, inflation_scale)
    elif isinstance(cfg, conf.models.FlowMatching):
        diffusion_path = dafm.diffusion_path.get_diffusion_path(cfg.diffusion_path, target_distribution_at_time_1=True)
        guidance = flow_matching_guidance.get_guidance(cfg.guidance)
        return FlowMatching(cfg, state_dimension, observation_noise_std, diffusion_path, inflation_scale, guidance)
    elif isinstance(cfg, conf.models.FlowMatchingMarginal):
        diffusion_path = dafm.diffusion_path.get_diffusion_path(cfg.diffusion_path, target_distribution_at_time_1=True)
        guidance = flow_matching_guidance.get_guidance(cfg.guidance)
        return FlowMatchingMarginal(cfg, observation_noise_std, diffusion_path, inflation_scale, guidance)
    elif isinstance(cfg, conf.models.FlowMatchingGaussianTarget):
        diffusion_path = dafm.diffusion_path.get_diffusion_path(cfg.diffusion_path, target_distribution_at_time_1=True)
        guidance = flow_matching_guidance.get_guidance(cfg.guidance)
        return FlowMatchingGaussianTarget(cfg, observation_noise_std, diffusion_path, inflation_scale, guidance)
    elif isinstance(cfg, conf.models.BootstrapParticleFilter):
        return BootstrapParticleFilter(cfg, observation_noise_std, inflation_scale, dynamics)
    elif isinstance(cfg, conf.models.EnsembleKalmanFilterPerturbedObservations):
        return EnsembleKalmanFilterPerturbedObservations(cfg, observation_noise_std, inflation_scale, dynamics)
    elif isinstance(cfg, conf.models.EnsembleRandomizedSquareRootFilter):
        return EnsembleRandomizedSquareRootFilter(cfg, observation_noise_std, inflation_scale, dynamics)
    elif isinstance(cfg, conf.models.LocalEnsembleTransformKalmanFilter):
        return LocalEnsembleTransformKalmanFilter(cfg, observation_noise_std, inflation_scale, dynamics)
    elif isinstance(cfg, conf.models.EnsembleKalmanFilterPerturbedObservationsIterative):
        return EnsembleKalmanFilterPerturbedObservationsIterative(cfg, observation_noise_std, inflation_scale, dynamics)
    else:
        raise ValueError(f'Unknown model: {cfg}')
