from functools import cache

import torch
import torch.nn as nn
import numpy as np
from einops import reduce

import conf.dataset
import conf.filter
import dafm.filters_classical
import dafm.filters_classical_iterative
import dafm.flow_matching_guidance
import dafm.prob_path
import dafm.utils


class Filter(nn.Module):
    def __init__(self, cfg, rng: np.random.Generator):
        super().__init__()
        self.cfg = cfg
        self.rng = rng

    def initialize(self, cfg_data_assimilation_setting, initial_ensemble, dynamics):
        pass

    @staticmethod
    def rng_normal(shape, rng, std, **kwargs):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = torch.float32
        if std > 0:
            noise = torch.tensor(rng.normal(scale=std, size=shape), **kwargs)
        else:
            noise = 0.
        return noise

    @staticmethod
    def log_prob_to_weight_safe(log_prob, failed_to_sum_to_one_threshold=1e-9):
        weights = (log_prob - log_prob.max()).exp()
        weights = torch.where(weights.sum() > failed_to_sum_to_one_threshold, weights, 1.)
        weights_sum = weights.sum()
        weights /= weights_sum
        return weights

    def inflate(self, ensemble):
        if self.cfg.inflation_scale != 1:
            ensemble_mean = reduce(ensemble, 'member ... -> ...', 'mean')
            ensemble_centered = ensemble - ensemble_mean
            ensemble = ensemble_mean + self.cfg.inflation_scale * ensemble_centered
        return ensemble

    def assimilate(self, time_step, predictive_ensemble, observation, obs_noise_std, observe):
        raise NotImplementedError()


class AddNoiseToObservationFilter(Filter):
    def assimilate(self, time_step, predictive_ensemble, observation, obs_noise_std, observe):
        ensemble = observation + self.rng_normal(predictive_ensemble.shape, self.rng, self.cfg.noise_std, device=predictive_ensemble.device)
        ensemble = self.inflate(ensemble)
        return ensemble


class Classical(Filter):
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
        return dafm.filters_classical.gaspari_cohn_correlation(
            dafm.filters_classical.pairwise_distances_torch(coords_a, coords_b, domain_lengths=domain_lengths),
            loc_radius_gc
        )


class BootstrapParticleFilter(Classical):
    def __init__(self, cfg, rng: np.random.Generator):
        super().__init__(cfg, rng)
        self._torch_noise_base_seed = int(self.rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        self._torch_noise_generators = {}

    def assimilate(self, time_step, predictive_ensemble, observation, obs_noise_std, observe):
        if self.cfg.model_noise_std > 0:
            predictive_ensemble = predictive_ensemble + self.cfg.model_noise_std * torch.randn(
                predictive_ensemble.shape,
                device=predictive_ensemble.device,
                dtype=predictive_ensemble.dtype,
                generator=dafm.utils.get_torch_rng_generator(
                    self._torch_noise_base_seed,
                    self._torch_noise_generators,
                    predictive_ensemble.device,
                ),
            )
        observation_likelihood_distribution = dafm.utils.Independent(
            dafm.utils.Normal(observation, obs_noise_std), 1,
        )
        log_observation_likelihood = observation_likelihood_distribution.log_prob_unnormalized(observe(predictive_ensemble))
        weights = self.log_prob_to_weight_safe(log_observation_likelihood)
        indicies = torch.multinomial(
            weights,
            num_samples=weights.numel(),
            replacement=True,
            generator=dafm.utils.get_torch_rng_generator(
                self._torch_noise_base_seed,
                self._torch_noise_generators,
                weights.device,
            ),
        )
        ensemble = predictive_ensemble[indicies]
        ensemble = self.inflate(ensemble)
        return ensemble


class KalmanFilter(Classical):
    def initialize(self, cfg_data_assimilation_setting, initial_ensemble, dynamics):
        self.dynamics_matrix = dynamics.linearize(initial_ensemble)
        if not hasattr(self, 'covariance'):
            self.state_identity_matrix = torch.eye(self.dynamics_matrix.shape[0], device=initial_ensemble.device)
            self.obs_identity_matrix = None
            self.covariance = self.cfg.ensemble_initial_std**2 * self.state_identity_matrix

    def assimilate(self, time_step, predictive_ensemble, observation, obs_noise_std, observe):
        predictive_covariance = self.dynamics_matrix @ self.covariance @ self.dynamics_matrix.mT + self.cfg.model_noise_std**2 * self.state_identity_matrix
        innovation = observation - observe(predictive_ensemble)
        observe_matrix = observe.linearize(predictive_ensemble)
        if self.obs_identity_matrix is None:
            self.obs_identity_matrix = torch.eye(observe_matrix.shape[0], device=predictive_ensemble.device)
        innovation_covariance = observe_matrix @ predictive_covariance @ observe_matrix.mT + obs_noise_std**2 * self.obs_identity_matrix
        try:
            kalman_gain = torch.linalg.solve(innovation_covariance, predictive_covariance @ observe_matrix.mT, left=False)
        except torch.linalg.LinAlgError:
            kalman_gain = predictive_covariance @ observe_matrix.mT @ torch.linalg.pinv(innovation_covariance)
        ensemble = predictive_ensemble + innovation @ kalman_gain.mT
        self.covariance = predictive_covariance - kalman_gain @ observe_matrix @ predictive_covariance
        return ensemble


class EnsembleKalmanFilterPerturbedObservations(Classical):
    def __init__(self, cfg, rng: np.random.Generator):
        super().__init__(cfg, rng)
        if self.cfg.use_torch_rng_for_perturbed_observations:
            self._torch_noise_base_seed = int(self.rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            self._torch_noise_generators = {}

    def assimilate(self, time_step, predictive_ensemble, observation, obs_noise_std, observe):
        localization_matrix_Lxy = None
        localization_matrix_Lyy = None
        if self.cfg.gaspari_cohn_localization_radius is not None:
            coords_state = self.coords(predictive_ensemble.shape[1], device=predictive_ensemble.device, dtype=predictive_ensemble.dtype)
            coords_observation = self.coords(observation.shape[1], device=predictive_ensemble.device, dtype=predictive_ensemble.dtype)
            domain_lengths = self.domain_lengths(predictive_ensemble.shape[1], device=predictive_ensemble.device, dtype=predictive_ensemble.dtype)
            localization_matrix_Lxy = self.L_thing(
                coords_state, coords_observation, self.cfg.gaspari_cohn_localization_radius,
                domain_lengths=domain_lengths,
            )
            localization_matrix_Lyy = self.L_thing(
                coords_observation, coords_observation, self.cfg.gaspari_cohn_localization_radius,
                domain_lengths=domain_lengths,
            )
        torch_noise_generator = None
        if self.cfg.use_torch_rng_for_perturbed_observations:
            torch_noise_generator = dafm.utils.get_torch_rng_generator(
                self._torch_noise_base_seed,
                self._torch_noise_generators,
                predictive_ensemble.device,
            )

        ensemble = dafm.filters_classical.ensemble_kalman_filter_analysis(
            rng=self.rng,
            ensemble_f=predictive_ensemble,
            observation_y=observation.squeeze(0),
            observation_operator_ens=observe,
            sigma_y=obs_noise_std,
            method='EnKF-PertObs',
            do_inflation=False,
            localization_matrix_Lxy=localization_matrix_Lxy,
            localization_matrix_Lyy=localization_matrix_Lyy,
            torch_noise_generator=torch_noise_generator,
        )[0]
        ensemble = self.inflate(ensemble)
        return ensemble


class EnsembleKalmanFilterPerturbedObservationsIterative(Classical):
    def __init__(self, cfg, rng: np.random.Generator):
        super().__init__(cfg, rng)
        self._torch_noise_base_seed = int(self.rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        self._torch_noise_generators = {}

    def initialize(self, cfg_data_assimilation_setting, initial_ensemble, dynamics):
        self.dataset = dynamics
        self.observe_every_n_time_steps = cfg_data_assimilation_setting.observe_every_n_time_steps

    def assimilate(self, time_step, predictive_ensemble, observation, obs_noise_std, observe):
        ensemble = dafm.filters_classical_iterative.ensemble_kalman_filter_analysis(
            ensemble_f=predictive_ensemble[None],
            observation_y=observation.squeeze(0),
            observation_operator_ens=observe,
            sigma_y=obs_noise_std,
            method='iEnKS-PertObs',
            inflation_factor=1.,
            ienks_lag=self.cfg.lag,
            ienks_niter=self.cfg.niter,
            ienks_wtol=self.cfg.wtol,
            torch_noise_generator=dafm.utils.get_torch_rng_generator(
                self._torch_noise_base_seed,
                self._torch_noise_generators,
                predictive_ensemble.device,
            ),
            model_args=dict(
                propagator=lambda _, ensemble, __: self.dataset.unflatten_integrate(time_step, time_step + 1, ensemble)[-1],
                rhs=None, dt=None,  # handled by self.dataset.unflatten_integrate
                steps_between_analyses=self.observe_every_n_time_steps,
            ),
        )[0][0]
        ensemble = self.inflate(ensemble)
        return ensemble


class EnsembleRandomizedSquareRootFilter(Classical):
    def assimilate(self, time_step, predictive_ensemble, observation, obs_noise_std, observe):
        ensemble = dafm.filters_classical.ensemble_kalman_filter_analysis(
            rng=self.rng,
            ensemble_f=predictive_ensemble,
            observation_y=observation.squeeze(0),
            observation_operator_ens=observe,
            sigma_y=obs_noise_std,
            method='ERSF',
            do_inflation=False,
        )[0]
        ensemble = self.inflate(ensemble)
        return ensemble


class LocalEnsembleTransformKalmanFilter(Classical):
    def assimilate(self, time_step, predictive_ensemble, observation, obs_noise_std, observe):
        coords_state = self.coords(predictive_ensemble.shape[1], device=predictive_ensemble.device, dtype=predictive_ensemble.dtype)
        coords_observation = self.coords(observation.shape[1], device=predictive_ensemble.device, dtype=predictive_ensemble.dtype)
        domain_lengths = self.domain_lengths(predictive_ensemble.shape[1], device=predictive_ensemble.device, dtype=predictive_ensemble.dtype)
        ensemble = dafm.filters_classical.ensemble_kalman_filter_analysis(
            rng=self.rng,
            ensemble_f=predictive_ensemble,
            observation_y=observation.squeeze(0),
            observation_operator_ens=observe,
            sigma_y=obs_noise_std,
            method='LETKF',
            localization_radius_letkf=self.cfg.gaspari_cohn_localization_radius,
            coords_state_letkf=coords_state,
            coords_obs_letkf=coords_observation,
            domain_lengths_letkf=domain_lengths,
            do_inflation=False,
        )[0]
        ensemble = self.inflate(ensemble)
        return ensemble


class EnsembleScoreFilter(Filter):
    def __init__(self, cfg, rng: np.random.Generator, prob_path):
        super().__init__(cfg, rng)
        self.prob_path = prob_path
        self._torch_noise_base_seed = int(self.rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        self._torch_noise_generators = {}

    def observation_likelihood_score_damping(self, t):
        return 1 - t

    def score(self, t, xt, means, std):
        conditional_prob_path = dafm.utils.Independent(dafm.utils.Normal(loc=means, scale=std, validate_args=False), 1)
        log_conditional_probs = conditional_prob_path.log_prob_unnormalized(xt)
        weights = self.log_prob_to_weight_safe(log_conditional_probs)
        conditional_scores = -(xt - means) / std.square()
        return reduce(weights[:, None] * conditional_scores, 'member dim -> dim', 'sum')

    def assimilate(self, time_step, predictive_ensemble, observation, obs_noise_std, observe):
        path_time = self.prob_path.linspace_time(self.cfg.sampling_time_step_count, device=predictive_ensemble.device)
        minus_time_step_size = path_time[1] - path_time[0]
        minus_time_step_size_abs_sqrt = minus_time_step_size.abs().sqrt()
        ensemblet = self.prob_path.sample_noise(path_time[0], predictive_ensemble)
        for time_step, t_now_and_next in enumerate(path_time.unfold(0, 2, 1)):
            t_now = t_now_and_next[0]
            means = self.prob_path.mean(t_now, predictive_ensemble)
            std = self.prob_path.std(t_now, predictive_ensemble)
            score = torch.vmap(
                lambda member: self.score(t_now, member, means, std),
                chunk_size=self.cfg.vmap_chunk_size,
            )(ensemblet)
            with torch.enable_grad():
                xt_grad = ensemblet.detach().requires_grad_()
                observation_likelihood_distribution = dafm.utils.Independent(
                    dafm.utils.Normal(observation, obs_noise_std), 1,
                )
                log_observation_likelihood = observation_likelihood_distribution.log_prob_unnormalized(observe(xt_grad))
                observation_score, *_ = torch.autograd.grad(
                    outputs=log_observation_likelihood.sum(),
                    inputs=xt_grad,
                )
            score = score + observation_score * self.observation_likelihood_score_damping(t_now)
            score = score.clamp(min=-self.cfg.sampling_max_score_norm, max=self.cfg.sampling_max_score_norm)
            g = self.prob_path.g(t_now)
            if self.cfg.sampler is conf.filter.Sampler.EULER:
                ensemblet = ensemblet + minus_time_step_size * (self.prob_path.f(t_now, ensemblet) - g.square() / 2 * score)
                ensemble = ensemblet
            elif self.cfg.sampler is conf.filter.Sampler.EULER_MARUYAMA:
                state_drift = ensemblet + minus_time_step_size * (self.prob_path.f(t_now, ensemblet) - g.square() * score)
                noise = torch.randn(
                    ensemblet.shape,
                    device=ensemblet.device,
                    dtype=ensemblet.dtype,
                    generator=dafm.utils.get_torch_rng_generator(
                        self._torch_noise_base_seed,
                        self._torch_noise_generators,
                        ensemblet.device,
                    ),
                )
                ensemblet = state_drift + minus_time_step_size_abs_sqrt * g * noise
                ensemble = ensemblet
            else:
                raise ValueError(f'Unsupported sampler: {self.cfg.sampler}')
        ensemble = self.inflate(ensemble)
        return ensemble


class EnsembleFlowFilter(Filter):
    def __init__(self, cfg, rng: np.random.Generator, prob_path, guide):
        super().__init__(cfg, rng)
        self.prob_path = prob_path
        self.guide = guide

    def velocity(self, t, xt, means, dt_means, std, dt_std):
        conditional_prob_path = dafm.utils.Independent(dafm.utils.Normal(loc=means, scale=std, validate_args=False), 1)
        log_conditional_probs = conditional_prob_path.log_prob_unnormalized(xt)
        weights = self.log_prob_to_weight_safe(log_conditional_probs)
        conditional_velocities = dt_std / std * (xt - means) + dt_means
        velocity = reduce(weights[:, None] * conditional_velocities, 'member dim -> dim', 'sum')
        return velocity

    def assimilate(self, time_step, predictive_ensemble, observation, obs_noise_std, observe):
        observation_likelihood_distribution = dafm.utils.Independent(
            dafm.utils.Normal(observation, obs_noise_std), 1,
        )
        path_time = self.prob_path.linspace_time(self.cfg.sampling_time_step_count, device=predictive_ensemble.device)
        time_step_size = path_time[1] - path_time[0]
        noise = self.prob_path.sample_noise(path_time[0], predictive_ensemble)
        ensemblet = noise
        for time_step, t_now_and_next in enumerate(path_time.unfold(0, 2, 1)):
            t_now, t_next = t_now_and_next
            means = self.prob_path.mean(t_now, predictive_ensemble)
            dt_means = self.prob_path.dt_mean(t_now, predictive_ensemble)
            std = self.prob_path.std(t_now, predictive_ensemble)
            dt_std = self.prob_path.dt_std(t_now, predictive_ensemble)
            velocity = torch.vmap(
                lambda xt: self.velocity(t_now, xt, means, dt_means, std, dt_std),
                chunk_size=self.cfg.vmap_chunk_size,
            )(ensemblet)
            guidance = self.guide(
                noise, predictive_ensemble, t_now, ensemblet, velocity,
                energy_function=lambda x1_predicted: -observation_likelihood_distribution.log_prob_unnormalized(observe(x1_predicted)),
            )
            if self.cfg.sampler is conf.filter.Sampler.EULER:
                ensemblet = ensemblet + time_step_size * (velocity + guidance)
                ensemble = ensemblet
            # elif self.cfg.sampler is conf.filter.Sampler.HEUN:
            #     xdot_now = velocity(t_now, ensemblet)
            #     temp = ensemblet + time_step_size * xdot_now
            #     ensemblet = ensemblet + time_step_size * (xdot_now + velocity(t_next, temp)) / 2
            #     ensemble = ensemblet
            else:
                raise ValueError(f'Unsupported sampler: {self.cfg.sampler}')
        ensemble = self.inflate(ensemble)
        return ensemble


def get_filter(cfg, *, cfg_dataset: conf.dataset.DynamicalSystemImpl = None, rng: np.random.Generator = None):
    match cfg:
        case conf.filter.AddNoiseToObservationFilter():
            if rng is None:
                raise ValueError('The "add noise to observation" filter requires a random number generator. '
                                 'Please pass the keyword argument rng.')
            return AddNoiseToObservationFilter(cfg, rng)
        case conf.filter.BootstrapParticleFilter():
            if rng is None:
                raise ValueError('The bootstrap particle filter requires a random number generator. '
                                 'Please pass the keyword argument rng.')
            return BootstrapParticleFilter(cfg, rng)
        case conf.filter.KalmanFilter():
            return KalmanFilter(cfg, rng)
        case conf.filter.EnsembleKalmanFilterPerturbedObservations():
            if rng is None:
                raise ValueError('The ensemble Kalman filter requires a random number generator. '
                                 'Please pass the keyword argument rng.')
            return EnsembleKalmanFilterPerturbedObservations(cfg, rng)
        case conf.filter.EnsembleKalmanFilterPerturbedObservationsIterative():
            return EnsembleKalmanFilterPerturbedObservationsIterative(cfg, rng)
        case conf.filter.EnsembleRandomizedSquareRootFilter():
            return EnsembleRandomizedSquareRootFilter(cfg, rng)
        case conf.filter.LocalEnsembleTransformKalmanFilter():
            return LocalEnsembleTransformKalmanFilter(cfg, rng)
        case conf.filter.EnsembleScoreFilter():
            if rng is None:
                raise ValueError('The ensemble score filter requires a random number generator. '
                                 'Please pass the keyword argument rng.')
            rng_prob_path, rng_filter = rng.spawn(2)
            prob_path = dafm.prob_path.get_prob_path(cfg.prob_path, rng_prob_path, target_distribution_at_time_1=False)
            return EnsembleScoreFilter(cfg, rng_filter, prob_path)
        case conf.filter.EnsembleFlowFilter():
            if rng is None:
                raise ValueError('The ensemble flow filter requires a random number generator. '
                                 'Please pass the keyword argument rng.')
            rng_prob_path, rng_filter = rng.spawn(2)
            prob_path = dafm.prob_path.get_prob_path(cfg.prob_path, rng_prob_path, target_distribution_at_time_1=True)
            guide = dafm.flow_matching_guidance.get_guidance(cfg.guidance, rng=rng_filter.spawn(1)[0])
            return EnsembleFlowFilter(cfg, rng_filter, prob_path, guide)
        case _:
            raise ValueError(f'Unknown filter: {cfg}')
