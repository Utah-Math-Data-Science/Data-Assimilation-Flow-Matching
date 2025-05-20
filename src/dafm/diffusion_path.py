import torch
import numpy as np

from conf import diffusion_path


class GaussianPath:
    """
    Gaussian conditional probability path.
    """
    def __init__(self, cfg, target_distribution_at_time_1=False):
        r"""
        Parameters
        ----------
        cfg: conf.diffusion_path.DiffusionPath
            The configuration of the Gaussian probability path.

        target_distribution_at_time_1: bool
            True if the probability path approximates the target distribution
            when :math:`t = 1` and is the noise distribution at :math:`t = 0`.
            Otherwise, the path approximates the target distribution when
            :math:`t = 0`.
        """
        self.cfg = cfg
        self.target_distribution_at_time_1 = target_distribution_at_time_1


class ConditionalOptimalTransport(GaussianPath):
    r"""
    This probability path approximates the target distribution when
    :math:`t = 1` and is the noise distribution at :math:`t = 0`.

    .. warning::
       Behavior for `target_distribution_at_time_1=False` is not implemented.
    """
    def sample_time(self, t_shape, **kwargs):
        return torch.rand(t_shape, **kwargs)

    def linspace_time(self, time_step_count, **kwargs):
        return torch.linspace(0., 1., time_step_count, **kwargs)

    def mean(self, t, data):
        """
        The mean of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        return t * data

    def dt_mean(self, t, data):
        return data

    def std(self, t, data):
        """
        The standard devation of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        return 1 - (1 - self.cfg.sigma_min) * t

    def dt_std(self, t, data):
        return t * 0 - 1 + self.cfg.sigma_min

    def sample_noise(self, t, data):
        """
        Sample noise with mean zero and standard deviation at time :math:`t`.
        This method makes sense when :math:`t` is the time where the
        probability path is closest to the noise distribution.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        return torch.randn_like(data) * self.std(t, data)


class PreviousPosteriorToPredictive(GaussianPath):
    r"""
    This probability path approximates the target (predictive) distribution
    when :math:`t = 1` and is the previous posterior distribution at
    :math:`t = 0`.

    .. warning::
       Behavior for `target_distribution_at_time_1=False` is not implemented.
    """
    previous_posterior = None

    def set_previous_posterior(self, previous_posterior):
        """
        Save sample from the previous posterior to be used to define the mean
        of the probability path.

        Parameters
        ----------
        previous_posterior: torch.Tensor
            Sample from the previous posterior.
        """
        self.previous_posterior = previous_posterior

    def sample_time(self, t_shape, **kwargs):
        return torch.rand(t_shape, **kwargs)

    def linspace_time(self, time_step_count, **kwargs):
        return torch.linspace(0., 1., time_step_count, **kwargs)

    def mean(self, t, predictive):
        """
        The mean of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        predictive: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        return t * predictive + (1 - t) * self.previous_posterior

    def dt_mean(self, t, predictive):
        return predictive - self.previous_posterior

    def std(self, t, predictive):
        """
        The standard devation of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        predictive: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        return self.cfg.sigma_min

    def dt_std(self, t, predictive):
        return 0.

    def sample_noise(self, t, predictive):
        """
        Sample noise with mean zero and standard deviation at time :math:`t`.
        This method makes sense when :math:`t` is the time where the
        probability path is closest to the noise distribution.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        predictive: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        previous_posterior = self.previous_posterior
        if self.cfg.use_independent_coupling:
            shuffle = torch.randperm(previous_posterior.shape[0], device=previous_posterior.device)
            previous_posterior = previous_posterior[shuffle]
        return previous_posterior + torch.randn_like(predictive) * self.std(t, predictive)


class VarianceExploding(GaussianPath):
    r"""
    This probability path approximates the target distribution when
    :math:`t = 0` and is the noise distribution at :math:`t = 1`.
    The probability path can be reversed by setting
    `target_distribution_at_time_1=True`.
    """
    def _reverse_time(self, t):
        r"""
        Reflects the time `t` in the interval :math:`[t_\min, 1]`.
        """
        return 1 - t + self.cfg.time_min

    def sample_time(self, t_shape, **kwargs):
        return torch.rand(t_shape, **kwargs) * (1 - self.cfg.time_min) + self.cfg.time_min

    def linspace_time(self, time_step_count, **kwargs):
        t = torch.linspace(1., self.cfg.time_min, time_step_count, **kwargs)
        if self.target_distribution_at_time_1:
            t = self._reverse_time(t)
        return t

    def mean(self, t, data):
        """
        The mean of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        return data

    def dt_mean(self, t, data):
        return data * 0

    def std(self, t, data):
        r"""
        The standard devation of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor

        Notes
        -----
        Eqn.(30) of [1]_ modified to be continuous as the probability path
        approaches the target distribution.
        But, we divide by :math:`2*\log(\sigma_\max/\sigma_\min)`? Why?
        This appears to simplify the form of :math:`g(t)`.

        References
        ----------
        .. [1] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
           Score-Based Generative Modeling through Stochastic Differential Equations
           (No. arXiv:2011.13456). arXiv. http://arxiv.org/abs/2011.13456
        """
        if self.target_distribution_at_time_1:
            t = self._reverse_time(t)
        return self.cfg.sigma_min * (
            ((self.cfg.sigma_max / self.cfg.sigma_min)**(2 * t) - 1)
            / 2
            / np.log(self.cfg.sigma_max / self.cfg.sigma_min)
        )**(1/2)

    def dt_std(self, t, data):
        change_of_time_variable = 1
        if self.target_distribution_at_time_1:
            t = self._reverse_time(t)
            change_of_time_variable = -1
        return t * 0 + change_of_time_variable * (
            self.cfg.sigma_min * (
                self.cfg.sigma_max / self.cfg.sigma_min
            )**(2 * t) / (
                ((self.cfg.sigma_max / self.cfg.sigma_min)**(2 * t) - 1)
                * 2
                / np.log(self.cfg.sigma_max / self.cfg.sigma_min)
            )**(1/2)
        )

    def f(self, t, data):
        return 0.

    def g(self, t):
        r"""
        :math:`\sqrt{\frac{\mathrm{d}}{\mathrm{d}t} \sigma_t^2}`
        """
        if self.target_distribution_at_time_1:
            t = self._reverse_time(t)
        return self.cfg.sigma_min * (self.cfg.sigma_max / self.cfg.sigma_min)**t

    def sample_noise(self, t, data):
        """
        Sample noise with mean zero and standard deviation at time :math:`t`.
        This method makes sense when :math:`t` is the time where the
        probability path is closest to the noise distribution.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        return torch.randn_like(data) * diffusion_path.std(t, data)


class Bao2024EnsembleScoreMatching(GaussianPath):
    def _reverse_time(self, t):
        r"""
        Reflects the time `t` in the interval :math:`[t_\min, 1]`.
        """
        return 1 - t

    def sample_time(self, t_shape, **kwargs):
        return torch.rand(t_shape, **kwargs)

    def linspace_time(self, time_step_count, **kwargs):
        t = torch.linspace(1., 0., time_step_count, **kwargs)
        if self.target_distribution_at_time_1:
            t = self._reverse_time(t)
        return t

    def alpha(self, t):
        return 1 - t * (1 - self.cfg.epsilon_alpha)

    def dt_log_alpha(self, t):
        return -(1 - self.cfg.epsilon_alpha) / self.alpha(t)

    def beta(self, t):
        return (self.cfg.epsilon_beta + t * (1 - self.cfg.epsilon_beta))**(1/2)

    def dt_squared_beta(self, t):
        return 1 - self.cfg.epsilon_beta

    def mean(self, t, data):
        """
        The mean of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        if self.target_distribution_at_time_1:
            t = self._reverse_time(t)
        return self.alpha(t) * data

    def dt_mean(self, t, data):
        raise NotImplementedError()

    def std(self, t, data):
        r"""
        The standard devation of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        if self.target_distribution_at_time_1:
            t = self._reverse_time(t)
        return self.beta(t)

    def dt_std(self, t, data):
        raise NotImplementedError()

    def f(self, t, data):
        return self.dt_log_alpha(t) * data

    def g(self, t):
        if self.target_distribution_at_time_1:
            raise NotImplementedError()
        return (self.dt_squared_beta(t) - 2 * self.dt_log_alpha(t) * self.beta(t).square()).sqrt()

    def sample_noise(self, t, data):
        """
        Sample noise with mean zero and standard deviation at time :math:`t`.
        This method makes sense when :math:`t` is the time where the
        probability path is closest to the noise distribution.

        Parameters
        ----------
        t: torch.Tensor
            Time along the probability path.

        data: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        if self.target_distribution_at_time_1:
            raise NotImplementedError()
        noise = torch.randn_like(data)
        if self.cfg.renormalize_sampled_noise:
            noise = (noise - noise.mean(0)) / noise.std(0)
        if self.cfg.sample_noise_scale_std:
            noise = noise * self.std(t, data)
        if self.cfg.sample_noise_add_mean:
            noise = self.mean(t, data) + noise
        return noise


def get_diffusion_path(cfg, target_distribution_at_time_1=False):
    if isinstance(cfg, diffusion_path.ConditionalOptimalTransport):
        return ConditionalOptimalTransport(cfg, target_distribution_at_time_1=target_distribution_at_time_1)
    if isinstance(cfg, diffusion_path.PreviousPosteriorToPredictive):
        return PreviousPosteriorToPredictive(cfg, target_distribution_at_time_1=target_distribution_at_time_1)
    elif isinstance(cfg, diffusion_path.VarianceExploding):
        return VarianceExploding(cfg, target_distribution_at_time_1=target_distribution_at_time_1)
    elif isinstance(cfg, diffusion_path.Bao2024EnsembleScoreMatching):
        return Bao2024EnsembleScoreMatching(cfg, target_distribution_at_time_1=target_distribution_at_time_1)
    else:
        raise ValueError(f'Unknown diffusion path: {cfg}')
