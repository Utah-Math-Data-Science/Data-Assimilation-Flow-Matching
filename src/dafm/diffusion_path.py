import torch
import numpy as np


from conf import diffusion_path


class GaussianPath:
    def __init__(self, cfg):
        self.cfg = cfg


class ConditionalOptimalTransport(GaussianPath):
    def sample_time(self, t_shape, **kwargs):
        return torch.rand(t_shape, **kwargs)

    def linspace_time(self, time_step_count, **kwargs):
        return torch.linspace(0., 1., time_step_count, **kwargs)

    def mean(self, t, x1):
        """
        The mean of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time in :math:`[0, 1]` along the probability path which is
            approximately the target distribution when :math:`t = 1` and is the
            noise distribution at :math:`t = 0`.

        x1: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        return t * x1

    def std(self, t, x1):
        """
        The standard devation of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time in :math:`[0, 1]` along the probability path which is
            approximately the target distribution when :math:`t = 1` and is the
            noise distribution at :math:`t = 0`.

        x1: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor

        Notes
        -----
        Eqn.(30) of [1]_ modified to be continuous as t -> 0.
        But, we divide by 2*log(sigma_max/sigma_min)? Why?

        References
        ----------
        .. [1] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
           Score-Based Generative Modeling through Stochastic Differential Equations
           (No. arXiv:2011.13456). arXiv. http://arxiv.org/abs/2011.13456
        """
        return 1 - (1 - self.cfg.sigma_min) * t

    def dt_std(self, t, x1):
        return torch.full_like(t, -1 + self.cfg.sigma_min)


class VarianceExploding(GaussianPath):
    def sample_time(self, t_shape, **kwargs):
        return torch.rand(t_shape, **kwargs) * (1 - self.cfg.diffusion_path.time_min) + self.cfg.diffusion_path.time_min

    def linspace_time(self, time_step_count, **kwargs):
        return torch.linspace(1., self.cfg.diffusion_path.time_min, time_step_count, **kwargs)

    def mean(self, t, x0):
        """
        The mean of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time in :math:`[0, 1]` along the probability path which is
            approximately the target distribution when :math:`t = 0` and is the
            noise distribution at :math:`t = 1`.

        x0: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor
        """
        return x0

    def std(self, t, x0):
        """
        The standard devation of the Gaussian conditional probability path.

        Parameters
        ----------
        t: torch.Tensor
            Time in :math:`[0, 1]` along the probability path which is
            approximately the target distribution when :math:`t = 0` and is the
            noise distribution at :math:`t = 1`.

        x0: torch.Tensor
            Sample from the target distribution.

        Returns
        -------
        torch.Tensor

        Notes
        -----
        Eqn.(30) of [1]_ modified to be continuous as t -> 0.
        But, we divide by 2*log(sigma_max/sigma_min)? Why?

        References
        ----------
        .. [1] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
           Score-Based Generative Modeling through Stochastic Differential Equations
           (No. arXiv:2011.13456). arXiv. http://arxiv.org/abs/2011.13456
        """
        return self.cfg.sigma_min * (
            ((self.cfg.sigma_max / self.cfg.sigma_min)**(2 * t) - 1)
            / 2
            / np.log(self.cfg.sigma_max / self.cfg.sigma_min)
        )**(1/2)

    def dt_std(self, t, x0):
        return torch.full_like(
            t,
            self.cfg.sigma_min * (
                self.cfg.sigma_max / self.cfg.sigma_min
            )**(2 * t) / (
                ((self.cfg.sigma_max / self.cfg.sigma_min)**(2 * t) - 1)
                * 2
                / np.log(self.cfg.sigma_max / self.cfg.sigma_min)
            )**(1/2)
        )

    def g(self, t):
        # this is sqrt(d/dt (self.sigma(t))^2)
        return self.cfg.diffusion_path.sigma_min * (self.cfg.diffusion_path.sigma_max / self.cfg.diffusion_path.sigma_min)**t


def get_diffusion_path(cfg):
    if isinstance(cfg, diffusion_path.ConditionalOptimalTransport):
        return ConditionalOptimalTransport(cfg)
    elif isinstance(cfg, diffusion_path.VarianceExploding):
        return VarianceExploding(cfg)
    else:
        raise ValueError(f'Unknown diffusion path: {cfg}')
