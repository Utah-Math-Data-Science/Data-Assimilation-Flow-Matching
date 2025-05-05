import torch
import torch.distributions
from einops import rearrange, reduce

from conf import flow_matching_guidance
import dafm.diffusion_path
from dafm import utils


class EnergyGuidance:
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, x0, x1, t, xt, dot_xt_unguided, energy_function):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class No(EnergyGuidance):
    def forward(self, x0, x1, t, xt, dot_xt_unguided, energy_function):
        return 0.


class MonteCarlo(EnergyGuidance):
    def __init__(self, cfg, diffusion_path):
        super().__init__(cfg)
        self.diffusion_path = diffusion_path

    def conditional_velocity(self, mean, dt_mean, std, dt_std, time, x):
        return dt_std / std * (x - mean) + dt_mean

    def forward(self, _x0, x1, t, xt, dot_xt_unguided, energy_function):
        # monte_carlo_sample_count = x1.shape[0]
        x1 = rearrange(x1, 'monte_carlo_sample_count dim -> monte_carlo_sample_count 1 dim')
        xt = rearrange(xt, 'predicted_state_count dim -> 1 predicted_state_count dim')

        mean = self.diffusion_path.mean(t, x1)
        std = self.diffusion_path.std(t, x1)
        # using Independent(Normal, 1) instead of MultivariateNormal is a trick
        # to specify the covariance matrix as scale * (identity matrix)
        pt_xt_given_z = utils.Independent(
            # authors used scale=.1, saying that when scale is too small, many more samples are needed
            utils.Normal(loc=mean, scale=std),
            1,
        )
        log_pt_xt_given_z = rearrange(
            pt_xt_given_z.log_prob_unnormalized(xt),
            'monte_carlo_sample_count predicted_state_count -> monte_carlo_sample_count predicted_state_count 1',
        )
        # log_samples = torch.tensor(monte_carlo_sample_count, device=xt.device).log()
        log_sample_count_times_pt_xt_given_z_div_pt_x = log_pt_xt_given_z.log_softmax(0)
        neg_energy = rearrange(
            -energy_function(rearrange(x1, 'monte_carlo_sample_count 1 dim -> monte_carlo_sample_count dim')),
            'monte_carlo_sample_count 1 -> monte_carlo_sample_count 1 1',
        )
        log_Z = reduce(
            neg_energy + log_sample_count_times_pt_xt_given_z_div_pt_x,
            'monte_carlo_sample_count predicted_state_count 1 -> 1 predicted_state_count 1',
            torch.logsumexp,
        )
        v_xt_given_z = self.conditional_velocity(
            mean, self.diffusion_path.dt_mean(t, x1),
            std, self.diffusion_path.dt_std(t, x1),
            t, xt
        )
        return reduce(
            (neg_energy - log_Z).expm1()
            * v_xt_given_z
            * (log_sample_count_times_pt_xt_given_z_div_pt_x).exp(),
            'monte_carlo_sample_count predicted_state_count dim -> predicted_state_count dim',
            'sum',
        )


class Local(EnergyGuidance):
    def __init__(self, cfg, scheduler):
        super().__init__(cfg)
        self.scheduler = scheduler

    def forward(self, x0, x1, t, xt, dot_xt_unguided, energy_function):
        x1_predicted = xt + (1 - t) * dot_xt_unguided
        with torch.enable_grad():
            x1_predicted.requires_grad_()
            grad_energy, *_ = torch.autograd.grad(
                outputs=energy_function(x1_predicted).sum(),
                inputs=x1_predicted,
            )
        return -self.scheduler(t) * grad_energy


def get_schedule(cfg):
    if isinstance(cfg, flow_matching_guidance.Constant):
        return lambda t: cfg.constant
    else:
        raise ValueError(f'Unknown schedule: {cfg}')


def get_guidance(cfg):
    if isinstance(cfg, flow_matching_guidance.No):
        return No(cfg)
    elif isinstance(cfg, flow_matching_guidance.MonteCarlo):
        diffusion_path = dafm.diffusion_path.get_diffusion_path(
            cfg.diffusion_path,
            target_distribution_at_time_1=True  # always flow matching model
        )
        return MonteCarlo(cfg, diffusion_path)
    elif isinstance(cfg, flow_matching_guidance.Local):
        schedule = get_schedule(cfg.schedule)
        return Local(cfg, schedule)
