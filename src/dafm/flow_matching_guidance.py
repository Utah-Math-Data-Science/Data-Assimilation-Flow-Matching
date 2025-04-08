import torch
import torch.distributions
from einops import rearrange, reduce

from conf import flow_matching_guidance


class EnergyGuidance:
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, t, xt, x0, x1, dot_xt_unguided, energy_function):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class No(EnergyGuidance):
    def forward(self, t, xt, x0, x1, dot_xt_unguided, energy_function):
        return 0.


class MonteCarlo(EnergyGuidance):
    def forward(self, t, xt, x0, x1, dot_xt_unguided, energy_function):
        mean = x1 * t
        std = 1 - t
        noise_flowed_to_t = mean + x0 * std
        # using Independent(Normal, 1) instead of MultivariateNormal is a trick
        # to specify the covariance matrix as scale * (identity matrix)
        conditional_distribution = torch.distributions.Independent(
            # authors used scale=.1, saying that when scale is too small, many more samples are needed
            torch.distributions.Normal(
                loc=rearrange(noise_flowed_to_t, 'predicted_state_count dim -> predicted_state_count 1 dim'),
                scale=(1 - t).clamp(min=self.cfg.time_min)
            ),
            1,
        )
        log_pt_xt_given_z = rearrange(
            conditional_distribution.log_prob(
                rearrange(xt, 'predicted_state_count dim -> 1 predicted_state_count dim')
            ),
            'monte_carlo_sample_count predicted_state_count -> monte_carlo_sample_count predicted_state_count 1',
        )
        log_samples = torch.tensor(xt.shape[0], device=xt.device).log()
        log_pt_x = reduce(
            log_pt_xt_given_z,
            'monte_carlo_sample_count predicted_state_count 1 -> predicted_state_count 1 1',
            torch.logsumexp,
        ) - log_samples
        neg_energy = rearrange(
            -energy_function(x1),
            'predicted_state_count 1 -> predicted_state_count 1 1',
        )
        Z = torch.exp(
            reduce(
                neg_energy + log_pt_xt_given_z,
                'monte_carlo_sample_count predicted_state_count 1 -> predicted_state_count 1 1',
                torch.logsumexp,
            ) - log_samples - log_pt_x
        )
        v_xt_given_z = rearrange(
            x1 - x0,
            'predicted_state_count dim -> predicted_state_count 1 dim',
        )
        return reduce(
            (neg_energy.exp() / Z - 1)
            * v_xt_given_z
            * (log_pt_xt_given_z - log_pt_x).exp(),
            'monte_carlo_sample_count predicted_state_count dim -> predicted_state_count dim',
            'mean',
        )


class Local(EnergyGuidance):
    def __init__(self, cfg, scheduler):
        super().__init__(cfg)
        self.scheduler = scheduler

    def forward(self, t, xt, x0, x1, dot_xt_unguided, energy_function):
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
        return MonteCarlo(cfg)
    elif isinstance(cfg, flow_matching_guidance.Local):
        schedule = get_schedule(cfg.schedule)
        return Local(cfg, schedule)
