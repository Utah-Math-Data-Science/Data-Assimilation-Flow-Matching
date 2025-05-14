import sqlalchemy as sa
from hydra_orm import orm


class DiffusionPath(orm.InheritableTable):
    pass


class ConditionalOptimalTransport(DiffusionPath):
    sigma_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)


class PreviousPosteriorToPredictive(DiffusionPath):
    sigma_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)
    use_independent_coupling: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)


class VarianceExploding(DiffusionPath):
    time_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)
    sigma_min: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)
    sigma_max: float = orm.make_field(orm.ColumnRequired(sa.Double), default=25.0)


class Bao2024EnsembleScoreMatching(DiffusionPath):
    epsilon_alpha: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.5)
    epsilon_beta: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.025)
    renormalize_sampled_noise: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    sample_noise_add_mean: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    sample_noise_scale_std: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
