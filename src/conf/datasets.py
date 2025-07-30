import enum

from typing import List, Any, Optional
from dataclasses import field

import omegaconf
from omegaconf import II
import hydra_orm.utils
from hydra_orm import orm
import sqlalchemy as sa

import conf.observe


class Integrator(enum.Enum):
    RUNGE_KUTTA_4 = enum.auto()
    EULER_MARUYAMA = enum.auto()


class StatePerturbation(enum.Enum):
    IDENTITY = enum.auto()
    BAO_ET_AL_DOUBLE_WELL = enum.auto()
    BAO_ET_AL_LORENZ_96 = enum.auto()


class Dataset(orm.InheritableTable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(observe='Full'),
        '_self_',
    ])
    trajectory_stored_on_gpu_max_state_dimension: int = field(default=200000)
    save_data_every_n_time_steps: Optional[int] = field(default=None)

    state_dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)

    model_noise_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)
    observation_noise_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)
    predicted_state_initial_condition_add_true_state: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    predicted_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)
    predicted_state_model_noise_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=II('.model_noise_std'))

    predicted_state_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)

    time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2)
    time_step_count_drop_first: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=0)
    time_step_size: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.1)

    observe_every_n_time_steps: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1)
    observe: conf.observe.Observe = orm.OneToManyField(conf.observe.Observe, default=omegaconf.MISSING)

    integrator: Integrator = orm.make_field(orm.ColumnRequired(sa.Enum(Integrator)), default=Integrator.EULER_MARUYAMA)

    state_perturbation: StatePerturbation = orm.make_field(orm.ColumnRequired(sa.Enum(StatePerturbation)), default=StatePerturbation.IDENTITY)

    use_predicted_state_perturbation: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    predicted_state_perturbation_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)

    @property
    def save_only_mean_std(self):
        return self.state_dimension >= 100#and self.predicted_state_count > 1


class Simple(Dataset):
    pass


class DoubleWell(Dataset):
    true_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.02)


class Lorenz63(Dataset):
    true_state_initial_condition_mean: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)
    true_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)

    sigma: float = orm.make_field(orm.ColumnRequired(sa.Double), default=28.)
    rho: float = orm.make_field(orm.ColumnRequired(sa.Double), default=10.)
    beta: float = orm.make_field(orm.ColumnRequired(sa.Double), default=8/3)
    rescaling: float = orm.make_field(orm.ColumnRequired(sa.Double), default=20.)


class Lorenz96(Dataset):
    true_state_initial_condition_mean: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)
    true_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)

    forcing: float = orm.make_field(orm.ColumnRequired(sa.Double), default=8.)

    def __post_init__(self):
        if self.state_dimension < 3:
            raise ValueError(f"The dimension of Lorenz '96 must be at least 4, not {self.state_dimension}.")


class NavierStokes(Dataset):
    grid_horizontal_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=128)
    grid_vertical_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=128)
    grid_width: float = orm.make_field(orm.ColumnRequired(sa.Double), default=2.)
    grid_height: float = orm.make_field(orm.ColumnRequired(sa.Double), default=2.)

    forcing_amplitude: float = orm.make_field(orm.ColumnRequired(sa.Double), default=5e-2)
    vertical_mode_number: float = orm.make_field(orm.ColumnRequired(sa.Double), default=8.)
    viscosity: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)

    pressure_poisson_solve_iteration_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=100)

    gaussian_process_length_scale: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.2)
    gaussian_process_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)


class KuramotoSivashinsky(Dataset):
    domain_pi_multiple: float = orm.make_field(orm.ColumnRequired(sa.Double), default=128)
    floating_point_precision: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=32)
