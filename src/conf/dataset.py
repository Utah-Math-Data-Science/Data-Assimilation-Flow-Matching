from dataclasses import field
from pathlib import Path
from typing import Any, List

import hydra_orm
import hydra_orm.utils
from hydra_orm import orm
import omegaconf
from omegaconf import II
import sqlalchemy as sa

import dafm.utils


def make_defaults_list_no_compare(defaults):
    f = hydra_orm.utils.make_defaults_list(defaults)
    f.compare = False
    return f


class DynamicalSystem(orm.InheritableTable):
    time_step_size: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)

    @property
    def channels(self):
        return 1

    @property
    def spatial_dims(self):
        raise NotImplementedError()


class UnitTestDataset(DynamicalSystem):
    @property
    def channels(self):
        return 1

    @property
    def spatial_dims(self):
        return [1, 1]


class Rotation2D(DynamicalSystem):
    radius: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)

    @property
    def channels(self):
        return 1

    @property
    def spatial_dims(self):
        return [2]


class NavierStokes2DBackwardFacingStepGLED(DynamicalSystem):
    gled_filepath: str = field(default='/mnta/taosData/diffusion-dynamics/G-LED/data/data_cat_f32.pt')

    grid_horizontal_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    grid_vertical_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    grid_width: float = orm.make_field(orm.ColumnRequired(sa.Double), default=10.0)
    grid_height: float = orm.make_field(orm.ColumnRequired(sa.Double), default=2.0)

    @property
    def channels(self):
        return 2

    @property
    def spatial_dims(self):
        return [self.grid_horizontal_count, self.grid_vertical_count]


class NavierStokes2DPeriodicBoundary(DynamicalSystem):
    grid_horizontal_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    grid_vertical_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=512)
    grid_width: float = orm.make_field(orm.ColumnRequired(sa.Double), default=2.0)
    grid_height: float = orm.make_field(orm.ColumnRequired(sa.Double), default=2.0)

    forcing_amplitude: float = orm.make_field(orm.ColumnRequired(sa.Double), default=5e-2)
    vertical_mode_number: float = orm.make_field(orm.ColumnRequired(sa.Double), default=8.0)
    viscosity: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1e-3)

    pressure_poisson_solve_iteration_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=100)

    gaussian_process_length_scale: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.2)
    gaussian_process_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.0)

    @property
    def channels(self):
        return 3

    @property
    def spatial_dims(self):
        return [self.grid_horizontal_count, self.grid_vertical_count]


class Lorenz63(DynamicalSystem):
    true_state_initial_condition_mean: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)
    true_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)

    sigma: float = orm.make_field(orm.ColumnRequired(sa.Double), default=10.0)
    rho: float = orm.make_field(orm.ColumnRequired(sa.Double), default=28.0)
    beta: float = orm.make_field(orm.ColumnRequired(sa.Double), default=8 / 3)
    rescaling: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.0)

    @property
    def spatial_dims(self):
        return [3]


class Lorenz96(DynamicalSystem):
    true_state_initial_condition_mean: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)
    true_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)

    dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    forcing: float = orm.make_field(orm.ColumnRequired(sa.Double), default=8.0)

    @property
    def spatial_dims(self):
        return [self.dimension]


class KuramotoSivashinsky(DynamicalSystem):
    dimension: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    domain_pi_multiple: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)

    @property
    def spatial_dims(self):
        return [self.dimension]


class DynamicalSystemImpl(orm.Table):
    defaults: List[Any] = make_defaults_list_no_compare([
        dict(system=omegaconf.MISSING),
        '_self_',
    ])

    system = orm.OneToManyField(DynamicalSystem, required=True, default=omegaconf.MISSING)

    data_dir: str = field(default=str((dafm.utils.DIR_ROOT / '..' / '..' / 'out' / 'revision-dafm' / 'datasets').resolve()))

    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=II('oc.select:rng_seed,0'))
    processed_code: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True), init=False, omegaconf_ignore=True)

    recompute_trajectory: bool = field(default=False)
    time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)

    @property
    def processed_filepath(self):
        return Path(self.data_dir) / f'{self.processed_code}.pt'

    @property
    def channels(self):
        return self.system.channels

    @property
    def spatial_dims(self):
        return self.system.spatial_dims

    @property
    def time_step_size(self):
        return self.system.time_step_size


@sa.event.listens_for(DynamicalSystemImpl, 'before_insert')
def set_dataset_processed_code(mapper, connection, target: DynamicalSystemImpl):
    processed_dataset = sa.select(DynamicalSystemImpl.processed_code).where(
        DynamicalSystemImpl.system == target.system,
        DynamicalSystemImpl.rng_seed == target.rng_seed,
    ).distinct()
    processed_dataset = connection.execute(processed_dataset)
    processed_dataset = list(zip(range(2), processed_dataset))
    assert len(processed_dataset) <= 1
    if len(processed_dataset) == 1:
        target.processed_code = processed_dataset[0][1][0]
    else:
        hydra_orm.utils.set_attr_to_func_value(
            DynamicalSystemImpl,
            DynamicalSystemImpl.processed_code.key,
            hydra_orm.utils.generate_random_string,
        )(mapper, connection, target)
