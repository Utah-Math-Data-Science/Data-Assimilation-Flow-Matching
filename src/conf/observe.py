import enum
import omegaconf
from hydra_orm import orm
import sqlalchemy as sa


class Observe(orm.InheritableTable):
    order: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)

    def __lt__(self, other):
        return self.order < other.order


class ObserveIdentity(Observe):
    pass


class ObserveATan(Observe):
    pass


class ObserveTanh(Observe):
    pass


class ObservePow(Observe):
    power: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)


class PatchMask(str, enum.Enum):
    CENTER_LEFT = 'CENTER_LEFT'
    CENTER_TOP = 'CENTER_TOP'
    LEFT = 'LEFT'
    TOP = 'TOP'
    DIAGONAL = 'DIAGONAL'


class ObserveMaskedPatches(Observe):
    patch_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    fliplr: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    flipud: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)
    patch_mask: PatchMask = orm.make_field(orm.ColumnRequired(sa.Enum(PatchMask)), default=omegaconf.MISSING)

    def __post_init__(self):
        self.patch_mask = PatchMask(self.patch_mask)
        if self.patch_mask in (PatchMask.CENTER_LEFT, PatchMask.LEFT) and self.flipud:
            raise ValueError(f'The patch mask {self.patch_mask.name} is invariant under up-down reflections. '
                             'Please set flipud=false.')
        if self.patch_mask in (PatchMask.CENTER_TOP, PatchMask.TOP) and self.fliplr:
            raise ValueError(f'The patch mask {self.patch_mask.name} is invariant under left-right reflections. '
                             'Please set flipud=false.')
        if self.patch_mask is PatchMask.DIAGONAL and self.flipud:
            raise ValueError('Please set fliplr=true and flipud=false to flip the DIAGONAL patch mask.')

    def validate_patch_count(self, spatial_dims):
        if len(spatial_dims) != 2:
            raise ValueError(f'ObserveMaskedPatches is only implemented for a spatial dimension of 2, not {len(spatial_dims)}')
        if self.patch_count <= 1:
            raise ValueError(f'The patch count must be greater than 1, not {self.patch_count}')
        if not all(d % self.patch_count for d in spatial_dims):
            raise ValueError(f'The patch cout must divide both of {spatial_dims}.')


class ObserveRandomDimensions(Observe):
    probability: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)
