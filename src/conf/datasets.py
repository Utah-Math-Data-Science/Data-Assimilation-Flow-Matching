from hydra_orm import orm
import sqlalchemy as sa


class Dataset(orm.InheritableTable):
    predicted_state_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=1000)
    time_step_count: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=100)
    time_step_size: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.1)


class DoubleWell(Dataset):
    model_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.2)
    observation_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.1)
    true_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.02)
    predicted_state_initial_condition_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.2)
    predicted_state_model_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)
