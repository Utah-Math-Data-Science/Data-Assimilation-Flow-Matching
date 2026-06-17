import enum
from bisect import bisect_left
from typing import List, Any, Optional, Tuple
from dataclasses import field
import itertools
import sys
from pathlib import Path

import sqlalchemy as sa
import omegaconf
from omegaconf import OmegaConf
import hydra
import hydra_orm.utils
from hydra_orm import orm

import conf.dataset
import conf.filter
import conf.flow_matching_guidance
import conf.observe
import conf.splitter
import dafm.utils


def get_engine(dir=str(dafm.utils.DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


engine = get_engine()
orm.create_all(engine)
Session = sa.orm.sessionmaker(engine)


def get_run_dir(hydra_init=dafm.utils.HYDRA_INIT, commit=True, engine_name='runs'):
    if '-m' in sys.argv or '--multirun' in sys.argv:
        raise ValueError("The flags '-m' and '--multirun' are not supported. Use GNU parallel instead.")
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = get_engine(name=engine_name)
        orm.create_all(engine)
        with sa.orm.Session(engine, expire_on_commit=False) as db:
            cfg = orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            # if commit and '-c' not in sys.argv:
            if commit:
                db.commit()
                cfg.run_dir.mkdir(exist_ok=True)
            return last_override, str(cfg.run_dir)


def set_run_dir(last_override, run_dir):
    run_dir_override = f'hydra.run.dir={run_dir}'
    if last_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(last_override + 1, run_dir_override)


class Split(str, enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class DataAssimilationSetting(orm.Table):
    dataset = orm.OneToManyField(conf.dataset.DynamicalSystemImpl, required=True, default=omegaconf.MISSING)
    splitter = orm.OneToManyField(conf.splitter.Splitter, default=omegaconf.MISSING)
    split: Split = orm.make_field(orm.ColumnRequired(sa.Enum(Split)), default=Split.TRAIN)

    reference_filter = orm.OneToManyField('Conf', required=False, enforce_element_type=False, column_name='ReferenceFilter')

    observes = orm.ManyToManyField(conf.observe.Observe, default_factory=list, enforce_element_type=False)
    obs_noise_std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)
    observe_every_n_time_steps: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)

    ensemble_initial_mean_is_true_state: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)

    @property
    def observation_time_steps(self):
        return range(
            getattr(self.splitter, f'start_{self.split}'),
            getattr(self.splitter, f'start_{self.split}') + getattr(self.splitter, f'len_{self.split}'),
            self.observe_every_n_time_steps
        )

    @property
    def cache_dir(self):
        return self.dataset.processed_filepath.parent

    @staticmethod
    def transform_reference_filter(session, conf_alt_id):
        if conf_alt_id is None:
            return None
        conf = session.query(Conf).filter_by(alt_id=conf_alt_id).first()
        assert conf is not None
        return conf

    def validate_observes(self, observes):
        if not any(isinstance(t, conf.observe.Observe) for t in observes):
            return
        observes = sorted(observes)
        if len(observes) == 0:
            raise ValueError('Please specify at least one Observe.')
        if len(observes) > 1 and any(isinstance(t, conf.observe.ObserveIdentity) for t in observes):
            raise ValueError('One or more ObserveIdentity were specified, but ObserveIdentity may only be used once and with no other Observe specified.')
        if (t1 := observes[0]).order != 0:
            raise ValueError(f'The first Observe {type(t1).__name__}(order={t1.order}) must have order=0.')
        for t1, t2 in itertools.pairwise(observes):
            if t2.order - t1.order != 1:
                raise ValueError(
                    f'Observe {type(t1).__name__}(order={t1.order}) preceding {type(t2).__name_}(order={t2.order}) '
                    'must have order value one less.'
                )
        for o in observes:
            if isinstance(o, conf.observe.ObserveMaskedPatches):
                if len(self.dataset.spatial_dims) == 1:
                    if o.patch_mask not in (conf.observe.PatchMask.CENTER_TOP, conf.observe.PatchMask.TOP):
                        raise ValueError(f"The states of the dataset {type(self.dataset).__name__} are represented as a vector, so only patch_mask CENTER_TOP and TOP are allowed, not {o.patch_mask}.")

    def __post_init__(self):
        if isinstance(self.reference_filter, Conf):
            if not self.reference_filter.save_ensemble_stats:
                raise ValueError('Please pass the conf alt_id of a run where save_ensemble_stats=true.')
            if self.reference_filter.setting.dataset != self.dataset:
                raise ValueError('The dataset of the reference filter must match the dataset currently being used.')
            if self.reference_filter.setting.observes != self.observes:
                raise ValueError('The observes of the reference filter must match the observes currently being used.')
            if self.reference_filter.setting.obs_noise_std != self.obs_noise_std:
                raise ValueError('The obs_noise_std of the reference filter must match the obs_noise_std currently being used.')
            if self.reference_filter.setting.observe_every_n_time_steps != self.observe_every_n_time_steps:
                raise ValueError('The observe_every_n_time_steps of the reference filter must match the observe_every_n_time_steps currently being used.')
        self.validate_observes(self.observes)
        if isinstance(self.splitter, conf.splitter.StartAndLen):
            splits = ('train', 'val', 'test')
            starts = sorted([(split, getattr(self.splitter, f'start_{split}')) for split in splits], key=lambda x: x[1])
            for split, start in starts:
                if start >= self.dataset.time_step_count + 1 and getattr(self.splitter, f'len_{split}') > 0:  # add one for initial condition
                    raise ValueError(f"The starting time step splitter.start_{split} must be less than or equal to dataset.time_step_count={self.dataset.time_step_count}.")
            for i, (split, start) in enumerate(starts):
                end = start + getattr(self.splitter, f'len_{split}')
                if i + 1 < bisect_left(starts, end, key=lambda x: x[1]):
                    overlapped_split, overlapped_start = starts[i + 1]
                    raise ValueError(
                        f"The split {split!r} overlaps with the split {overlapped_split!r}. "
                        f"The end of split {split!r} ({end}) must be less than or equal to the start of split {overlapped_split!r} ({overlapped_start}). "
                        f"With splitter.start_{split}={start}, set splitter.len_{split}={overlapped_start - start}"
                    )
            for i, (split, start) in enumerate(starts):
                end = start + getattr(self.splitter, f'len_{split}')
                if end > self.dataset.time_step_count + 1:
                    exceeded = end - (self.dataset.time_step_count + 1)
                    raise ValueError(
                        f"The end of split {split!r} exceeds the end of the dataset by {exceeded} time step(s). "
                        f"Maybe set splitter.len_{split}={getattr(self.splitter, f'len_{split}') - exceeded}?"
                    )


class Conf(orm.Table):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        {'dataset@setting.dataset': omegaconf.MISSING},
        {'splitter@setting.splitter': omegaconf.MISSING},
        dict(filter=omegaconf.MISSING),
        '_self_',
    ])
    root_dir: str = field(default=str(dafm.utils.DIR_ROOT.resolve()))
    out_dir: str = field(default=str((dafm.utils.DIR_ROOT/'..'/'..'/'out'/'revision-dafm').resolve()))
    run_subdir: str = field(default='runs')
    device: str = field(default='cuda')

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2376999025)
    setting = orm.OneToManyField(DataAssimilationSetting, default_factory=DataAssimilationSetting)
    filter = orm.OneToManyField(conf.filter.Filter, required=True, default=omegaconf.MISSING)
    save_ensemble_stats: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id


class OptunaTrialParamType(str, enum.Enum):
    FLOAT = 'float'
    INTEGER = 'integer'


class OptunaDirection(str, enum.Enum):
    MINIMIZE = 'minimize'
    MAXIMIZE = 'maximize'


class OptunaTrialParam(orm.Table):
    __table_args__ = (
        sa.UniqueConstraint('override', 'type', 'min', 'max', 'log', name='uq_optuna_trial_param'),
    )

    override: str = orm.make_field(orm.ColumnRequired(sa.String), default=omegaconf.MISSING)
    type: OptunaTrialParamType = orm.make_field(orm.ColumnRequired(sa.Enum(OptunaTrialParamType)), default=omegaconf.MISSING)
    min: Optional[float] = orm.make_field(sa.Column(sa.Double), default=None)
    max: Optional[float] = orm.make_field(sa.Column(sa.Double), default=None)
    log: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    def __post_init__(self):
        self.type = OptunaTrialParamType(self.type)
        if self.type in (OptunaTrialParamType.FLOAT, OptunaTrialParamType.INTEGER):
            if self.min is None or self.max is None:
                raise ValueError(f'Parameter {self.override} requires min and max values.')
            if self.max < self.min:
                raise ValueError(f'Parameter {self.override} has max={self.max} < min={self.min}.')


class OptunaStudy(orm.Table):
    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    root_dir: str = field(default=str(dafm.utils.DIR_ROOT.resolve()))
    out_dir: str = field(default=str((dafm.utils.DIR_ROOT/'..'/'..'/'out'/'revision-dafm').resolve()))
    run_subdir: str = field(default='optuna_runs')
    fixed_overrides: Tuple[str, ...] = field(default_factory=tuple)
    base_conf = orm.OneToManyField(Conf, required=True, enforce_element_type=False, column_name='BaseConf')
    params = orm.ManyToManyField(OptunaTrialParam, default_factory=list, enforce_element_type=False)

    direction: OptunaDirection = orm.make_field(orm.ColumnRequired(sa.Enum(OptunaDirection)), default=OptunaDirection.MINIMIZE)
    n_trials_total: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=50)
    metric: str = orm.make_field(orm.ColumnRequired(sa.String), default='rmse')

    @staticmethod
    def transform_base_conf(session, conf_alt_id):
        conf = session.query(Conf).filter_by(alt_id=conf_alt_id).first()
        if conf is None:
            raise ValueError(f'Unknown Conf alt_id: {conf_alt_id}')
        return conf

    def __post_init__(self):
        self.direction = OptunaDirection(self.direction)
        if self.n_trials_total < 0:
            raise ValueError(f'n_trials_total must be >= 0, not {self.n_trials_total}')
        if len(self.params) == 0:
            raise ValueError('At least one Optuna trial param must be specified.')

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id


sa.event.listens_for(Conf, 'before_insert')(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)

sa.event.listens_for(OptunaStudy, 'before_insert')(
    hydra_orm.utils.set_attr_to_func_value(OptunaStudy, OptunaStudy.alt_id.key, hydra_orm.utils.generate_random_string)
)


orm.store_config(Conf)
orm.store_config(conf.splitter.StartAndLen, group=DataAssimilationSetting.splitter.key, name=f'_{conf.splitter.StartAndLen.__name__}')
for cfg in (
    conf.dataset.DynamicalSystemImpl,
):
    orm.store_config(cfg, group=DataAssimilationSetting.dataset.key)
for cfg in (
    conf.dataset.Lorenz63,
    conf.dataset.Lorenz96,
    conf.dataset.KuramotoSivashinsky,
    conf.dataset.NavierStokes2DPeriodicBoundary,
    conf.dataset.NavierStokes2DBackwardFacingStepGLED,
    conf.dataset.UnitTestDataset,
    conf.dataset.Rotation2D,
):
    orm.store_config(cfg, group=f'{DataAssimilationSetting.dataset.key}/{conf.dataset.DynamicalSystemImpl.system.key}')
for cfg in (
    conf.filter.AddNoiseToObservationFilter,
    conf.filter.BootstrapParticleFilter,
    conf.filter.KalmanFilter,
    conf.filter.EnsembleKalmanFilterPerturbedObservations,
    conf.filter.EnsembleKalmanFilterPerturbedObservationsIterative,
    conf.filter.EnsembleRandomizedSquareRootFilter,
    conf.filter.LocalEnsembleTransformKalmanFilter,
    conf.filter.EnsembleScoreFilter,
    conf.filter.EnsembleFlowFilter,
):
    orm.store_config(cfg, group=Conf.filter.key, name=f'_{cfg.__name__}')
for cfg in (
    conf.prob_path.Bao2024EnsembleScoreMatching,
    conf.prob_path.ConditionalOptimalTransport,
    conf.prob_path.FilteringToPredictive,
    conf.prob_path.VarianceExploding,
):
    orm.store_config(cfg, group=f'{Conf.filter.key}/{conf.filter.EnsembleScoreFilter.prob_path.key}')
for cfg in (
    conf.flow_matching_guidance.No,
    conf.flow_matching_guidance.MonteCarlo,
    conf.flow_matching_guidance.Local,
):
    orm.store_config(cfg, group=f'{Conf.filter.key}/{conf.filter.EnsembleFlowFilter.guidance.key}', name=f'_{cfg.__name__}')
orm.store_config(conf.flow_matching_guidance.Constant, group=f'{Conf.filter.key}/{conf.filter.EnsembleFlowFilter.guidance.key}/{conf.flow_matching_guidance.Local.schedule.key}')
orm.store_config(OptunaTrialParam)
orm.store_config(OptunaStudy)
