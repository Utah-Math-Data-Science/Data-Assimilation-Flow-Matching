from functools import partial
from dataclasses import dataclass
from collections import Counter
from queue import Queue
from pathlib import Path
import pprint
import os
import sys
import subprocess
import hydra
from hydra.core.override_parser.overrides_parser import OverridesParser
import optuna
import optuna_dashboard
import polars as pl
from omegaconf import OmegaConf
import sqlalchemy as sa

import conf.conf
import dafm.utils


log = dafm.utils.getLoggerByFilename(__file__)


GPUS = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')
GPU_QUEUE = Queue()
MAX_TRIALS_PER_GPU = 1
for gpu in GPUS:
    for _ in range(MAX_TRIALS_PER_GPU):
        GPU_QUEUE.put(gpu)

HYDRA_OPTUNA_STUDY_INIT = {**dafm.utils.HYDRA_INIT, 'config_name': 'optuna_study'}
OPTUNA_STORAGE = 'sqlite:///optuna.sqlite'


@dataclass(frozen=True)
class TrialParamSpec:
    override: str
    type: conf.conf.OptunaTrialParamType
    min: float
    max: float
    log: bool

    def validate(self):
        if not self.override:
            raise ValueError('Each optuna trial parameter requires a non-empty override key.')
        if self.max < self.min:
            raise ValueError(f'Parameter {self.override} has max={self.max} < min={self.min}.')

    def dummy_value(self):
        if self.type == conf.conf.OptunaTrialParamType.INTEGER:
            return int(self.min)
        return float(self.min)

    def suggest_value(self, trial):
        if self.type == conf.conf.OptunaTrialParamType.INTEGER:
            return trial.suggest_int(self.override, int(self.min), int(self.max), log=self.log)
        return trial.suggest_float(self.override, float(self.min), float(self.max), log=self.log)

    def to_override(self, value):
        return f'{self.override}={value}'


def split_cli_args(argv):
    allowed_optuna_study_overrides = (
        'optuna_study.direction=',
        'optuna_study.n_trials_total=',
    )

    fixed_overrides = []
    optuna_study_overrides = []
    optuna_trial_params_args = []
    for arg in argv:
        if arg.startswith('optuna_trial_params='):
            optuna_trial_params_args.append(arg)
        elif arg.startswith('optuna_study.'):
            if not arg.startswith(allowed_optuna_study_overrides):
                raise ValueError(
                    'Only optuna_study.direction and optuna_study.n_trials_total may be set from the command line.'
                )
            optuna_study_overrides.append(arg)
        else:
            fixed_overrides.append(arg)

    if len(optuna_trial_params_args) != 1:
        raise ValueError(
            f'Expected exactly one optuna_trial_params=... argument, but found {len(optuna_trial_params_args)}. '
            'Pass optuna_trial_params as a single Hydra list override.'
        )

    return fixed_overrides, optuna_trial_params_args[0], optuna_study_overrides


def parse_optuna_trial_params(optuna_trial_params_override):
    parsed_override = OverridesParser.create().parse_overrides([optuna_trial_params_override])[0]
    if parsed_override.key_or_group != 'optuna_trial_params':
        raise ValueError('Expected an optuna_trial_params override.')
    params = parsed_override.value()
    if not isinstance(params, list) or len(params) == 0:
        raise ValueError('optuna_trial_params must be a non-empty list.')

    specs = []
    for param in params:
        if not isinstance(param, dict):
            raise ValueError(f'Each optuna_trial_params item must be an object, got {param!r}.')
        param_type = conf.conf.OptunaTrialParamType(param['type'])
        min_value = param.get('min')
        max_value = param.get('max')
        if min_value is None or max_value is None:
            raise ValueError(f'Parameter {param.get("override")} requires both min and max.')
        spec = TrialParamSpec(
            override=str(param['override']),
            type=param_type,
            min=float(min_value),
            max=float(max_value),
            log=bool(param.get('log', False)),
        )
        spec.validate()
        specs.append(spec)

    specs = sorted(specs, key=lambda s: (s.override, s.type.value, s.min, s.max, s.log))
    duplicate_counts = Counter(spec.override for spec in specs)
    duplicates = sorted([name for name, count in duplicate_counts.items() if count > 1])
    if duplicates:
        raise ValueError(f'Duplicate override keys in optuna_trial_params are not supported: {duplicates}')
    return specs


def compose_and_insert_base_conf(fixed_overrides, trial_params):
    dummy_overrides = [spec.to_override(spec.dummy_value()) for spec in trial_params]
    with hydra.initialize(
        version_base=dafm.utils.HYDRA_INIT['version_base'],
        config_path=dafm.utils.HYDRA_INIT['config_path'],
    ):
        cfg = hydra.compose(
            config_name=dafm.utils.HYDRA_INIT['config_name'],
            overrides=[*fixed_overrides, *dummy_overrides],
        )

    with conf.conf.Session() as db:
        base_conf = conf.conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        return base_conf.alt_id


def _format_trial_param_value(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, str):
        return v
    return repr(v)


def derive_metric_from_fixed_overrides(fixed_overrides):
    if any(override.startswith('setting.reference_filter=') for override in fixed_overrides):
        return 'rmse_from_reference_filter'
    return 'rmse'


def build_optuna_study_overrides(base_conf_alt_id, trial_params, fixed_overrides, metric, optuna_study_cli_overrides):
    trial_params = sorted(trial_params, key=lambda s: (s.override, s.type.value, s.min, s.max, s.log))
    params_override = 'optuna_study.params=[' + ','.join(
        '{' + ','.join([
            '_target_:conf.conf.OptunaTrialParam',
            f'override:{spec.override}',
            f'type:{spec.type.value}',
            f'min:{_format_trial_param_value(spec.min)}',
            f'max:{_format_trial_param_value(spec.max)}',
            f'log:{_format_trial_param_value(spec.log)}',
        ]) + '}'
        for spec in trial_params
    ) + ']'
    fixed_overrides_override = 'optuna_study.fixed_overrides=[' + ','.join(repr(v) for v in fixed_overrides) + ']'
    return [
        f'optuna_study.base_conf={base_conf_alt_id}',
        params_override,
        fixed_overrides_override,
        f'optuna_study.metric="{metric}"',
        *optuna_study_cli_overrides,
    ]


def get_remaining_trials(n_trials_total, completed_trials):
    return max(0, n_trials_total - completed_trials)


def get_optuna_study_run_dir(study_overrides):
    with hydra.initialize(
        version_base=HYDRA_OPTUNA_STUDY_INIT['version_base'],
        config_path=HYDRA_OPTUNA_STUDY_INIT['config_path'],
    ):
        cfg = hydra.compose(
            config_name=HYDRA_OPTUNA_STUDY_INIT['config_name'],
            overrides=study_overrides,
        )

    with conf.conf.Session() as db:
        optuna_study_cfg = OmegaConf.to_container(cfg.optuna_study, resolve=True)
        optuna_study_cfg.pop('fixed_overrides', None)
        optuna_study: conf.conf.OptunaStudy = conf.conf.orm.instantiate_and_insert_config(
            db,
            optuna_study_cfg,
        )
        db.commit()
        optuna_study.run_dir.mkdir(exist_ok=True)
        return str(optuna_study.run_dir)


@hydra.main(**HYDRA_OPTUNA_STUDY_INIT)
def main(cfg):
    fixed_overrides = tuple(cfg.optuna_study.fixed_overrides)
    with conf.conf.Session() as db:
        optuna_study_cfg = OmegaConf.to_container(cfg.optuna_study, resolve=True)
        optuna_study_cfg.pop('fixed_overrides', None)
        optuna_study: conf.conf.OptunaStudy = conf.conf.orm.instantiate_and_insert_config(
            db,
            optuna_study_cfg,
        )
        db.commit()
        log.info('Command: python %s', ' '.join(sys.argv[:-1]))
        log.info(pprint.pformat(optuna_study))
        log.info('Output directory: %s', optuna_study.run_dir)

        trial_params = [
            TrialParamSpec(
                override=param.override,
                type=param.type,
                min=param.min,
                max=param.max,
                log=param.log,
            )
            for param in optuna_study.params
        ]

        study = optuna.create_study(
            storage=OPTUNA_STORAGE,
            study_name=optuna_study.alt_id,
            load_if_exists=True,
            direction=optuna_study.direction.value,
            pruner=optuna.pruners.ThresholdPruner(lower=-1),  # set lower to impossible value to prune only on NaN
        )
        study.optimize(
            partial(objective, fixed_overrides, trial_params, optuna_study.metric, study),
            n_trials=get_remaining_trials(optuna_study.n_trials_total, len(study.trials)),
            n_jobs=len(GPUS) * MAX_TRIALS_PER_GPU,
        )


def set_study_note(study, loss, alt_id):
    with conf.conf.Session() as db:
        cfg = db.execute(sa.select(conf.conf.Conf).filter_by(alt_id=alt_id)).first()[0]
        optuna_dashboard.save_note(
            study,
            f"""
            ### Loss
            ```
            {loss}
            ```
            ### Conf
            ```
            {pprint.pformat(cfg)}
            ```
            """,
        )


def objective(fixed_overrides, trial_params, metric, study, trial):
    gpu = GPU_QUEUE.get()
    cmd = [
        'python',
        'src/dafm/main.py',
        *fixed_overrides,
        *(spec.to_override(spec.suggest_value(trial)) for spec in trial_params),
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu

    try:
        process = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        run_dir = Path(process.stdout.split('Output directory: ')[-1].splitlines()[0])
        alt_id = run_dir.name

        trial.set_user_attr('alt_id', alt_id)
        set_study_note(study, metric, alt_id)
        loss = (
            pl.scan_csv(run_dir/'metrics.csv')
            .tail(50)
            .select(pl.mean(metric))
            .collect()
            .item()
        )
        return loss
    except subprocess.CalledProcessError:
        raise optuna.exceptions.TrialPruned()
    finally:
        GPU_QUEUE.put(gpu)


if __name__ == '__main__':
    conf.conf.orm.create_all(conf.conf.engine)
    fixed_overrides, optuna_trial_params_override, optuna_study_cli_overrides = split_cli_args(sys.argv[1:])
    trial_params = parse_optuna_trial_params(optuna_trial_params_override)
    metric = derive_metric_from_fixed_overrides(fixed_overrides)

    base_conf_alt_id = compose_and_insert_base_conf(fixed_overrides, trial_params)
    study_overrides = build_optuna_study_overrides(
        base_conf_alt_id,
        trial_params,
        fixed_overrides,
        metric,
        optuna_study_cli_overrides,
    )
    sys.argv = [sys.argv[0], *study_overrides]
    run_dir = get_optuna_study_run_dir(study_overrides)
    sys.argv.append(f'hydra.run.dir={run_dir}')
    main()
