import argparse
import dataclasses
import enum
from pathlib import Path
from typing import Any

import hydra
from hydra._internal.callbacks import Callbacks
from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_automatic_config_search_path
from hydra.core.global_hydra import GlobalHydra
from hydra.core.utils import JobStatus, run_job
from hydra.types import HydraContext
from omegaconf import OmegaConf, open_dict

import conf.conf
import dafm.main
import dafm.utils


IDENTITY_KEYS = {'id', 'alt_id'}
HYDRA_CONFIG_DIR = '../../conf'


def nullable_string(val):
    if not val or val == 'None':
        return None
    return val


def parse_bool(val):
    normalized = str(val).strip().lower()
    if normalized in {'true', '1', 'yes', 'y'}:
        return True
    if normalized in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Expected boolean value for --save-ensemble-stats, got {val!r}.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Fetch a Conf by alt_id, patch it, insert it, and run it with Hydra internals.',
    )
    parser.add_argument('--base-alt-id', required=True, help='Existing Conf.alt_id to use as the base run config.')
    parser.add_argument('--rng-seed', type=int, required=True, help='rng_seed override')
    parser.add_argument('--reference-filter', type=nullable_string)
    parser.add_argument('--target-setting-id', type=int)
    parser.add_argument('--save-ensemble-stats', type=parse_bool, default=False)
    return parser.parse_args(argv)


def fetch_conf_container(base_alt_id: str) -> dict[str, Any]:
    with conf.conf.Session() as db:
        cfg = db.query(conf.conf.Conf).filter_by(alt_id=base_alt_id).first()
        if cfg is None:
            raise ValueError(f'Unknown Conf alt_id: {base_alt_id}')
        return conf_to_clean_container(cfg)


def fetch_setting_container(target_setting_id: int) -> dict[str, Any]:
    with conf.conf.Session() as db:
        setting = db.query(conf.conf.DataAssimilationSetting).filter_by(id=target_setting_id).first()
        if setting is None:
            raise ValueError(f'Unknown DataAssimilationSetting id: {target_setting_id}')
        container = conf_to_clean_container(setting)
        if not isinstance(container, dict):
            raise TypeError('Expected DataAssimilationSetting container to be a dictionary.')
        return container


def _is_identity_key(key: str) -> bool:
    return key in IDENTITY_KEYS or key.endswith('_id')


def _to_container(value: Any, *, root: bool = False) -> Any:
    if isinstance(value, enum.Enum):
        if isinstance(value.value, str):
            return value.value
        return value.name
    if dataclasses.is_dataclass(value):
        if not root and isinstance(value, conf.conf.Conf):
            return value.alt_id
        out = {}
        for field in dataclasses.fields(value):
            key = field.name
            if not field.init:
                continue
            if _is_identity_key(key):
                continue
            if not hasattr(value, key):
                continue
            out[key] = _to_container(getattr(value, key), root=False)
        return out
    if isinstance(value, tuple):
        return tuple(_to_container(v, root=False) for v in value)
    if isinstance(value, list):
        return [_to_container(v, root=False) for v in value]
    if isinstance(value, dict):
        return {k: _to_container(v, root=False) for k, v in value.items() if not _is_identity_key(str(k))}
    if isinstance(value, Path):
        return str(value)
    return value


def conf_to_clean_container(cfg: conf.conf.Conf) -> dict[str, Any]:
    container = _to_container(cfg, root=True)
    if not isinstance(container, dict):
        raise TypeError('Expected Conf container to be a dictionary.')
    return container


def patch_conf_container(cfg: dict[str, Any], args) -> dict[str, Any]:
    cfg['rng_seed'] = args.rng_seed
    cfg['save_ensemble_stats'] = args.save_ensemble_stats
    if args.target_setting_id is not None:
        cfg['setting'] = fetch_setting_container(args.target_setting_id)
        target_rng_seed = cfg['setting']['dataset']['rng_seed']
        if target_rng_seed != args.rng_seed:
            raise ValueError(
                'target setting dataset rng_seed must match --rng-seed: '
                f'{target_rng_seed} != {args.rng_seed}'
            )
    cfg['setting']['dataset']['rng_seed'] = args.rng_seed
    cfg['setting']['reference_filter'] = args.reference_filter
    return cfg


def insert_conf(cfg_container: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    with conf.conf.Session() as db:
        cfg = conf.conf.orm.instantiate_and_insert_config(db, cfg_container)
        db.commit()
        return cfg.alt_id, cfg.run_dir, conf_to_clean_container(cfg)


def run_conf_with_hydra(
    task_cfg_container: dict[str, Any],
    run_dir: Path,
    task_overrides: list[str] | None = None,
    task_function=None,
) -> None:
    if task_function is None:
        task_function = dafm.main.run

    task_cfg = OmegaConf.create(task_cfg_container)

    with hydra.initialize_config_dir(version_base=None, config_dir=str((dafm.utils.DIR_ROOT / 'conf').resolve())):
        hydra_only_cfg = hydra.compose(config_name=None, overrides=[], return_hydra_config=True)

    cfg_with_hydra = hydra_only_cfg
    with open_dict(cfg_with_hydra):
        for key, value in task_cfg.items():
            cfg_with_hydra[key] = value

    cfg_with_hydra.hydra.job.name = 'main'
    cfg_with_hydra.hydra.run.dir = str(run_dir)
    cfg_with_hydra.hydra.overrides.task = task_overrides or []

    search_path = create_automatic_config_search_path(
        calling_file=__file__,
        calling_module=None,
        config_path=HYDRA_CONFIG_DIR,
    )
    hydra_app = Hydra.create_main_hydra2(task_name='rerun_from_alt_id', config_search_path=search_path)
    callbacks = Callbacks(cfg_with_hydra)
    callbacks.on_run_start(config=cfg_with_hydra, config_name='conf')
    try:
        job_return = run_job(
            task_function=task_function,
            config=cfg_with_hydra,
            job_dir_key='hydra.run.dir',
            job_subdir_key=None,
            hydra_context=HydraContext(config_loader=hydra_app.config_loader, callbacks=callbacks),
            configure_logging=True,
        )
        callbacks.on_run_end(config=cfg_with_hydra, config_name='conf', job_return=job_return)
        if job_return.status != JobStatus.COMPLETED:
            if isinstance(job_return.return_value, Exception):
                raise job_return.return_value
            raise RuntimeError(f'Hydra job failed with status={job_return.status}.')
    finally:
        GlobalHydra.instance().clear()


def main(argv=None):
    args = parse_args(argv)
    base_container = fetch_conf_container(args.base_alt_id)
    patched_container = patch_conf_container(base_container, args)
    new_alt_id, run_dir, inserted_container = insert_conf(patched_container)
    run_conf_with_hydra(
        inserted_container,
        run_dir=run_dir,
        task_overrides=[
            f'base_alt_id={args.base_alt_id}',
            f'rng_seed={patched_container["rng_seed"]}',
            f'save_ensemble_stats={str(args.save_ensemble_stats).lower()}',
            *([f'target_setting_id={args.target_setting_id}'] if args.target_setting_id is not None else []),
            *([f'reference_filter={args.reference_filter}'] if args.reference_filter is not None else []),
        ],
    )
    print(f'Base alt_id: {args.base_alt_id}')
    print(f'New alt_id: {new_alt_id}')
    print(f'Output directory: {run_dir}')


if __name__ == '__main__':
    main()
