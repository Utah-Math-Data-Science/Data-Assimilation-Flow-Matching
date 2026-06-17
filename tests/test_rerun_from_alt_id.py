from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from fixtures import engine, init_hydra_cfg

import conf.conf
from dafm.rerun_from_alt_id import (
    conf_to_clean_container,
    fetch_conf_container,
    fetch_setting_container,
    insert_conf,
    patch_conf_container,
    run_conf_with_hydra,
)


BASE_CONF_OVERRIDES = [
    '+experiment=Lorenz63Spantini2022',
    'filter=EnsembleKalmanFilterPerturbedObservations',
]


def _make_base_conf(engine, extra_overrides=None):
    overrides = [*BASE_CONF_OVERRIDES, 'filter.gaspari_cohn_localization_radius=1']
    if extra_overrides is not None:
        overrides.extend(extra_overrides)
    cfg = init_hydra_cfg('conf', overrides)
    with conf.conf.sa.orm.Session(engine) as db:
        row = conf.conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        return row.alt_id


def test_fetch_conf_container_unknown_alt_id_raises(engine, monkeypatch):
    session = conf.conf.sa.orm.sessionmaker(engine)
    monkeypatch.setattr(conf.conf, 'Session', session)

    try:
        fetch_conf_container('nope0000')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'Unknown Conf alt_id' in str(e)


def test_conf_to_clean_container_strips_identity_fields(engine):
    base_alt_id = _make_base_conf(engine)
    with conf.conf.sa.orm.Session(engine) as db:
        cfg = db.query(conf.conf.Conf).filter_by(alt_id=base_alt_id).first()
        container = conf_to_clean_container(cfg)

    assert 'id' not in container
    assert 'alt_id' not in container
    assert 'filter_id' not in container
    assert 'setting_id' not in container
    assert 'setting' in container
    assert 'id' not in container['setting']
    assert 'dataset_id' not in container['setting']


def test_patch_and_insert_conf_changes_seed(engine, monkeypatch):
    session = conf.conf.sa.orm.sessionmaker(engine)
    monkeypatch.setattr(conf.conf, 'Session', session)

    base_alt_id = _make_base_conf(engine)
    base_container = fetch_conf_container(base_alt_id)
    original_seed = base_container['rng_seed']
    target_seed = 462133975 if original_seed != 462133975 else 979497033
    patched = patch_conf_container(
        base_container,
        SimpleNamespace(rng_seed=target_seed, reference_filter=None, target_setting_id=None, save_ensemble_stats=False),
    )
    new_alt_id, _, inserted_container = insert_conf(patched)

    assert new_alt_id != base_alt_id
    assert inserted_container['rng_seed'] == target_seed
    assert inserted_container['rng_seed'] != original_seed


def test_fetch_setting_container_unknown_id_raises(engine, monkeypatch):
    session = conf.conf.sa.orm.sessionmaker(engine)
    monkeypatch.setattr(conf.conf, 'Session', session)

    try:
        fetch_setting_container(999999)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'Unknown DataAssimilationSetting id' in str(e)


def test_patch_conf_container_replaces_setting_by_id(engine, monkeypatch):
    session = conf.conf.sa.orm.sessionmaker(engine)
    monkeypatch.setattr(conf.conf, 'Session', session)

    base_alt_id = _make_base_conf(engine)
    target_alt_id = _make_base_conf(engine, ['rng_seed=462133975'])

    base_container = fetch_conf_container(base_alt_id)
    with conf.conf.sa.orm.Session(engine) as db:
        target_setting_id = db.query(conf.conf.Conf).filter_by(alt_id=target_alt_id).first().setting.id
    target_container = fetch_conf_container(target_alt_id)

    patched = patch_conf_container(
        base_container,
        SimpleNamespace(
            rng_seed=462133975,
            reference_filter=None,
            target_setting_id=target_setting_id,
            save_ensemble_stats=False,
        ),
    )

    assert patched['setting']['dataset']['rng_seed'] == 462133975
    assert patched['setting']['obs_noise_std'] == target_container['setting']['obs_noise_std']


def test_patch_conf_container_target_setting_rng_mismatch_raises(engine, monkeypatch):
    session = conf.conf.sa.orm.sessionmaker(engine)
    monkeypatch.setattr(conf.conf, 'Session', session)

    base_alt_id = _make_base_conf(engine)
    target_alt_id = _make_base_conf(engine, ['rng_seed=462133975'])

    base_container = fetch_conf_container(base_alt_id)
    with conf.conf.sa.orm.Session(engine) as db:
        target_setting_id = db.query(conf.conf.Conf).filter_by(alt_id=target_alt_id).first().setting.id

    try:
        patch_conf_container(
            base_container,
            SimpleNamespace(
                rng_seed=979497033,
                reference_filter=None,
                target_setting_id=target_setting_id,
                save_ensemble_stats=False,
            ),
        )
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'target setting dataset rng_seed must match --rng-seed' in str(e)


def test_run_conf_with_hydra_writes_hydra_artifacts(engine, monkeypatch, tmp_path):
    session = conf.conf.sa.orm.sessionmaker(engine)
    monkeypatch.setattr(conf.conf, 'Session', session)

    base_alt_id = _make_base_conf(engine)
    task_cfg = fetch_conf_container(base_alt_id)
    run_dir = tmp_path / 'rerun'

    def dummy_task(cfg):
        Path(run_dir / 'task_called.txt').write_text(str(cfg.rng_seed), encoding='utf-8')

    run_conf_with_hydra(
        task_cfg,
        run_dir=run_dir,
        task_overrides=['base_alt_id=test1234', 'rng_seed=462133975'],
        task_function=dummy_task,
    )

    assert (run_dir / 'task_called.txt').exists()
    assert (run_dir / '.hydra' / 'config.yaml').exists()
    assert (run_dir / '.hydra' / 'hydra.yaml').exists()
    overrides_file = run_dir / '.hydra' / 'overrides.yaml'
    assert overrides_file.exists()
    assert 'base_alt_id=test1234' in overrides_file.read_text(encoding='utf-8')
