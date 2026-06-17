import pytest
from omegaconf import OmegaConf

from fixtures import init_hydra_cfg, engine

from conf import conf
from dafm.main_optuna import (
    TrialParamSpec,
    build_optuna_study_overrides,
    compose_and_insert_base_conf,
    derive_metric_from_fixed_overrides,
    get_remaining_trials,
    parse_optuna_trial_params,
    split_cli_args,
)


BASE_CONF_OVERRIDES = [
    '+experiment=Lorenz63Spantini2022',
    'filter=EnsembleKalmanFilterPerturbedObservations',
]


def test_split_cli_args():
    fixed, params_arg, study_overrides = split_cli_args([
        '+experiment=Lorenz63Spantini2022',
        'filter=EnsembleKalmanFilterPerturbedObservations',
        'optuna_study.direction=maximize',
        'optuna_study.n_trials_total=100',
        'optuna_trial_params=[{override:filter.inflation_scale,type:float,min:1,max:10,log:true}]',
    ])
    assert fixed == [
        '+experiment=Lorenz63Spantini2022',
        'filter=EnsembleKalmanFilterPerturbedObservations',
    ]
    assert study_overrides == ['optuna_study.direction=maximize', 'optuna_study.n_trials_total=100']
    assert params_arg.startswith('optuna_trial_params=')

    with pytest.raises(ValueError):
        split_cli_args([
            '+experiment=Lorenz63Spantini2022',
            'filter=EnsembleKalmanFilterPerturbedObservations',
        ])

    with pytest.raises(ValueError):
        split_cli_args([
            'optuna_trial_params=[{override:filter.inflation_scale,type:float,min:1,max:10,log:true}]',
            'optuna_trial_params=[{override:filter.gaspari_cohn_localization_radius,type:integer,min:1,max:8}]',
        ])

    with pytest.raises(ValueError):
        split_cli_args([
            '+experiment=Lorenz63Spantini2022',
            'optuna_study.metric=rmse',
            'optuna_trial_params=[{override:filter.inflation_scale,type:float,min:1,max:10,log:true}]',
        ])


def test_parse_optuna_trial_params_sorts_and_normalizes():
    specs = parse_optuna_trial_params(
        'optuna_trial_params=['
        '{override:filter.inflation_scale,type:float,min:1,max:10,log:true},'
        '{override:filter.gaspari_cohn_localization_radius,type:integer,min:1,max:8}'
        ']'
    )
    assert [s.override for s in specs] == [
        'filter.gaspari_cohn_localization_radius',
        'filter.inflation_scale',
    ]
    assert specs[0].log is False
    assert specs[1].log is True


def test_compose_and_insert_base_conf_is_stable(engine, monkeypatch):
    session = conf.sa.orm.sessionmaker(engine)
    monkeypatch.setattr(conf, 'Session', session)

    fixed_overrides = [
        *BASE_CONF_OVERRIDES,
    ]
    trial_params = [
        TrialParamSpec(
            override='filter.gaspari_cohn_localization_radius',
            type=conf.OptunaTrialParamType.INTEGER,
            min=1,
            max=8,
            log=False,
        ),
        TrialParamSpec(
            override='filter.inflation_scale',
            type=conf.OptunaTrialParamType.FLOAT,
            min=1,
            max=10,
            log=True,
        ),
    ]

    alt_id_1 = compose_and_insert_base_conf(fixed_overrides, trial_params)
    alt_id_2 = compose_and_insert_base_conf(fixed_overrides, trial_params)

    assert alt_id_1 == alt_id_2


def test_hydra_orm_optuna_study_dedup_by_base_conf_plus_param_set(engine, monkeypatch):
    session = conf.sa.orm.sessionmaker(engine)
    monkeypatch.setattr(conf, 'Session', session)

    cfg = init_hydra_cfg('conf', [
        *BASE_CONF_OVERRIDES,
        'filter.gaspari_cohn_localization_radius=1',
    ])

    with conf.sa.orm.Session(engine) as db:
        base_conf = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        base_conf_alt_id = base_conf.alt_id

    trial_params = [
        TrialParamSpec(
            override='filter.gaspari_cohn_localization_radius',
            type=conf.OptunaTrialParamType.INTEGER,
            min=1,
            max=8,
            log=False,
        ),
        TrialParamSpec(
            override='filter.inflation_scale',
            type=conf.OptunaTrialParamType.FLOAT,
            min=1,
            max=10,
            log=True,
        ),
    ]

    study_cfg = {
        '_target_': 'conf.conf.OptunaStudy',
        'base_conf': base_conf_alt_id,
        'params': [
            {
                '_target_': 'conf.conf.OptunaTrialParam',
                'override': spec.override,
                'type': spec.type.value,
                'min': spec.min,
                'max': spec.max,
                'log': spec.log,
            }
            for spec in trial_params
        ],
        'direction': 'minimize',
        'n_trials_total': 60,
        'metric': 'rmse',
    }

    with conf.sa.orm.Session(engine) as db:
        study_1 = conf.orm.instantiate_and_insert_config(db, study_cfg)
        db.commit()
        study_alt_id_1 = study_1.alt_id

    with conf.sa.orm.Session(engine) as db:
        study_cfg_reordered = {
            **study_cfg,
            'params': list(reversed(study_cfg['params'])),
        }
        study_2 = conf.orm.instantiate_and_insert_config(db, study_cfg_reordered)
        db.commit()
        study_alt_id_2 = study_2.alt_id

    assert study_alt_id_1 == study_alt_id_2

    changed_trial_params = [
        TrialParamSpec(
            override='filter.gaspari_cohn_localization_radius',
            type=conf.OptunaTrialParamType.INTEGER,
            min=1,
            max=9,
            log=False,
        ),
        TrialParamSpec(
            override='filter.inflation_scale',
            type=conf.OptunaTrialParamType.FLOAT,
            min=1,
            max=10,
            log=True,
        ),
    ]
    changed_study_cfg = {
        **study_cfg,
        'params': [
            {
                '_target_': 'conf.conf.OptunaTrialParam',
                'override': spec.override,
                'type': spec.type.value,
                'min': spec.min,
                'max': spec.max,
                'log': spec.log,
            }
            for spec in changed_trial_params
        ],
    }
    with conf.sa.orm.Session(engine) as db:
        study_3 = conf.orm.instantiate_and_insert_config(db, changed_study_cfg)
        db.commit()
        study_alt_id_3 = study_3.alt_id

    assert study_alt_id_3 != study_alt_id_1


def test_build_optuna_study_overrides_contains_optuna_run_fields():
    trial_params = [
        TrialParamSpec(
            override='filter.inflation_scale',
            type=conf.OptunaTrialParamType.FLOAT,
            min=1,
            max=10,
            log=True,
        ),
    ]

    overrides = build_optuna_study_overrides(
        base_conf_alt_id='abcd1234',
        trial_params=trial_params,
        fixed_overrides=[*BASE_CONF_OVERRIDES],
        metric='rmse',
        optuna_study_cli_overrides=['optuna_study.direction=maximize', 'optuna_study.n_trials_total=123'],
    )

    assert 'optuna_study.base_conf=abcd1234' in overrides
    assert any(o.startswith('optuna_study.params=[') for o in overrides)
    assert any(o.startswith('optuna_study.fixed_overrides=[') for o in overrides)
    assert 'optuna_study.direction=maximize' in overrides
    assert 'optuna_study.n_trials_total=123' in overrides


def test_build_optuna_study_overrides_can_compose_optuna_study_cfg():
    trial_params = [
        TrialParamSpec(
            override='filter.inflation_scale',
            type=conf.OptunaTrialParamType.FLOAT,
            min=1,
            max=10,
            log=True,
        ),
    ]
    overrides = build_optuna_study_overrides(
        base_conf_alt_id='abcd1234',
        trial_params=trial_params,
        fixed_overrides=[*BASE_CONF_OVERRIDES],
        metric='rmse',
        optuna_study_cli_overrides=['optuna_study.direction=MINIMIZE', 'optuna_study.n_trials_total=5'],
    )
    cfg = init_hydra_cfg('optuna_study', overrides)
    assert cfg.optuna_study.base_conf == 'abcd1234'


def test_derive_metric_from_fixed_overrides():
    metric = derive_metric_from_fixed_overrides([
        *BASE_CONF_OVERRIDES,
    ])
    assert metric == 'rmse'

    metric_ref = derive_metric_from_fixed_overrides([
        *BASE_CONF_OVERRIDES,
        'setting.reference_filter=abcd1234',
    ])
    assert metric_ref == 'rmse_from_reference_filter'


def test_get_remaining_trials_uses_total_target_semantics():
    assert get_remaining_trials(60, 0) == 60
    assert get_remaining_trials(60, 12) == 48
    assert get_remaining_trials(60, 60) == 0
    assert get_remaining_trials(60, 75) == 0


def test_optuna_study_run_dir_uses_optuna_runs(engine):
    cfg = init_hydra_cfg('conf', [
        *BASE_CONF_OVERRIDES,
        'filter.gaspari_cohn_localization_radius=1',
    ])
    with conf.sa.orm.Session(engine) as db:
        base_conf = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        base_conf_alt_id = base_conf.alt_id

    study_cfg = {
        '_target_': 'conf.conf.OptunaStudy',
        'base_conf': base_conf_alt_id,
        'params': [{
            '_target_': 'conf.conf.OptunaTrialParam',
            'override': 'filter.inflation_scale',
            'type': 'float',
            'min': 1,
            'max': 10,
            'log': True,
        }],
    }
    with conf.sa.orm.Session(engine, expire_on_commit=False) as db:
        study = conf.orm.instantiate_and_insert_config(db, study_cfg)
        db.commit()

    assert study.run_subdir == 'optuna_runs'
    assert study.run_dir.parts[-2] == 'optuna_runs'
