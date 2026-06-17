import copy
import dataclasses
import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import conf.filter
from dafm.benchmark_filter import (
    BENCHMARK_ITERS,
    BENCHMARK_SYNCHRONIZE_CUDA,
    BENCHMARK_TIMINGS_FILENAME,
    BENCHMARK_WARMUP_ITERS,
    _apply_benchmark_runtime_overrides,
    _write_benchmark_timings_csv,
    parse_args,
)
from dafm import filters_classical, filters_classical_iterative


def test_hardcoded_benchmark_constants_are_valid():
    assert BENCHMARK_WARMUP_ITERS >= 0
    assert BENCHMARK_ITERS > 0
    assert isinstance(BENCHMARK_SYNCHRONIZE_CUDA, bool)
    assert BENCHMARK_TIMINGS_FILENAME.endswith('.csv')


def test_parse_args_requires_alt_id():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_accepts_alt_id():
    args = parse_args(['--alt-id', 'abcd1234'])
    assert args.alt_id == 'abcd1234'


@dataclasses.dataclass
class _DummyFilter:
    wtol: float | None = 1e-5
    use_torch_rng_for_perturbed_observations: bool = False


def test_runtime_overrides_force_ienkf_po_fixed_work():
    cfg = SimpleNamespace(filter=conf.filter.EnsembleKalmanFilterPerturbedObservationsIterative(
        ensemble_size=3,
        ensemble_initial_std=1.0,
        wtol=1e-5,
    ))
    da_filter = SimpleNamespace(cfg=_DummyFilter(wtol=1e-5))

    _apply_benchmark_runtime_overrides(cfg, da_filter)

    assert cfg.filter.wtol is None
    assert da_filter.cfg.wtol is None


def test_runtime_overrides_force_enkf_po_torch_rng():
    cfg = SimpleNamespace(filter=conf.filter.EnsembleKalmanFilterPerturbedObservations(
        ensemble_size=3,
        ensemble_initial_std=1.0,
        gaspari_cohn_localization_radius=None,
        use_torch_rng_for_perturbed_observations=False,
    ))
    da_filter = SimpleNamespace(cfg=_DummyFilter(use_torch_rng_for_perturbed_observations=False))

    _apply_benchmark_runtime_overrides(cfg, da_filter)

    assert cfg.filter.use_torch_rng_for_perturbed_observations is True
    assert da_filter.cfg.use_torch_rng_for_perturbed_observations is True


def test_write_benchmark_timings_csv_has_requested_columns(tmp_path):
    cfg = SimpleNamespace(run_dir=tmp_path)
    csv_path = _write_benchmark_timings_csv(cfg, [0.01, 0.02])

    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert set(rows[0].keys()) == {
        'elapsed_s',
        'elapsed_ms',
        'benchmark_iters',
        'benchmark_warmup_iters',
        'synchronize_cuda',
        'call_index',
    }
    assert rows[0]['call_index'] == '0'
    assert rows[1]['call_index'] == '1'


def test_iterative_filter_config_fields_use_non_prefixed_names():
    cfg = conf.filter.EnsembleKalmanFilterPerturbedObservationsIterative(
        ensemble_size=3,
        ensemble_initial_std=1.0,
    )
    assert cfg.lag == 1
    assert cfg.niter == 10
    assert cfg.wtol == 1e-5
    assert not hasattr(cfg, 'ienks_lag')
    assert not hasattr(cfg, 'ienks_niter')
    assert not hasattr(cfg, 'ienks_wtol')


def test_ienks_wtol_none_runs_full_iteration_count():
    call_counter = {'count': 0}

    def model_propagator(_, ensemble, __):
        call_counter['count'] += 1
        return ensemble

    ensemble_f = torch.tensor([[[1.0, -0.5], [0.3, 0.8], [-1.1, 0.2]]], dtype=torch.float32)
    observation_y = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
    n_iter = 4
    lag = 1
    steps_between_analyses = 2

    filters_classical_iterative._ienks_analysis(
        ensemble_f=ensemble_f,
        observation_y=observation_y,
        observation_operator_ens=lambda x: x,
        sigma_y=0.1,
        upd_a='Sqrt',
        model_propagator=model_propagator,
        model_rhs=None,
        model_dt=None,
        nIter=n_iter,
        wtol=None,
        Lag=lag,
        steps_between_analyses=steps_between_analyses,
    )

    assert call_counter['count'] == n_iter * lag * steps_between_analyses


def test_ienks_numeric_wtol_can_stop_early():
    call_counter = {'count': 0}

    def model_propagator(_, ensemble, __):
        call_counter['count'] += 1
        return ensemble

    ensemble_f = torch.tensor([[[1.0, -0.5], [0.3, 0.8], [-1.1, 0.2]]], dtype=torch.float32)
    observation_y = torch.tensor([[0.1, -0.2]], dtype=torch.float32)
    lag = 1
    steps_between_analyses = 2

    filters_classical_iterative._ienks_analysis(
        ensemble_f=ensemble_f,
        observation_y=observation_y,
        observation_operator_ens=lambda x: x,
        sigma_y=0.1,
        upd_a='Sqrt',
        model_propagator=model_propagator,
        model_rhs=None,
        model_dt=None,
        nIter=4,
        wtol=1e12,
        Lag=lag,
        steps_between_analyses=steps_between_analyses,
    )

    assert call_counter['count'] == lag * steps_between_analyses


def test_enkf_pertobs_uses_torch_rng_when_provided_without_advancing_numpy_rng():
    rng = np.random.default_rng(123)
    rng_state_before = copy.deepcopy(rng.bit_generator.state)

    ensemble_f = torch.tensor([[1.0, -0.5], [0.3, 0.8], [-1.1, 0.2]], dtype=torch.float32)
    observation_y = torch.tensor([0.1, -0.2], dtype=torch.float32)

    torch_generator = torch.Generator(device='cpu')
    torch_generator.manual_seed(7)

    ensemble_a, _ = filters_classical.ensemble_kalman_filter_analysis(
        rng=rng,
        ensemble_f=ensemble_f,
        observation_y=observation_y,
        observation_operator_ens=lambda x: x,
        sigma_y=0.1,
        method='EnKF-PertObs',
        do_inflation=False,
        torch_noise_generator=torch_generator,
    )

    assert ensemble_a.shape == ensemble_f.shape
    assert rng.bit_generator.state == rng_state_before


def test_enkf_pertobs_default_numpy_rng_path_still_works():
    rng = np.random.default_rng(123)
    rng_state_before = copy.deepcopy(rng.bit_generator.state)

    ensemble_f = torch.tensor([[1.0, -0.5], [0.3, 0.8], [-1.1, 0.2]], dtype=torch.float32)
    observation_y = torch.tensor([0.1, -0.2], dtype=torch.float32)

    ensemble_a, _ = filters_classical.ensemble_kalman_filter_analysis(
        rng=rng,
        ensemble_f=ensemble_f,
        observation_y=observation_y,
        observation_operator_ens=lambda x: x,
        sigma_y=0.1,
        method='EnKF-PertObs',
        do_inflation=False,
    )

    assert ensemble_a.shape == ensemble_f.shape
    assert rng.bit_generator.state != rng_state_before
