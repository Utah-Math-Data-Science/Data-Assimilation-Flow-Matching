import argparse
import csv
import pprint
import sys
import time
import numpy as np
import sqlalchemy as sa
import torch
from einops import rearrange

import conf.conf
import conf.filter
import dafm.datasets
import dafm.filters
import dafm.observe
import dafm.utils


log = dafm.utils.getLoggerByFilename(__file__)


BENCHMARK_WARMUP_ITERS = 1
BENCHMARK_ITERS = 50
BENCHMARK_SYNCHRONIZE_CUDA = True
BENCHMARK_TIMINGS_FILENAME = 'benchmark_filter.csv'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Benchmark filter assimilate() from an existing Conf alt_id.')
    parser.add_argument('--alt-id', required=True, help='Existing Conf.alt_id to benchmark.')
    return parser.parse_args(argv)


def _synchronize_if_needed(cfg: conf.conf.Conf):
    if BENCHMARK_SYNCHRONIZE_CUDA and str(cfg.device).startswith('cuda'):
        torch.cuda.synchronize(device=cfg.device)


def _build_assimilation_inputs(cfg: conf.conf.Conf):
    dataset = dafm.datasets.get_dataset(
        cfg.setting.dataset,
        np.random.default_rng(dafm.utils.RNG_RANDBITS[cfg.setting.dataset.rng_seed]['DATASET']),
        cfg.device,
    )
    dataset.prepare_data()
    dataset.setup('fit')

    observe = dafm.observe.build_observe(
        cfg.setting.observes,
        cfg_dataset=cfg.setting.dataset,
        rng=np.random.default_rng(dafm.utils.RNG_RANDBITS[cfg.rng_seed]['OBSERVE']),
    )
    observe_unflatten = dafm.observe.Unflatten(observe)

    rng_initial_condition, rng_obs_noise = np.random.default_rng(
        dafm.utils.RNG_RANDBITS[cfg.rng_seed]['DATA_ASSIMILATION']
    ).spawn(2)

    split_start = getattr(cfg.setting.splitter, f'start_{cfg.setting.split}')
    true_initial_condition = dataset[split_start]
    if cfg.setting.ensemble_initial_mean_is_true_state:
        loc = true_initial_condition
    else:
        loc = torch.zeros_like(true_initial_condition)
    filtering_ensemble = torch.tensor(
        rng_initial_condition.normal(
            loc=loc.numpy(),
            scale=cfg.filter.ensemble_initial_std,
            size=(cfg.filter.ensemble_size, *true_initial_condition.shape),
        ),
        device=cfg.device,
        dtype=true_initial_condition.dtype,
    )

    time_step = next(iter(cfg.setting.observation_time_steps))
    predictive_ensemble = dataset.integrate(split_start, time_step, filtering_ensemble)[-1]
    data = dataset[time_step][None].to(cfg.device)

    observation = observe(data)
    observation = observation + observe.get_sparsity_mask(device=cfg.device) * torch.tensor(
        rng_obs_noise.normal(scale=cfg.setting.obs_noise_std, size=observation.shape),
        device=cfg.device,
        dtype=observation.dtype,
    )

    return dict(
        dataset=dataset,
        observe_unflatten=observe_unflatten,
        time_step=time_step,
        initial_ensemble_flat=rearrange(filtering_ensemble, 'member ... -> member (...)'),
        predictive_ensemble_flat=rearrange(predictive_ensemble, 'member ... -> member (...)'),
        observation_flat=rearrange(observation, '1 ... -> 1 (...)'),
    )


def _apply_benchmark_runtime_overrides(cfg: conf.conf.Conf, da_filter):
    if isinstance(cfg.filter, conf.filter.EnsembleKalmanFilterPerturbedObservationsIterative):
        cfg.filter.wtol = None
        da_filter.cfg.wtol = None
    if isinstance(cfg.filter, conf.filter.EnsembleKalmanFilterPerturbedObservations):
        cfg.filter.use_torch_rng_for_perturbed_observations = True
        da_filter.cfg.use_torch_rng_for_perturbed_observations = True


def _write_benchmark_timings_csv(cfg: conf.conf.Conf, per_call_elapsed_s):
    csv_path = cfg.run_dir / BENCHMARK_TIMINGS_FILENAME
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'elapsed_s',
                'elapsed_ms',
                'benchmark_iters',
                'benchmark_warmup_iters',
                'synchronize_cuda',
                'call_index',
            ],
        )
        writer.writeheader()
        for call_index, elapsed_s in enumerate(per_call_elapsed_s):
            writer.writerow(dict(
                elapsed_s=elapsed_s,
                elapsed_ms=1000.0 * elapsed_s,
                benchmark_iters=BENCHMARK_ITERS,
                benchmark_warmup_iters=BENCHMARK_WARMUP_ITERS,
                synchronize_cuda=BENCHMARK_SYNCHRONIZE_CUDA,
                call_index=call_index,
            ))
    return csv_path


def benchmark_filter(cfg: conf.conf.Conf):
    inputs = _build_assimilation_inputs(cfg)

    da_filter = dafm.filters.get_filter(
        cfg.filter,
        cfg_dataset=cfg.setting.dataset,
        rng=np.random.default_rng(dafm.utils.RNG_RANDBITS[cfg.rng_seed]['FILTER']),
    )
    _apply_benchmark_runtime_overrides(cfg, da_filter)

    if hasattr(cfg.filter, 'prob_path') and isinstance(cfg.filter.prob_path, conf.prob_path.FilteringToPredictive):
        da_filter.prob_path.set_previous_filtering(inputs['initial_ensemble_flat'])

    da_filter.initialize(cfg.setting, inputs['initial_ensemble_flat'], inputs['dataset'])

    for _ in range(BENCHMARK_WARMUP_ITERS):
        da_filter.assimilate(
            inputs['time_step'],
            inputs['predictive_ensemble_flat'],
            inputs['observation_flat'],
            cfg.setting.obs_noise_std,
            inputs['observe_unflatten'],
        )

    _synchronize_if_needed(cfg)
    per_call_elapsed_s = []
    last_ensemble = None
    for _ in range(BENCHMARK_ITERS):
        _synchronize_if_needed(cfg)
        start = time.perf_counter()
        last_ensemble = da_filter.assimilate(
            inputs['time_step'],
            inputs['predictive_ensemble_flat'],
            inputs['observation_flat'],
            cfg.setting.obs_noise_std,
            inputs['observe_unflatten'],
        )
        _synchronize_if_needed(cfg)
        end = time.perf_counter()
        per_call_elapsed_s.append(end - start)

    total_s = float(sum(per_call_elapsed_s))
    mean_ms = 1000.0 * total_s / BENCHMARK_ITERS
    checksum = float(last_ensemble.sum().detach().cpu().item()) if last_ensemble is not None else float('nan')
    timings_csv_path = _write_benchmark_timings_csv(cfg, per_call_elapsed_s)

    return dict(
        alt_id=cfg.alt_id,
        filter=type(cfg.filter).__name__,
        dataset=type(cfg.setting.dataset.system).__name__,
        device=str(cfg.device),
        warmup_iters=BENCHMARK_WARMUP_ITERS,
        iters=BENCHMARK_ITERS,
        synchronize_cuda=BENCHMARK_SYNCHRONIZE_CUDA,
        total_s=total_s,
        mean_ms=mean_ms,
        output_checksum=checksum,
        timings_csv_path=str(timings_csv_path),
    )


def run(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    with conf.conf.Session() as db:
        cfg = db.execute(sa.select(conf.conf.Conf).filter_by(alt_id=args.alt_id)).first()
        if cfg is None:
            raise ValueError(f'Unknown Conf alt_id: {args.alt_id}')
        cfg = cfg[0]

        log.info('Command: python %s', ' '.join(['src/dafm/benchmark_filter.py', *argv]))
        log.info('Benchmark alt_id: %s', cfg.alt_id)
        log.info(pprint.pformat(dict(filter=cfg.filter, dataset=cfg.setting.dataset, device=cfg.device)))
        log.info(pprint.pformat(dict(
            BENCHMARK_WARMUP_ITERS=BENCHMARK_WARMUP_ITERS,
            BENCHMARK_ITERS=BENCHMARK_ITERS,
            BENCHMARK_SYNCHRONIZE_CUDA=BENCHMARK_SYNCHRONIZE_CUDA,
            BENCHMARK_TIMINGS_FILENAME=BENCHMARK_TIMINGS_FILENAME,
        )))

        result = benchmark_filter(cfg)
        log.info('Per-call timing CSV: %s', result['timings_csv_path'])
        print(
            'benchmark_filter '
            f'alt_id={result["alt_id"]} '
            f'filter={result["filter"]} '
            f'dataset={result["dataset"]} '
            f'device={result["device"]} '
            f'warmup_iters={result["warmup_iters"]} '
            f'iters={result["iters"]} '
            f'synchronize_cuda={result["synchronize_cuda"]} '
            f'total_s={result["total_s"]:.6f} '
            f'mean_ms={result["mean_ms"]:.6f} '
            f'output_checksum={result["output_checksum"]:.6f}'
        )


if __name__ == '__main__':
    run()
