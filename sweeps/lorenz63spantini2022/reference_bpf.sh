#!/bin/bash
source ~/.bashrc
job_slot_to_gpu=(-1 0 1 2 3 4)

# env_parallel --delay 0 --eta --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' 'python src/dafm/main.py rng_seed={rng_seed} +experiment=Lorenz63Spantini2022ReferenceBPF setting.observe_every_n_time_steps={observe_every_n_time_steps} save_ensemble_stats=true' ::: observe_every_n_time_steps 2 4 8 16 ::: rng_seed 2376999025 462133975 979497033 97616566 715319214 19704671
env_parallel --delay 0 --eta --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' 'python src/dafm/main.py rng_seed={rng_seed} +experiment=Lorenz63Spantini2022ReferenceBPF setting.observe_every_n_time_steps={observe_every_n_time_steps} setting.obs_noise_std=.1 save_ensemble_stats=true "setting.observes=[{_target_:conf.observe.ObserveATan,order:0}]"' ::: observe_every_n_time_steps 2 4 8 ::: rng_seed 462133975 979497033 97616566 715319214 19704671
