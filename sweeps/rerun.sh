#!/bin/bash
source ~/.bashrc
job_slot_to_gpu=(-1 4)
# job_slot_to_gpu=(-1 6 7)

# env_parallel --delay 0 --eta -j $((5 * (${#job_slot_to_gpu[@]} - 1) )) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' 'python src/dafm/rerun_from_alt_id.py --base-alt-id {alt_id} --target-setting-id {training_data_assimilation_setting_id} --rng-seed {rng_seed} --reference-filter {testing_reference_filter}' :::: notebooks/reruns.csv
env_parallel --delay 0 --eta -j $((5 * (${#job_slot_to_gpu[@]} - 1) )) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' 'python src/dafm/rerun_from_alt_id.py --base-alt-id {training_alt_id} --rng-seed {testing_rng_seed} --save-ensemble-stats {save_ensemble_stats}' :::: notebooks/reruns.csv
# env_parallel --eta --delay 0 -j $((5 * (${#job_slot_to_gpu[@]} - 1) )) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' 'python src/dafm/rerun_from_alt_id.py --base-alt-id {alt_id} --target-setting-id {training_data_assimilation_setting_id} --rng-seed {rng_seed}' ::: $(duckdb runs.sqlite -c "copy (select c.* from 'notebooks/reruns.csv' c join Conf using (alt_id) join Filter on Filter = Filter.id where sa_inheritance in ('LocalEnsembleTransformKalmanFilter')) to '/dev/stdout'")
