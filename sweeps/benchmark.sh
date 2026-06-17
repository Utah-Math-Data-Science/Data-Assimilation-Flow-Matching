#!/bin/bash
source ~/.bashrc
job_slot_to_gpu=(-1 4 5 6 7)
# job_slot_to_gpu=(-1 7)

# env_parallel --delay 0 --eta -j $((5 * (${#job_slot_to_gpu[@]} - 1) )) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' 'python src/dafm/rerun_from_alt_id.py --base-alt-id {alt_id} --target-setting-id {testing_data_assimilation_setting_id} --rng-seed {rng_seed} --reference-filter {testing_reference_filter}' :::: notebooks/reruns.csv
env_parallel --eta --delay 0 -j $((1 * (${#job_slot_to_gpu[@]} - 1) )) --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[(( ({%} % (${#job_slot_to_gpu[@]} - 1) ) + 1 ))]}' 'python src/dafm/benchmark_filter.py --alt-id {alt_id}' ::: $(duckdb -c "copy (select testing_alt_id as alt_id from 'sweeps/benchmark_*.csv' where filter_name = 'LETKF') to '/dev/stdout'")
