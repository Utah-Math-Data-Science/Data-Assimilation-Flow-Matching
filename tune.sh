#!/bin/bash
source ~/.bashrc
job_slot_to_gpu=(-1 0)

# EnFF
env_parallel --eta -j 1 --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' python src/dafm/main.py dataset={dataset} model={model} model/guidance=LocalConstant model.guidance.schedule.constant={lambda} model.sampling_time_step_count={sampling_time_step_count} model.diffusion_path.sigma_min={sigma_min} ::: $(duckdb runs.sqlite -c "copy (select * from sweep_enff_lambda) to '/dev/stdout'") ::: $(duckdb runs.sqlite -c "copy (select * from sweep_enff_sigma_min) to '/dev/stdout'") ::: model FlowMatchingMarginalConditionalOptimalTransport FlowMatchingMarginalPreviousPosteriorToPredictive ::: dataset Lorenz96Bao2024EnSF KuramotoSivashinsky NavierStokesDim256 ::: $(duckdb runs.sqlite -c "copy (select * from sweep_sampling_time_step_count) to '/dev/stdout'")
# EnSF
env_parallel --eta -j 1 --colsep , --header : CUDA_VISIBLE_DEVICES='${job_slot_to_gpu[{%}]}' python src/dafm/main.py dataset={dataset} model=ScoreMatchingMarginalBao2024EnSF model.sampling_score_norm=LInfty model.sampling_time_step_count={sampling_time_step_count} model.diffusion_path.epsilon_alpha={epsilon_alpha} model.diffusion_path.epsilon_beta={epsilon_beta} ::: $(duckdb runs.sqlite -c "copy (select * from sweep_ensf_epsilon_alpha) to '/dev/stdout'") ::: $(duckdb runs.sqlite -c "copy (select * from sweep_ensf_epsilon_beta) to '/dev/stdout'") ::: dataset Lorenz96Bao2024EnSF KuramotoSivashinsky NavierStokesDim256 ::: $(duckdb runs.sqlite -c "copy (select * from sweep_sampling_time_step_count) to '/dev/stdout'")
