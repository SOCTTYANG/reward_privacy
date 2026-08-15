#!/usr/bin/env bash
# Run this only after scripts/run_one_target_roberta_33_dp.sh has completed for Llama2-7B.
# Do not enable nounset: the cluster's /etc/bashrc references optional variables.
set -eo pipefail
source ~/.bashrc
PROJECT_DIR="${PROJECT_DIR:-/mnt/model_data/projects/bai-rm-extraction-exp}"
cd "${PROJECT_DIR}"

bash scripts/run_one_target_roberta_33_dp.sh /mnt/model_data/models/llama2-13b-hf llama2_13b
bash scripts/run_one_target_roberta_33_dp.sh /mnt/model_data/models/llama32-3b llama32_3b
bash scripts/run_one_target_roberta_33_dp.sh /mnt/model_data/models/mistral-7b-v0.1 mistral_7b
bash scripts/run_one_target_roberta_33_dp.sh /mnt/model_data/models/qwen3-8b qwen3_8b
