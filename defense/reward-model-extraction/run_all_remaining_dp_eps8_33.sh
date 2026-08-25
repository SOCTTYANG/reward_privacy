#!/usr/bin/env bash
# Run this only after scripts/run_one_target_roberta_33_dp.sh has completed for Llama2-7B.
# Do not enable nounset: the cluster's /etc/bashrc references optional variables.
set -eo pipefail
source ~/.bashrc
PROJECT_DIR="${PROJECT_DIR:-/path/to/code}"
RUN_ONE="${PROJECT_DIR}/defense/reward-model-extraction/run_one_target_roberta_33_dp.sh"

bash "${RUN_ONE}" /path/to/models/llama2-13b-hf llama2_13b
bash "${RUN_ONE}" /path/to/models/llama32-3b llama32_3b
bash "${RUN_ONE}" /path/to/models/mistral-7b-v0.1 mistral_7b
bash "${RUN_ONE}" /path/to/models/qwen3-8b qwen3_8b
