#!/usr/bin/env bash
# First DP verification run requested for Llama2-7B, then the complete 3.3 pipeline.
# Do not enable nounset: the cluster's /etc/bashrc references optional variables.
set -eo pipefail
source ~/.bashrc
PROJECT_DIR="${PROJECT_DIR:-/path/to/code}"
LLAMA2_7B_BASE="${LLAMA2_7B_BASE:-/path/to/target-model/llama2-7b}"
bash "${PROJECT_DIR}/defense/reward-model-extraction/run_one_target_roberta_33_dp.sh" "${LLAMA2_7B_BASE}" llama2_7b
