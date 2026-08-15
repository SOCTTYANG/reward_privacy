#!/bin/bash
set -euo pipefail

BASELINE_SAVE_ROOT="${BASELINE_SAVE_ROOT:-/mnt/model_data/yang_safe/baseline}"
LOG_ROOT="${LOG_ROOT:-${BASELINE_SAVE_ROOT}/logs/baseline_spv_mia_membership_steps_5models}"
LINES="${LINES:-120}"

MODEL_TAGS=(
  "llama2-7b"
  "llama2-13b-hf"
  "llama32-3b"
  "qwen3-8b"
  "mistral-7b-v0.1"
)

LOG_FILES=()
for model_tag in "${MODEL_TAGS[@]}"; do
  LOG_FILES+=("${LOG_ROOT}/${model_tag}.log")
done

echo "============================================================"
echo "[TAIL] SPV-MIA five-model realtime logs"
echo "[LOG_ROOT] ${LOG_ROOT}"
echo "[LINES] ${LINES}"
echo "============================================================"
for log_file in "${LOG_FILES[@]}"; do
  echo "${log_file}"
done
echo "============================================================"

tail -n "${LINES}" -F "${LOG_FILES[@]}"
