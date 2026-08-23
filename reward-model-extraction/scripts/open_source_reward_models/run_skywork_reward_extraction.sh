#!/usr/bin/env bash
# Query the public Skywork reward model, then train a local substitute RM.
# Adjust PROJECT_DIR, SKyWORK_MODEL, and STUDENT_MODEL for the host before use.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SKYWORK_MODEL="${SKYWORK_MODEL:-/run/media/vipuser/Data/models/Skywork-Reward-Llama-3.1-8B}"
STUDENT_MODEL="${STUDENT_MODEL:-${PROJECT_DIR}/models/roberta-base}"
QUERY_COUNT="${QUERY_COUNT:-5000}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SKIP_QUERY="${SKIP_QUERY:-0}"
SCORED_AUX_PATH="${PROJECT_DIR}/data/scored_aux_skywork_reward_llama31_8b.jsonl"

cd "${PROJECT_DIR}"

if [[ "${SKIP_QUERY}" != "1" ]]; then
  python -m src.score_aux_with_skywork_reward \
    --model_path "${SKYWORK_MODEL}" \
    --aux_path "${PROJECT_DIR}/data/aux_dolly.jsonl" \
    --output_path "${SCORED_AUX_PATH}" \
    --max_samples "${QUERY_COUNT}" --batch_size 2 --max_length "${MAX_LENGTH}" --bf16
elif [[ ! -f "${SCORED_AUX_PATH}" ]]; then
  echo "[ERROR] SKIP_QUERY=1 requires an existing scored file: ${SCORED_AUX_PATH}" >&2
  exit 1
else
  echo "[INFO] Skipping target queries; reusing ${SCORED_AUX_PATH}"
fi

python -m src.train_extracted_rm_two_stage \
  --student_model_path "${STUDENT_MODEL}" \
  --hh_pref_train_path "${PROJECT_DIR}/data/hh_pref_train.jsonl" \
  --hh_pref_eval_path "${PROJECT_DIR}/data/hh_pref_test.jsonl" \
  --pku_pref_eval_path "${PROJECT_DIR}/data/pku10k_pref_eval.jsonl" \
  --scored_aux_path "${SCORED_AUX_PATH}" \
  --output_dir "${PROJECT_DIR}/output/extracted_rm_skywork_reward_llama31_8b" \
  --max_hh_train_samples 5000 --max_hh_eval_samples 1000 --max_pku_eval_samples 1000 \
  --max_aux_samples "${QUERY_COUNT}" --pref_epochs 1 --distill_epochs 1 \
  --pref_batch_size 8 --distill_batch_size 8 --eval_batch_size 16 \
  --max_length 512 --pref_lr 2e-5 --distill_lr 1e-5 --seed 42
