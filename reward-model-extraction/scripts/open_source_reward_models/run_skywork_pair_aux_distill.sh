#!/usr/bin/env bash
# Train a clean RoBERTa student with HH Stage A and a supplied 500-pair
# preference file as the 1,000-response Skywork Stage-B query budget.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TARGET_MODEL="${TARGET_MODEL:-/run/media/vipuser/Data/models/Skywork-Reward-Llama-3.1-8B}"
# The base RoBERTa checkpoint is stored on the shared model volume, while the
# experiment repository itself lives under /mnt/model_data.
STUDENT_MODEL="${STUDENT_MODEL:-/mnt/bai_data/projects/bai-rm-extraction-exp/models/roberta-base}"
HH_TRAIN="${HH_TRAIN:-${PROJECT_DIR}/data/hh_pref_10k/hh_pref_train.jsonl}"
HH_TEST="${HH_TEST:-${PROJECT_DIR}/data/hh_pref_10k/hh_pref_test.jsonl}"
PAIR_SOURCE="${PAIR_SOURCE:?Set PAIR_SOURCE to a 500-pair JSONL file.}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME, e.g. skywork_member500.}"
WORK_DIR="${WORK_DIR:-${PROJECT_DIR}/data/${RUN_NAME}_aux}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/extracted_rm_${RUN_NAME}_teacher_rank_zscore_lr2e6}"

mkdir -p "${WORK_DIR}"
QUERIES="${WORK_DIR}/queries.jsonl"
SCORED="${WORK_DIR}/scored_skywork.jsonl"

cd "${PROJECT_DIR}"
python -m src.prepare_preference_aux --input_path "${PAIR_SOURCE}" --output_path "${QUERIES}" --max_pairs 500
python -m src.score_aux_with_skywork_reward \
  --model_path "${TARGET_MODEL}" --aux_path "${QUERIES}" --output_path "${SCORED}" \
  --batch_size 2 --max_length 2048 --bf16
python -m src.train_extracted_rm_two_stage \
  --student_model_path "${STUDENT_MODEL}" \
  --hh_pref_train_path "${HH_TRAIN}" --hh_pref_eval_path "${HH_TEST}" \
  --pku_pref_eval_path "${PROJECT_DIR}/data/pku10k_pref_eval.jsonl" \
  --scored_aux_path "${SCORED}" --output_dir "${OUTPUT_DIR}" \
  --max_hh_train_samples 5000 --max_hh_eval_samples 1000 \
  --max_pku_eval_samples 1000 --max_aux_samples 1000 \
  --pref_epochs 1 --distill_epochs 1 --pref_batch_size 8 --distill_batch_size 8 \
  --eval_batch_size 16 --max_length 512 --pref_lr 2e-5 --distill_lr 2e-6 \
  --normalize_distill_scores \
  --stage_b_regression_weight 0.5 --stage_b_teacher_pair_weight 0.5 --seed 42
