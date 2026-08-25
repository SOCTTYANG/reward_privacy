#!/usr/bin/env bash
# Experimental variant: unlike the Section 4.3 regression-only Stage B, this
# launcher additionally uses ranking derived solely from paired target scores.
# It is retained separately and never uses attacker-preference labels in Stage B.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STUDENT_MODEL="${STUDENT_MODEL:-/path/to/code/models/roberta-base}"
ATTACKER_PREFERENCE_TRAIN="${ATTACKER_PREFERENCE_TRAIN:-${PROJECT_DIR}/data/attacker_preference_dataset_10k/attacker_preference_dataset_train.jsonl}"
ATTACKER_PREFERENCE_TEST="${ATTACKER_PREFERENCE_TEST:-${PROJECT_DIR}/data/attacker_preference_dataset_10k/attacker_preference_dataset_test.jsonl}"
SCORED_ATTACKER_PREFERENCE_AUX="${SCORED_ATTACKER_PREFERENCE_AUX:-${PROJECT_DIR}/data/scored_aux_public_reward_model_attacker_preference_disjoint.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/extracted_rm_public_reward_model_attacker_preference_teacher_rank_zscore_lr2e6}"

cd "${PROJECT_DIR}"

python -m src.train_extracted_rm_two_stage \
  --student_model_path "${STUDENT_MODEL}" \
  --attacker_preference_dataset_train_path "${ATTACKER_PREFERENCE_TRAIN}" --attacker_preference_dataset_eval_path "${ATTACKER_PREFERENCE_TEST}" \
  --defender_eval_eval_path "${PROJECT_DIR}/data/defender_evaluation_pref_eval.jsonl" \
  --scored_aux_path "${SCORED_ATTACKER_PREFERENCE_AUX}" --output_dir "${OUTPUT_DIR}" \
  --max_attacker_preference_train_samples 5000 --max_attacker_preference_eval_samples 1000 \
  --max_defender_evaluation_eval_samples 1000 --max_aux_samples 10000 \
  --pref_epochs 1 --distill_epochs 1 --pref_batch_size 8 --distill_batch_size 8 \
  --eval_batch_size 16 --max_length 512 --pref_lr 2e-5 --distill_lr 2e-6 \
  --normalize_distill_scores \
  --stage_b_regression_weight 0.5 --stage_b_teacher_pair_weight 0.5 --seed 42
