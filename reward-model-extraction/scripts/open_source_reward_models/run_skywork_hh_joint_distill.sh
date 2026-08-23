#!/usr/bin/env bash
# Reuse the existing 10k disjoint Skywork queries. Stage-B ranking direction
# is derived solely from each pair's two Skywork scores, never HH labels.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STUDENT_MODEL="${STUDENT_MODEL:-/mnt/bai_data/projects/bai-rm-extraction-exp/models/roberta-base}"
HH_TRAIN="${HH_TRAIN:-${PROJECT_DIR}/data/hh_pref_10k/hh_pref_train.jsonl}"
HH_TEST="${HH_TEST:-${PROJECT_DIR}/data/hh_pref_10k/hh_pref_test.jsonl}"
SCORED_HH_AUX="${SCORED_HH_AUX:-${PROJECT_DIR}/data/scored_aux_skywork_hh_disjoint.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/extracted_rm_skywork_hh_teacher_rank_zscore_lr2e6}"

cd "${PROJECT_DIR}"

python -m src.train_extracted_rm_two_stage \
  --student_model_path "${STUDENT_MODEL}" \
  --hh_pref_train_path "${HH_TRAIN}" --hh_pref_eval_path "${HH_TEST}" \
  --pku_pref_eval_path "${PROJECT_DIR}/data/pku10k_pref_eval.jsonl" \
  --scored_aux_path "${SCORED_HH_AUX}" --output_dir "${OUTPUT_DIR}" \
  --max_hh_train_samples 5000 --max_hh_eval_samples 1000 \
  --max_pku_eval_samples 1000 --max_aux_samples 10000 \
  --pref_epochs 1 --distill_epochs 1 --pref_batch_size 8 --distill_batch_size 8 \
  --eval_batch_size 16 --max_length 512 --pref_lr 2e-5 --distill_lr 2e-6 \
  --normalize_distill_scores \
  --stage_b_regression_weight 0.5 --stage_b_teacher_pair_weight 0.5 --seed 42
