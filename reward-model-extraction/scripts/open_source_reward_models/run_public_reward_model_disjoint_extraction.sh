#!/usr/bin/env bash
# Stage A and Stage B use disjoint Attacker Preference source triples.  Stage B queries both
# chosen and rejected responses for every source triple.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PUBLIC_REWARD_MODEL="${PUBLIC_REWARD_MODEL:-/path/to/public-reward-model}"
STUDENT_MODEL="${STUDENT_MODEL:-/path/to/code/models/roberta-base}"
STAGE_A_PAIRS="${STAGE_A_PAIRS:-5000}"
STAGE_B_PAIRS="${STAGE_B_PAIRS:-5000}"

ATTACKER_PREFERENCE_TRAIN="${ATTACKER_PREFERENCE_TRAIN:-${PROJECT_DIR}/data/attacker_preference_dataset_train.jsonl}"
ATTACKER_PREFERENCE_TEST="${ATTACKER_PREFERENCE_TEST:-${PROJECT_DIR}/data/attacker_preference_dataset_test.jsonl}"
ATTACKER_PREFERENCE_AUX="${ATTACKER_PREFERENCE_AUX:-${PROJECT_DIR}/data/aux_attacker_preference_disjoint_for_public_reward_model.jsonl}"
SCORED_ATTACKER_PREFERENCE_AUX="${SCORED_ATTACKER_PREFERENCE_AUX:-${PROJECT_DIR}/data/scored_aux_public_reward_model_attacker_preference_disjoint.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/extracted_rm_public_reward_model_attacker_preference_disjoint}"

cd "${PROJECT_DIR}"

# The source split is by triple before flattening: Stage A [0, 5000),
# Stage B [5000, 10000).  A triple can never leak one response into each stage.
python -m src.prepare_attacker_auxiliary_dataset_disjoint \
  --input_path "${ATTACKER_PREFERENCE_TRAIN}" --output_path "${ATTACKER_PREFERENCE_AUX}" \
  --skip_pairs "${STAGE_A_PAIRS}" --max_pairs "${STAGE_B_PAIRS}"

python -m src.score_aux_with_public_reward_model \
  --model_path "${PUBLIC_REWARD_MODEL}" --aux_path "${ATTACKER_PREFERENCE_AUX}" \
  --output_path "${SCORED_ATTACKER_PREFERENCE_AUX}" --max_samples "$((STAGE_B_PAIRS * 2))" \
  --batch_size 2 --max_length 2048 --bf16

python -m src.train_extracted_rm_two_stage \
  --student_model_path "${STUDENT_MODEL}" \
  --attacker_preference_dataset_train_path "${ATTACKER_PREFERENCE_TRAIN}" --attacker_preference_dataset_eval_path "${ATTACKER_PREFERENCE_TEST}" \
  --defender_eval_eval_path "${PROJECT_DIR}/data/defender_evaluation_pref_eval.jsonl" \
  --scored_aux_path "${SCORED_ATTACKER_PREFERENCE_AUX}" --output_dir "${OUTPUT_DIR}" \
  --max_attacker_preference_train_samples "${STAGE_A_PAIRS}" --max_attacker_preference_eval_samples 1000 \
  --max_defender_evaluation_eval_samples 1000 --max_aux_samples "$((STAGE_B_PAIRS * 2))" \
  --pref_epochs 1 --distill_epochs 1 --pref_batch_size 8 --distill_batch_size 8 \
  --eval_batch_size 16 --max_length 512 --pref_lr 2e-5 --distill_lr 1e-5 --seed 42
