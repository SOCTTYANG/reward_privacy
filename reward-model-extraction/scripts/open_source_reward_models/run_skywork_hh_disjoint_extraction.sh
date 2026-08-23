#!/usr/bin/env bash
# Stage A and Stage B use disjoint HH source triples.  Stage B queries both
# chosen and rejected responses for every source triple.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SKYWORK_MODEL="${SKYWORK_MODEL:-/run/media/vipuser/Data/models/Skywork-Reward-Llama-3.1-8B}"
STUDENT_MODEL="${STUDENT_MODEL:-/mnt/bai_data/projects/bai-rm-extraction-exp/models/roberta-base}"
STAGE_A_PAIRS="${STAGE_A_PAIRS:-5000}"
STAGE_B_PAIRS="${STAGE_B_PAIRS:-5000}"

HH_TRAIN="${HH_TRAIN:-${PROJECT_DIR}/data/hh_pref_train.jsonl}"
HH_TEST="${HH_TEST:-${PROJECT_DIR}/data/hh_pref_test.jsonl}"
HH_AUX="${HH_AUX:-${PROJECT_DIR}/data/aux_hh_disjoint_for_skywork.jsonl}"
SCORED_HH_AUX="${SCORED_HH_AUX:-${PROJECT_DIR}/data/scored_aux_skywork_hh_disjoint.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/extracted_rm_skywork_hh_disjoint}"

cd "${PROJECT_DIR}"

# The source split is by triple before flattening: Stage A [0, 5000),
# Stage B [5000, 10000).  A triple can never leak one response into each stage.
python -m src.prepare_hh_aux_disjoint \
  --input_path "${HH_TRAIN}" --output_path "${HH_AUX}" \
  --skip_pairs "${STAGE_A_PAIRS}" --max_pairs "${STAGE_B_PAIRS}"

python -m src.score_aux_with_skywork_reward \
  --model_path "${SKYWORK_MODEL}" --aux_path "${HH_AUX}" \
  --output_path "${SCORED_HH_AUX}" --max_samples "$((STAGE_B_PAIRS * 2))" \
  --batch_size 2 --max_length 2048 --bf16

python -m src.train_extracted_rm_two_stage \
  --student_model_path "${STUDENT_MODEL}" \
  --hh_pref_train_path "${HH_TRAIN}" --hh_pref_eval_path "${HH_TEST}" \
  --pku_pref_eval_path "${PROJECT_DIR}/data/pku10k_pref_eval.jsonl" \
  --scored_aux_path "${SCORED_HH_AUX}" --output_dir "${OUTPUT_DIR}" \
  --max_hh_train_samples "${STAGE_A_PAIRS}" --max_hh_eval_samples 1000 \
  --max_pku_eval_samples 1000 --max_aux_samples "$((STAGE_B_PAIRS * 2))" \
  --pref_epochs 1 --distill_epochs 1 --pref_batch_size 8 --distill_batch_size 8 \
  --eval_batch_size 16 --max_length 512 --pref_lr 2e-5 --distill_lr 1e-5 --seed 42
