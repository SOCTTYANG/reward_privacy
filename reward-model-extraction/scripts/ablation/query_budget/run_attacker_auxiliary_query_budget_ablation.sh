#!/usr/bin/env bash
# Query-budget ablation for model extraction.
#
# Runs two independent experiments with 2,500 (50%) and 1,250 (25%) attacker-auxiliary-dataset
# queries.  Each query budget is scored afresh by the fixed LLaMA2-7B teacher,
# then split deterministically into 90% distillation train / 10% validation.

# Some cluster /etc/bashrc files read unset variables, so enable nounset only
# after the shell and Conda environment have been initialized.
set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/code}"
CONDA_ENV="${CONDA_ENV:-rm_extract}"
TEACHER_BASE="${TEACHER_BASE:-/path/to/target-model/llama2-7b}"
TEACHER_ADAPTER="${TEACHER_ADAPTER:-${PROJECT_DIR}/output/target_rm_llama2_7b_lora_full_margin5_e2}"
ROBERTA_MODEL="${ROBERTA_MODEL:-${PROJECT_DIR}/models/roberta-base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/output}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/query_budget_ablation}"
GPU="${GPU:-0}"

source ~/.bashrc
conda activate "${CONDA_ENV}"
set -u
cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}"

test -f "${TEACHER_ADAPTER}/adapter_config.json"
test -f "${TEACHER_ADAPTER}/adapter_model.safetensors"

run_student() {
  local scored_aux="$1"
  local budget_name="$2"
  local query_count="$3"

  CUDA_VISIBLE_DEVICES="${GPU}" python -m src.train_extracted_rm_two_stage \
    --student_model_path "${ROBERTA_MODEL}" \
    --attacker_preference_dataset_train_path data/attacker_preference_dataset_train.jsonl \
    --attacker_preference_dataset_eval_path data/attacker_preference_dataset_test.jsonl \
    --scored_aux_path "${scored_aux}" \
    --defender_eval_eval_path data/test.jsonl \
    --output_dir "${OUTPUT_ROOT}/ours_llama2_7b_roberta_base_${budget_name}" \
    --max_attacker_preference_train_samples 5000 \
    --max_attacker_preference_eval_samples 1000 \
    --max_aux_samples "${query_count}" \
    --max_defender_evaluation_eval_samples 1000 \
    --aux_train_ratio 0.9 \
    --pref_epochs 1 \
    --distill_epochs 1 \
    --pref_batch_size 8 \
    --distill_batch_size 8 \
    --eval_batch_size 16 \
    --max_length 512 \
    --pref_lr 2e-5 \
    --distill_lr 1e-5 \
    --seed 42 \
    > "${LOG_DIR}/roberta_base_${budget_name}.log" 2>&1
}

run_budget() {
  local budget_name="$1"
  local query_count="$2"
  local scored_aux="data/scored_aux_llama2_7b_margin5_e2_${budget_name}.jsonl"

  echo "[INFO] ${budget_name}: querying ${query_count} attacker-auxiliary-dataset records"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m src.score_aux_with_llama_lora \
    --base_model_path "${TEACHER_BASE}" \
    --lora_adapter_path "${TEACHER_ADAPTER}" \
    --aux_path data/attacker_auxiliary_dataset.jsonl \
    --output_path "${scored_aux}" \
    --max_samples "${query_count}" \
    --batch_size 2 \
    --max_length 512 \
    --bf16 \
    > "${LOG_DIR}/score_${budget_name}.log" 2>&1

  # 2,500 -> 2,250 train + 250 validation; 1,250 -> 1,125 train + 125 validation.
  run_student "${scored_aux}" "${budget_name}" "${query_count}"
}

run_budget half 2500
run_budget quarter 1250

echo "[OK] Query-budget ablation complete. Results are under ${OUTPUT_ROOT}."
