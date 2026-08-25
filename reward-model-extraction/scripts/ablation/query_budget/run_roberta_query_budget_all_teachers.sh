#!/usr/bin/env bash
# Run the 50%/25% attacker-auxiliary-dataset query-budget ablation for four additional teachers.
# Every teacher uses an independently re-scored attacker-auxiliary-dataset subset, RoBERTa-base,
# the fixed 90/10 distillation split, and the standard teacher/student diff
# evaluation (500 train + 500 Defender Evaluation records; seed 42).

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/path/to/code}"
CODE_PROJECT="${CODE_PROJECT:-/path/to/code}"
CONDA_ENV="${CONDA_ENV:-rm_extract}"
ROBERTA_MODEL="${ROBERTA_MODEL:-${CODE_PROJECT}/models/roberta-base}"
MODEL_ROOT="${MODEL_ROOT:-/path/to/models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/output}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/query_budget_all_teachers}"
GPU="${GPU:-0}"

source ~/.bashrc
conda activate "${CONDA_ENV}"
set -u
cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

for data_file in attacker_auxiliary_dataset.jsonl attacker_preference_dataset_train.jsonl attacker_preference_dataset_test.jsonl test.jsonl; do
  test -f "${PROJECT_DIR}/data/${data_file}" || {
    echo "[ERROR] Missing ${PROJECT_DIR}/data/${data_file}" >&2
    exit 1
  }
done
test -f "${CODE_PROJECT}/data/train.jsonl"

run_student() {
  local teacher_name="$1"
  local budget_name="$2"
  local query_count="$3"
  local scored_aux="$4"
  local output_dir="${OUTPUT_ROOT}/ours_${teacher_name}_roberta_base_${budget_name}"

  CUDA_VISIBLE_DEVICES="${GPU}" python -m src.train_extracted_rm_two_stage \
    --student_model_path "${ROBERTA_MODEL}" \
    --attacker_preference_dataset_train_path data/attacker_preference_dataset_train.jsonl \
    --attacker_preference_dataset_eval_path data/attacker_preference_dataset_test.jsonl \
    --scored_aux_path "${scored_aux}" \
    --defender_eval_eval_path data/test.jsonl \
    --output_dir "${output_dir}" \
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
    > "${LOG_DIR}/${teacher_name}_${budget_name}_train.log" 2>&1
}

run_diff() {
  local teacher_name="$1"
  local teacher_base="$2"
  local teacher_adapter="$3"
  local budget_name="$4"

  CUDA_VISIBLE_DEVICES="${GPU}" python scripts/eval_target_vs_substitute_diff.py \
    --target_base_model "${teacher_base}" \
    --target_adapter_path "${teacher_adapter}" \
    --substitute_model_path "${OUTPUT_ROOT}/ours_${teacher_name}_roberta_base_${budget_name}" \
    --train_path "${CODE_PROJECT}/data/train.jsonl" \
    --test_path "${CODE_PROJECT}/data/test.jsonl" \
    --sample_train 500 \
    --sample_test 500 \
    --output_dir "${OUTPUT_ROOT}/diff_ours_${teacher_name}_roberta_base_${budget_name}" \
    --max_length 512 \
    --target_batch_size 1 \
    --substitute_batch_size 16 \
    --seed 42 \
    --bf16 \
    > "${LOG_DIR}/${teacher_name}_${budget_name}_diff.log" 2>&1
}

run_budget() {
  local teacher_name="$1"
  local teacher_base="$2"
  local teacher_adapter="$3"
  local budget_name="$4"
  local query_count="$5"
  local scored_aux="${PROJECT_DIR}/data/scored_aux_${teacher_name}_${budget_name}.jsonl"

  echo "[INFO] ${teacher_name}/${budget_name}: querying ${query_count} attacker-auxiliary-dataset records"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m src.score_aux_with_llama_lora \
    --base_model_path "${teacher_base}" \
    --lora_adapter_path "${teacher_adapter}" \
    --aux_path data/attacker_auxiliary_dataset.jsonl \
    --output_path "${scored_aux}" \
    --target_model_name "${teacher_name}" \
    --max_samples "${query_count}" \
    --batch_size 2 \
    --max_length 512 \
    --bf16 \
    > "${LOG_DIR}/${teacher_name}_${budget_name}_score.log" 2>&1

  run_student "${teacher_name}" "${budget_name}" "${query_count}" "${scored_aux}"
  run_diff "${teacher_name}" "${teacher_base}" "${teacher_adapter}" "${budget_name}"
}

run_teacher() {
  local teacher_name="$1"
  local teacher_base="$2"
  local teacher_adapter="$3"

  test -d "${teacher_base}"
  test -f "${teacher_adapter}/adapter_config.json"
  test -f "${teacher_adapter}/adapter_model.safetensors"

  run_budget "${teacher_name}" "${teacher_base}" "${teacher_adapter}" half 2500
  run_budget "${teacher_name}" "${teacher_base}" "${teacher_adapter}" quarter 1250
}

run_teacher mistral_7b \
  "${MODEL_ROOT}/mistral-7b-v0.1" \
  "${CODE_PROJECT}/output/target_rm_mistral_7b_full_margin5_e1"
run_teacher qwen3_8b \
  "${MODEL_ROOT}/qwen3-8b" \
  "${CODE_PROJECT}/output/target_rm_qwen3_8b_full_margin5_e1"
run_teacher llama32_3b \
  "${MODEL_ROOT}/llama32-3b" \
  "${CODE_PROJECT}/output/target_rm_llama32_3b_full_margin5_e1"
run_teacher llama2_13b \
  "${MODEL_ROOT}/llama2-13b-hf" \
  "${CODE_PROJECT}/output/target_rm_llama2_13b_full_margin5_e1"

echo "[OK] All four teacher query-budget ablations completed."
