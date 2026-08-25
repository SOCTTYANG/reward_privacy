#!/usr/bin/env bash
set -eo pipefail

# Compare each Defender Evaluation-10K two-stage RoBERTa substitute against its own target RM.
# Results are written to output/diff_defender_evaluation_<teacher>_to_roberta/diff_summary.json.

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR="${PROJECT_DIR:-/path/to/code}"
MODEL_DIR="${MODEL_DIR:-/path/to/models}"
cd "${PROJECT_DIR}"

run_diff() {
  local teacher_name="$1"
  local target_base="$2"
  local target_adapter="$3"

  local student="${PROJECT_DIR}/output/defender_evaluation_${teacher_name}_to_roberta"
  local output_dir="${PROJECT_DIR}/output/diff_defender_evaluation_${teacher_name}_to_roberta"

  for required_path in "${target_base}" "${target_adapter}" "${student}"; do
    if [[ ! -d "${required_path}" ]]; then
      echo "[ERROR] Required model directory does not exist: ${required_path}" >&2
      exit 1
    fi
  done

  echo "======================================================"
  echo "Diff evaluation: ${teacher_name} target RM -> Defender Evaluation10K RoBERTa"
  echo "======================================================"

  python scripts/eval_target_vs_substitute_diff.py \
    --target_base_model "${target_base}" \
    --target_adapter_path "${target_adapter}" \
    --substitute_model_path "${student}" \
    --train_path "${PROJECT_DIR}/data/train.jsonl" \
    --test_path "${PROJECT_DIR}/data/test.jsonl" \
    --sample_train 500 \
    --sample_test 500 \
    --output_dir "${output_dir}" \
    --max_length 512 \
    --target_batch_size 1 \
    --substitute_batch_size 16 \
    --bf16
}

run_diff "llama2_13b" \
  "${MODEL_DIR}/llama2-13b-hf" \
  "${PROJECT_DIR}/output/target_rm_llama2_13b_full_margin5_e1"

run_diff "llama32_3b" \
  "${MODEL_DIR}/llama32-3b" \
  "${PROJECT_DIR}/output/target_rm_llama32_3b_full_margin5_e1"

run_diff "mistral_7b" \
  "${MODEL_DIR}/mistral-7b-v0.1" \
  "${PROJECT_DIR}/output/target_rm_mistral_7b_full_margin5_e1"

run_diff "qwen3_8b" \
  "${MODEL_DIR}/qwen3-8b" \
  "${PROJECT_DIR}/output/target_rm_qwen3_8b_full_margin5_e1"

echo "[OK] All four Defender Evaluation-10K diff evaluations finished."
echo "[INFO] Summaries: output/diff_defender_evaluation_*_to_roberta/diff_summary.json"
