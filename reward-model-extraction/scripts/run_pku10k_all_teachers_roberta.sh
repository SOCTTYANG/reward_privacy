#!/usr/bin/env bash
set -eo pipefail

# Run the four missing PKU-10K Stage-A + teacher-distillation Stage-B
# experiments for a RoBERTa-base student.  The target RMs and their scored
# auxiliary data must already have been produced before this script is run.

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME="${HF_HOME:-/mnt/bai_data/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-/mnt/bai_data/cache/torch}"

# By default use the repository that contains this script.  PROJECT_DIR can
# still be overridden for a different mounted checkout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if [[ -z "${STUDENT_MODEL:-}" ]]; then
  if [[ -d "${PROJECT_DIR}/models/roberta-base" ]]; then
    STUDENT_MODEL="${PROJECT_DIR}/models/roberta-base"
  elif [[ -d "/mnt/model_data/models/roberta-base" ]]; then
    STUDENT_MODEL="/mnt/model_data/models/roberta-base"
  elif [[ -d "/mnt/bai_data/models/roberta-base" ]]; then
    STUDENT_MODEL="/mnt/bai_data/models/roberta-base"
  else
    STUDENT_MODEL="${PROJECT_DIR}/models/roberta-base"
  fi
fi
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/pku10k_teacher_scaling_roberta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/output}"

cd "${PROJECT_DIR}"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

run_experiment() {
  local teacher_name="$1"
  local scored_aux_name="$2"

  local scored_aux_path="${PROJECT_DIR}/data/${scored_aux_name}"
  local output_dir="${OUTPUT_ROOT}/pku10k_${teacher_name}_to_roberta"
  local metrics_path="${output_dir}/two_stage_extracted_rm_metrics.json"
  local log_path="${LOG_DIR}/pku10k_${teacher_name}_to_roberta.log"

  for required_path in \
    "${STUDENT_MODEL}" \
    "${PROJECT_DIR}/data/pku10k_pref_train.jsonl" \
    "${PROJECT_DIR}/data/pku10k_pref_eval.jsonl" \
    "${PROJECT_DIR}/data/test.jsonl" \
    "${scored_aux_path}"
  do
    if [[ ! -e "${required_path}" ]]; then
      echo "[ERROR] Missing required path: ${required_path}" >&2
      exit 1
    fi
  done

  if [[ -f "${metrics_path}" ]]; then
    echo "[SKIP] Completed metrics already exist: ${metrics_path}"
    return
  fi

  echo "======================================================"
  echo "PKU-10K two-stage extraction: ${teacher_name} -> RoBERTa-base"
  echo "Stage A: PKU-10K preference data | Stage B: teacher distillation"
  echo "Output: ${output_dir}"
  echo "======================================================"

  python -m src.train_extracted_rm_two_stage \
    --student_model_path "${STUDENT_MODEL}" \
    --hh_pref_train_path "${PROJECT_DIR}/data/pku10k_pref_train.jsonl" \
    --hh_pref_eval_path "${PROJECT_DIR}/data/pku10k_pref_eval.jsonl" \
    --scored_aux_path "${scored_aux_path}" \
    --pku_pref_eval_path "${PROJECT_DIR}/data/test.jsonl" \
    --output_dir "${output_dir}" \
    --max_hh_train_samples 9500 \
    --max_hh_eval_samples 500 \
    --max_aux_samples 5000 \
    --max_pku_eval_samples 1000 \
    --aux_train_ratio 0.9 \
    --pref_epochs 1 \
    --distill_epochs 1 \
    --pref_batch_size 8 \
    --distill_batch_size 8 \
    --eval_batch_size 16 \
    --max_length 512 \
    --pref_lr 2e-5 \
    --distill_lr 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --grad_clip 1.0 \
    --seed 42 \
    2>&1 | tee "${log_path}"
}

run_experiment "llama2_13b" "scored_aux_llama2_13b_full_margin5_e1.jsonl"
run_experiment "llama32_3b" "scored_aux_llama32_3b_full_margin5_e1.jsonl"
run_experiment "mistral_7b" "scored_aux_mistral_7b_full_margin5_e1.jsonl"
run_experiment "qwen3_8b" "scored_aux_qwen3_8b_full_margin5_e1.jsonl"

echo "[OK] All four PKU-10K RoBERTa-base extraction runs finished."
