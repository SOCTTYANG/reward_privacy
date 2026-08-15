#!/usr/bin/env bash
set -euo pipefail

M_VALUE="${M_VALUE:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MI_DIR="${PROJECT_DIR}/method"

STAGE32_SCRIPT="${MI_DIR}/3_2_candidate_response_generation.py"
STAGE33_SCRIPT="${STAGE33_SCRIPT:-${MI_DIR}/3_3_llm_update_ppo_full.py}"
STAGE34_SCRIPT="${MI_DIR}/3_4_membership_inference.py"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -x "/home/vipuser/.conda/envs/rm_extract/bin/python" ]]; then
    PYTHON_BIN="/home/vipuser/.conda/envs/rm_extract/bin/python"
  elif [[ -x "/home/vipuser/.conda/envs/safe-rlhf/bin/python" ]]; then
    PYTHON_BIN="/home/vipuser/.conda/envs/safe-rlhf/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/model_data/yang_safe/mia_ppo_gradient_only_m${M_VALUE}_5models_cuda0}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
export HF_HOME="${HF_HOME:-${OUTPUT_ROOT}/huggingface_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"

MEMBER_FILE="${MEMBER_FILE:-/mnt/bai_data/yang-safe/data/mia_overfit/rm_member_train_512.jsonl}"
if [[ ! -f "${MEMBER_FILE}" && -f "/run/media/vipuser/data/yang-safe/data/mia_overfit/rm_member_train_512.jsonl" ]]; then
  MEMBER_FILE="/run/media/vipuser/data/yang-safe/data/mia_overfit/rm_member_train_512.jsonl"
fi

NONMEMBER_FILE="${NONMEMBER_FILE:-/mnt/bai_data/yang-safe/data/mia_overfit/rm_nonmember_test_512.jsonl}"
if [[ ! -f "${NONMEMBER_FILE}" && -f "/run/media/vipuser/data/yang-safe/data/mia_overfit/rm_nonmember_test_512.jsonl" ]]; then
  NONMEMBER_FILE="/run/media/vipuser/data/yang-safe/data/mia_overfit/rm_nonmember_test_512.jsonl"
fi
if [[ ! -f "${NONMEMBER_FILE}" ]]; then
  echo "[WARN] Cannot find NONMEMBER_FILE: ${NONMEMBER_FILE}"
  echo "[WARN] Falling back to PKU test set."
  NONMEMBER_FILE="/mnt/bai_data/yang-safe/data/pku30k_filtered/pku30k_test_cosine_le_085_xyz.jsonl"
fi

# Fixed reward model. Only pretrained/policy model changes across runs.
RM_BASE_MODEL="${RM_BASE_MODEL:-/home/vipuser/Desktop/model/llama2-7b}"
RM_ADAPTER="${RM_ADAPTER:-/mnt/bai_data/yang-safe/output/rm_lora_overfit_member512_e80_lr1e4_ga1_r64_margin8}"

LLAMA2_7B="${LLAMA2_7B:-/home/vipuser/Desktop/model/llama2-7b}"
MISTRAL_7B="${MISTRAL_7B:-/mnt/model_data/models/mistral-7b-v0.1}"
LLAMA32_3B="${LLAMA32_3B:-/mnt/model_data/models/llama32-3b}"
QWEN3_8B="${QWEN3_8B:-/mnt/model_data/models/qwen3-8b}"
LLAMA2_13B="${LLAMA2_13B:-/mnt/model_data/models/llama2-13b-hf}"

MODEL_TAGS=("llama2-7b" "mistral-7b-v0.1" "llama32-3b" "qwen3-8b" "llama2-13b-hf")
MODEL_PATHS=("${LLAMA2_7B}" "${MISTRAL_7B}" "${LLAMA32_3B}" "${QWEN3_8B}" "${LLAMA2_13B}")

MEMBER_SIZE="${MEMBER_SIZE:-512}"
NONMEMBER_SIZE="${NONMEMBER_SIZE:-512}"
SEED="${SEED:-42}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
REWARD_BATCH_SIZE="${REWARD_BATCH_SIZE:-1}"
REWARD_MAX_LENGTH="${REWARD_MAX_LENGTH:-768}"
POLICY_MAX_LENGTH="${POLICY_MAX_LENGTH:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_MODEL_TAGS="${RUN_MODEL_TAGS:-}"
SKIP_MODEL_TAGS="${SKIP_MODEL_TAGS:-}"
OVERWRITE="${OVERWRITE:-0}"

GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
STAGE3_SAVE_STEPS="${STAGE3_SAVE_STEPS:-100000000}"

LAMBDA2="${LAMBDA2:-1.0}"
TARGET_FPR="${TARGET_FPR:-5.0}"
CLIP_EPS="${CLIP_EPS:-0.2}"
DELTA="${DELTA:?Set DELTA to a fixed threshold selected on an independent calibration set}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing ${label}: ${path}"
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing ${label}: ${path}"
    exit 1
  fi
}

tag_selected() {
  local needle="$1"
  local csv="$2"
  local item
  local -a items
  [[ -z "${csv}" ]] && return 0
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

tag_skipped() {
  local needle="$1"
  local csv="$2"
  local item
  local -a items
  [[ -z "${csv}" ]] && return 1
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

best_policy_dir() {
  local stage33_dir="$1"
  local found_policy

  if [[ -f "${stage33_dir}/final_policy/config.json" ]]; then
    echo "${stage33_dir}/final_policy"
    return 0
  fi

  found_policy="$(find "${stage33_dir}" -type f -name config.json 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -n "${found_policy}" ]]; then
    dirname "${found_policy}"
  fi
}

require_file "${STAGE32_SCRIPT}" "MIA step-2 script"
require_file "${STAGE33_SCRIPT}" "MIA step-3 script"
require_file "${STAGE34_SCRIPT}" "MIA step-4 script"
require_file "${MEMBER_FILE}" "MEMBER_FILE"
require_file "${NONMEMBER_FILE}" "NONMEMBER_FILE"
require_dir "${RM_BASE_MODEL}" "RM_BASE_MODEL"
require_dir "${RM_ADAPTER}" "RM_ADAPTER"

for i in "${!MODEL_TAGS[@]}"; do
  if tag_selected "${MODEL_TAGS[$i]}" "${RUN_MODEL_TAGS}" && ! tag_skipped "${MODEL_TAGS[$i]}" "${SKIP_MODEL_TAGS}"; then
    require_dir "${MODEL_PATHS[$i]}" "${MODEL_TAGS[$i]}"
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

SUMMARY_CSV="${OUTPUT_ROOT}/mia_ppo_gradient_only_m${M_VALUE}_table_metrics.csv"
MANIFEST_TSV="${OUTPUT_ROOT}/mia_ppo_gradient_only_m${M_VALUE}_outputs.tsv"
printf "model_tag,ASR,AUC,T@1%%F,T@5%%F\n" > "${SUMMARY_CSV}"
printf "model_tag\ttable_csv\tsummary_json\tpolicy_model\trun_dir\n" > "${MANIFEST_TSV}"

echo "============================================================"
echo "[MIA PPO-Gradient-Only Pipeline]"
echo "[M         ] ${M_VALUE}"
echo "[GPU       ] CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}"
echo "[FIXED RM  ] ${RM_BASE_MODEL} + ${RM_ADAPTER}"
echo "[SIGNAL    ] final_score = -${LAMBDA2} * PPO grad_norm; no reward gap signal"
echo "[VARY      ] only pretrained/policy model changes"
echo "[OUTPUT    ] ${OUTPUT_ROOT}"
echo "[LOG       ] ${LOG_ROOT}"
echo "[SUMMARY   ] ${SUMMARY_CSV}"
echo "[MANIFEST  ] ${MANIFEST_TSV}"
echo "[PYTHON    ] ${PYTHON_BIN}"
if [[ -n "${RUN_MODEL_TAGS}" ]]; then
  echo "[RUN ONLY  ] ${RUN_MODEL_TAGS}"
fi
if [[ -n "${SKIP_MODEL_TAGS}" ]]; then
  echo "[SKIP      ] ${SKIP_MODEL_TAGS}"
fi
echo "============================================================"

run_one_model() {
  local model_tag="$1"
  local model_path="$2"
  local run_name="${model_tag}-m${M_VALUE}-ppo-gradient-only-mia"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local stage31_dir="${run_dir}/3.1-Target-Data"
  local stage32_dir="${run_dir}/3.2-Candidate-Response-Generation"
  local stage33_dir="${run_dir}/3.3-LLM-Update-FullPPO"
  local stage34_dir="${run_dir}/3.4-PPO-Gradient-Only-MIA"
  local stage31_member="${stage31_dir}/member.jsonl"
  local stage31_nonmember="${stage31_dir}/nonmember.jsonl"
  local stage32_output_name="3.2_candidate_response_generation_m${M_VALUE}_${model_tag}.json"
  local stage32_output="${stage32_dir}/${stage32_output_name}"
  local policy_dir="${stage33_dir}/final_policy"
  local table_csv="${stage34_dir}/3.4_mia_table_metrics.csv"
  local summary_json="${stage34_dir}/3.4_membership_inference_summary.json"

  mkdir -p "${stage31_dir}" "${stage32_dir}" "${stage33_dir}" "${stage34_dir}"

  echo "============================================================"
  echo "[RUN] ${run_name}"
  echo "[MODEL] ${model_path}"
  echo "[DIR  ] ${run_dir}"
  echo "============================================================"

  if [[ "${OVERWRITE}" == "1" || ! -s "${stage31_member}" ]]; then
    cp "${MEMBER_FILE}" "${stage31_member}"
  fi
  if [[ "${OVERWRITE}" == "1" || ! -s "${stage31_nonmember}" ]]; then
    cp "${NONMEMBER_FILE}" "${stage31_nonmember}"
  fi

  echo "============================================================"
  echo "[3.2] Candidate response generation, M=${M_VALUE}"
  echo "============================================================"
  if [[ -s "${stage32_output}" && "${OVERWRITE}" != "1" ]]; then
    echo "[3.2] Found existing output, skip: ${stage32_output}"
  else
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" "${STAGE32_SCRIPT}" \
      --member_path "${stage31_member}" \
      --nonmember_path "${stage31_nonmember}" \
      --pretrained_llm_path "${model_path}" \
      --reward_base_model "${RM_BASE_MODEL}" \
      --reward_adapter_path "${RM_ADAPTER}" \
      --output_dir "${stage32_dir}" \
      --output_filename "${stage32_output_name}" \
      --member_size "${MEMBER_SIZE}" \
      --nonmember_size "${NONMEMBER_SIZE}" \
      --seed "${SEED}" \
      --m "${M_VALUE}" \
      --generation_batch_size "${GENERATION_BATCH_SIZE}" \
      --max_prompt_length "${MAX_PROMPT_LENGTH}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --reward_batch_size "${REWARD_BATCH_SIZE}" \
      --reward_max_length "${REWARD_MAX_LENGTH}"
  fi

  echo "============================================================"
  echo "[3.3] LLM update PPO"
  echo "============================================================"
  policy_dir="$(best_policy_dir "${stage33_dir}")"
  if [[ -n "${policy_dir}" && "${OVERWRITE}" != "1" ]]; then
    echo "[3.3] Found existing full policy, skip: ${policy_dir}"
  else
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" "${STAGE33_SCRIPT}" \
      --input_path "${stage32_output}" \
      --base_model "${model_path}" \
      --output_dir "${stage33_dir}" \
      --max_length "${MAX_PROMPT_LENGTH}" \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --learning_rate 1e-5 \
      --clip_eps "${CLIP_EPS}" \
      --logging_steps 10 \
      --save_steps "${STAGE3_SAVE_STEPS}" \
      --seed "${SEED}"

    policy_dir="$(best_policy_dir "${stage33_dir}")"
  fi

  if [[ -z "${policy_dir}" || ! -f "${policy_dir}/config.json" ]]; then
    echo "[ERROR] Cannot find full policy after Step 3 in: ${stage33_dir}"
    exit 1
  fi

  echo "============================================================"
  echo "[3.4] PPO-gradient-only membership inference"
  echo "============================================================"
  if [[ -s "${table_csv}" && "${OVERWRITE}" != "1" ]]; then
    echo "[3.4] Found existing table metrics, skip: ${table_csv}"
  else
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" "${STAGE34_SCRIPT}" \
      --input_path "${stage32_output}" \
      --reward_base_model "${RM_BASE_MODEL}" \
      --reward_adapter_path "${RM_ADAPTER}" \
      --policy_base_model "${model_path}" \
      --policy_model_path "${policy_dir}" \
      --output_dir "${stage34_dir}" \
      --policy_max_length "${POLICY_MAX_LENGTH}" \
      --lambda1 0.0 \
      --lambda2 "${LAMBDA2}" \
      --score_signal ppo_gradient_only \
      --target_fpr "${TARGET_FPR}" \
      --clip_eps "${CLIP_EPS}" \
      --delta "${DELTA}"
  fi

  if [[ ! -s "${table_csv}" ]]; then
    echo "[ERROR] Missing final table metrics: ${table_csv}"
    exit 1
  fi

  local metrics_line
  metrics_line="$(tail -n 1 "${table_csv}" | tr -d '\r')"
  printf "%s,%s\n" "${model_tag}" "${metrics_line}" >> "${SUMMARY_CSV}"
  printf "%s\t%s\t%s\t%s\t%s\n" "${model_tag}" "${table_csv}" "${summary_json}" "${policy_dir}" "${run_dir}" >> "${MANIFEST_TSV}"

  echo "[DONE] ${model_tag}"
  echo "[TABLE] ${table_csv}"
}

for i in "${!MODEL_TAGS[@]}"; do
  model_tag="${MODEL_TAGS[$i]}"
  if ! tag_selected "${model_tag}" "${RUN_MODEL_TAGS}"; then
    echo "[SKIP] ${model_tag}: not in RUN_MODEL_TAGS=${RUN_MODEL_TAGS}"
    continue
  fi
  if tag_skipped "${model_tag}" "${SKIP_MODEL_TAGS}"; then
    echo "[SKIP] ${model_tag}: in SKIP_MODEL_TAGS=${SKIP_MODEL_TAGS}"
    continue
  fi

  log_path="${LOG_ROOT}/${model_tag}.log"
  echo "[LAUNCH] ${model_tag}; log=${log_path}"
  set +e
  run_one_model "${model_tag}" "${MODEL_PATHS[$i]}" > "${log_path}" 2>&1
  rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo "[ERROR] ${model_tag} failed with exit code ${rc}. Log: ${log_path}"
    if [[ -f "${log_path}" ]]; then
      tail -n 120 "${log_path}" || true
    fi
    exit "${rc}"
  fi

  tail -n 8 "${log_path}" || true
done

echo "============================================================"
echo "[ALL DONE]"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Manifest TSV: ${MANIFEST_TSV}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log root   : ${LOG_ROOT}"
echo "============================================================"
