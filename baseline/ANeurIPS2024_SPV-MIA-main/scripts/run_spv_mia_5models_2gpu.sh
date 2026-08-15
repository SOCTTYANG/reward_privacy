#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
MI_DIR="${PROJECT_DIR}/membership inference"
SPV_DIR="${SCRIPT_DIR}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/vipuser/.conda/envs/safe-rlhf/bin/python" ]]; then
    PYTHON_BIN="/home/vipuser/.conda/envs/safe-rlhf/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
BASELINE_SAVE_ROOT="${BASELINE_SAVE_ROOT:-/mnt/model_data/yang_safe/baseline}"
export HF_HOME="${HF_HOME:-${BASELINE_SAVE_ROOT}/huggingface_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${BASELINE_SAVE_ROOT}/baseline_spv_mia_membership_steps_5models}"
LOG_ROOT="${LOG_ROOT:-${BASELINE_SAVE_ROOT}/logs/baseline_spv_mia_membership_steps_5models}"
RESUME_OUTPUT_ROOTS="${RESUME_OUTPUT_ROOTS:-/mnt/bai_data/yang-safe/baseline_spv_mia_membership_steps_5models}"
RUN_MODEL_TAGS="${RUN_MODEL_TAGS:-}"

MEMBER_FILE="${MEMBER_FILE:-/mnt/bai_data/yang-safe/data/mia_overfit/rm_member_train_512.jsonl}"
NONMEMBER_FILE="${NONMEMBER_FILE:-/mnt/bai_data/yang-safe/data/mia_overfit/rm_nonmember_test_512.jsonl}"
if [[ ! -f "${NONMEMBER_FILE}" ]]; then
  echo "[WARN] Cannot find NONMEMBER_FILE: ${NONMEMBER_FILE}"
  echo "[WARN] Falling back to PKU test set."
  NONMEMBER_FILE="/mnt/bai_data/yang-safe/data/pku30k_filtered/pku30k_test_cosine_le_085_xyz.jsonl"
fi

# Fixed reward model. Only the stage-2 pretrained LLM changes across runs.
RM_BASE_MODEL="${RM_BASE_MODEL:-/home/vipuser/Desktop/model/llama2-7b}"
RM_ADAPTER="${RM_ADAPTER:-/mnt/bai_data/yang-safe/output/rm_lora_overfit_member512_e80_lr1e4_ga1_r64_margin8}"
STAGE3_BASE_MODEL="${STAGE3_BASE_MODEL:-/home/vipuser/Desktop/model/llama2-7b}"

if [[ -z "${LLAMA2_7B:-}" ]]; then
  if [[ -d "/mnt/model_data/models/llama2-7b" ]]; then
    LLAMA2_7B="/mnt/model_data/models/llama2-7b"
  elif [[ -d "/mnt/model_data/models/llama2-7b-hf" ]]; then
    LLAMA2_7B="/mnt/model_data/models/llama2-7b-hf"
  else
    LLAMA2_7B="/home/vipuser/Desktop/model/llama2-7b"
  fi
fi
LLAMA2_13B="${LLAMA2_13B:-/mnt/model_data/models/llama2-13b-hf}"
LLAMA32_3B="${LLAMA32_3B:-/mnt/model_data/models/llama32-3b}"
QWEN3_8B="${QWEN3_8B:-/mnt/model_data/models/qwen3-8b}"
MISTRAL_7B="${MISTRAL_7B:-/mnt/model_data/models/mistral-7b-v0.1}"

STAGE2_MODEL_TAGS=("llama2-7b" "llama2-13b-hf" "llama32-3b" "qwen3-8b" "mistral-7b-v0.1")
STAGE2_MODEL_PATHS=("${LLAMA2_7B}" "${LLAMA2_13B}" "${LLAMA32_3B}" "${QWEN3_8B}" "${MISTRAL_7B}")

# Optional extra models: EXTRA_STAGE2_MODEL_SPECS="tag=/path,tag2=/path2"
EXTRA_STAGE2_MODEL_SPECS="${EXTRA_STAGE2_MODEL_SPECS:-}"
if [[ -n "${EXTRA_STAGE2_MODEL_SPECS}" ]]; then
  IFS=',' read -r -a EXTRA_SPECS <<< "${EXTRA_STAGE2_MODEL_SPECS}"
  for spec in "${EXTRA_SPECS[@]}"; do
    tag="${spec%%=*}"
    path="${spec#*=}"
    if [[ -z "${tag}" || -z "${path}" || "${tag}" == "${path}" ]]; then
      echo "[ERROR] Bad EXTRA_STAGE2_MODEL_SPECS item: ${spec}"
      exit 1
    fi
    STAGE2_MODEL_TAGS+=("${tag}")
    STAGE2_MODEL_PATHS+=("${path}")
  done
fi

MEMBER_SIZE="${MEMBER_SIZE:-512}"
NONMEMBER_SIZE="${NONMEMBER_SIZE:-512}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_CANDIDATES="${NUM_CANDIDATES:-3}"
STAGE3_SAVE_STEPS="${STAGE3_SAVE_STEPS:-100000000}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MARGIN="${MARGIN:-8}"

FPR="${FPR:-5.0}"
FPRS="${FPRS:-1.0,5.0}"
SPV_SAMPLE_NUMBER="${SPV_SAMPLE_NUMBER:-10}"
SPV_MASK_MODEL="${SPV_MASK_MODEL:-t5-base}"
SPV_PERTURBATION_SOURCE="${SPV_PERTURBATION_SOURCE:-candidates}"
SPV_SCORE_MODE="${SPV_SCORE_MODE:-prob}"
SPV_TORCH_DTYPE="${SPV_TORCH_DTYPE:-bfloat16}"
SPV_LOCAL_FILES_ONLY="${SPV_LOCAL_FILES_ONLY:-0}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

STAGE32_SCRIPT="${MI_DIR}/3_2_candidate_response_generation.py"
STAGE33_SCRIPT="${MI_DIR}/3_3_llm_update_ppo_full.py"
SPV_ATTACK_SCRIPT="${SPV_DIR}/spv_mia_safe_rlhf.py"

require_file() {
  local path="$1"
  local name="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Missing ${name}: ${path}"
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local name="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing ${name}: ${path}"
    echo "[HINT] Override model paths with LLAMA2_7B, LLAMA2_13B, LLAMA32_3B, QWEN3_8B, MISTRAL_7B, or EXTRA_STAGE2_MODEL_SPECS."
    exit 1
  fi
}

model_in_csv() {
  local needle="$1"
  local csv="$2"
  local item
  local -a items

  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

should_run_model() {
  local model_tag="$1"
  [[ -z "${RUN_MODEL_TAGS}" ]] || model_in_csv "${model_tag}" "${RUN_MODEL_TAGS}"
}

latest_checkpoint_lora_dir() {
  local stage33_dir="$1"
  if [[ ! -d "${stage33_dir}" ]]; then
    return 0
  fi

  find "${stage33_dir}" -maxdepth 2 -path "*/checkpoint-*/adapter_config.json" -print 2>/dev/null \
    | sed 's#/adapter_config.json$##' \
    | sort -V \
    | tail -n 1
}

best_policy_lora_dir() {
  local stage33_dir="$1"
  if [[ -f "${stage33_dir}/final_policy_lora/adapter_config.json" ]]; then
    echo "${stage33_dir}/final_policy_lora"
    return 0
  fi

  latest_checkpoint_lora_dir "${stage33_dir}"
}

copy_file_if_missing() {
  local src="$1"
  local dst="$2"
  local label="$3"

  if [[ -s "${dst}" || ! -s "${src}" ]]; then
    return 1
  fi

  mkdir -p "$(dirname "${dst}")"
  cp -a "${src}" "${dst}"
  echo "[RESUME] Copied ${label}: ${src} -> ${dst}"
  return 0
}

copy_dir_if_missing() {
  local src="$1"
  local dst="$2"
  local label="$3"

  if [[ -e "${dst}" || ! -d "${src}" ]]; then
    return 1
  fi

  mkdir -p "$(dirname "${dst}")"
  cp -a "${src}" "${dst}"
  echo "[RESUME] Copied ${label}: ${src} -> ${dst}"
  return 0
}

print_failed_log_tail() {
  local model_tag="$1"
  local rc="$2"
  local log_path="${LOG_ROOT}/${model_tag}.log"

  echo "[ERROR] ${model_tag} failed with exit code ${rc}. Log: ${log_path}"
  if [[ -f "${log_path}" ]]; then
    echo "==================== ${model_tag} log tail ===================="
    tail -n 120 "${log_path}" || true
    echo "================== end ${model_tag} log tail =================="
  fi
}

require_file "${STAGE32_SCRIPT}" "membership inference 3.2 script"
require_file "${STAGE33_SCRIPT}" "membership inference 3.3 script"
require_file "${SPV_ATTACK_SCRIPT}" "SPV-MIA Safe-RLHF attack script"
require_file "${MEMBER_FILE}" "MEMBER_FILE"
require_file "${NONMEMBER_FILE}" "NONMEMBER_FILE"
require_dir "${RM_BASE_MODEL}" "RM_BASE_MODEL"
require_dir "${RM_ADAPTER}" "RM_ADAPTER"
require_dir "${STAGE3_BASE_MODEL}" "STAGE3_BASE_MODEL"

for i in "${!STAGE2_MODEL_TAGS[@]}"; do
  require_dir "${STAGE2_MODEL_PATHS[$i]}" "${STAGE2_MODEL_TAGS[$i]}"
done

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

if [[ "${SKIP_PYTHON_PREFLIGHT:-0}" != "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import importlib
import sys

missing = []
for module in ("torch", "transformers", "peft", "sklearn", "numpy", "tqdm"):
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append(f"{module}: {exc}")

if missing:
    print("[ERROR] PYTHON_BIN is missing required packages:", sys.executable)
    for item in missing:
        print("  -", item)
    sys.exit(1)

print("[INFO] Python preflight OK:", sys.executable)
PY
fi

run_one_model() {
  local gpu_id="$1"
  local model_tag="$2"
  local stage2_pretrained_llm_path="$3"

  local run_name="${model_tag}-stage2llm-r${LORA_R}-margin${MARGIN}-spv-baseline"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local stage31_dir="${run_dir}/3.1-Target-Data"
  local stage32_dir="${run_dir}/3.2-Candidate-Response-Generation"
  local stage33_dir="${run_dir}/3.3-LLM-Update-FullPPO"
  local stage34_spv_dir="${run_dir}/3.4-SPV-MIA-Baseline"

  local stage31_member="${stage31_dir}/member.jsonl"
  local stage31_nonmember="${stage31_dir}/nonmember.jsonl"
  local stage32_output_name="3.2_candidate_response_generation.json"
  local stage32_output="${stage32_dir}/${stage32_output_name}"
  local policy_lora="${stage33_dir}/final_policy_lora"
  local -a resume_roots=()
  local -a resume_run_dirs=()
  local resume_root
  local resume_run_dir

  IFS=',' read -r -a resume_roots <<< "${RESUME_OUTPUT_ROOTS}"
  for resume_root in "${resume_roots[@]}"; do
    if [[ -z "${resume_root}" ]]; then
      continue
    fi
    resume_run_dir="${resume_root}/${run_name}"
    if [[ "${resume_run_dir}" != "${run_dir}" && -d "${resume_run_dir}" ]]; then
      resume_run_dirs+=("${resume_run_dir}")
    fi
  done

  mkdir -p "${stage31_dir}" "${stage32_dir}" "${stage33_dir}" "${stage34_spv_dir}"

  echo "============================================================"
  echo "[RUN] ${run_name} on visible GPU ${gpu_id}"
  echo "[STAGE2_LLM ] ${stage2_pretrained_llm_path}"
  echo "[STAGE3_BASE] ${STAGE3_BASE_MODEL}"
  echo "[REWARD_BASE] ${RM_BASE_MODEL}"
  echo "[REWARD_LORA] ${RM_ADAPTER}"
  echo "[RUN_DIR    ] ${run_dir}"
  if (( ${#resume_run_dirs[@]} > 0 )); then
    echo "[RESUME_FROM] ${resume_run_dirs[*]}"
  fi
  echo "============================================================"

  echo "============================================================"
  echo "[3.1] Prepare target data"
  echo "============================================================"
  if [[ ! -s "${stage31_member}" ]]; then
    cp "${MEMBER_FILE}" "${stage31_member}"
  fi
  if [[ ! -s "${stage31_nonmember}" ]]; then
    cp "${NONMEMBER_FILE}" "${stage31_nonmember}"
  fi
  echo "[3.1] member    -> ${stage31_member}"
  echo "[3.1] nonmember -> ${stage31_nonmember}"

  echo "============================================================"
  echo "[3.2] Candidate response generation"
  echo "============================================================"
  if [[ ! -s "${stage32_output}" ]]; then
    for resume_run_dir in "${resume_run_dirs[@]}"; do
      if copy_file_if_missing \
        "${resume_run_dir}/3.2-Candidate-Response-Generation/${stage32_output_name}" \
        "${stage32_output}" \
        "3.2 output"; then
        break
      fi
    done
  fi

  if [[ -s "${stage32_output}" ]]; then
    echo "[3.2] Found existing output, skip generation: ${stage32_output}"
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${STAGE32_SCRIPT}" \
      --member_path "${stage31_member}" \
      --nonmember_path "${stage31_nonmember}" \
      --pretrained_llm_path "${stage2_pretrained_llm_path}" \
      --reward_base_model "${RM_BASE_MODEL}" \
      --reward_adapter_path "${RM_ADAPTER}" \
      --output_dir "${stage32_dir}" \
      --output_filename "${stage32_output_name}" \
      --member_size "${MEMBER_SIZE}" \
      --nonmember_size "${NONMEMBER_SIZE}" \
      --seed "${SEED}" \
      --m "${NUM_CANDIDATES}" \
      --generation_batch_size 1 \
      --max_prompt_length "${MAX_LENGTH}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --temperature 1.0 \
      --top_p 0.9 \
      --reward_batch_size 1 \
      --reward_max_length "${MAX_LENGTH}"
  fi
  echo "[3.2] output -> ${stage32_output}"

  echo "============================================================"
  echo "[3.3] LLM update Full PPO"
  echo "============================================================"
  local reusable_policy_lora
  reusable_policy_lora="$(best_policy_lora_dir "${stage33_dir}")"

  if [[ -z "${reusable_policy_lora}" ]]; then
    for resume_run_dir in "${resume_run_dirs[@]}"; do
      local source_policy_lora
      source_policy_lora="$(best_policy_lora_dir "${resume_run_dir}/3.3-LLM-Update-FullPPO")"
      if [[ -n "${source_policy_lora}" ]]; then
        local copied_policy_lora
        if [[ "$(basename "${source_policy_lora}")" == "final_policy_lora" ]]; then
          copied_policy_lora="${stage33_dir}/final_policy_lora"
        else
          copied_policy_lora="${stage33_dir}/$(basename "${source_policy_lora}")"
        fi

        if copy_dir_if_missing "${source_policy_lora}" "${copied_policy_lora}" "3.3 policy LoRA"; then
          reusable_policy_lora="${copied_policy_lora}"
        elif [[ -f "${copied_policy_lora}/adapter_config.json" ]]; then
          reusable_policy_lora="${copied_policy_lora}"
        fi
        break
      fi
    done
  fi

  if [[ -n "${reusable_policy_lora}" ]]; then
    policy_lora="${reusable_policy_lora}"
    echo "[3.3] Found reusable policy LoRA, skip PPO update: ${policy_lora}"
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${STAGE33_SCRIPT}" \
      --input_path "${stage32_output}" \
      --base_model "${STAGE3_BASE_MODEL}" \
      --output_dir "${stage33_dir}" \
      --max_length "${MAX_LENGTH}" \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --learning_rate 1e-5 \
      --weight_decay 0.0 \
      --clip_eps 0.2 \
      --max_grad_norm 1.0 \
      --logging_steps 10 \
      --save_steps "${STAGE3_SAVE_STEPS}" \
      --seed "${SEED}" \
      --normalize_advantage \
      --lora_r "${LORA_R}" \
      --lora_alpha "${LORA_ALPHA}" \
      --lora_dropout "${LORA_DROPOUT}"
  fi

  if [[ ! -f "${policy_lora}/adapter_config.json" ]]; then
    local found_lora
    found_lora="$(best_policy_lora_dir "${stage33_dir}")"
    if [[ -n "${found_lora}" ]]; then
      policy_lora="${found_lora}"
    else
      echo "[ERROR] Cannot find policy LoRA adapter after 3.3 in: ${stage33_dir}"
      exit 1
    fi
  fi
  echo "[3.3] policy LoRA -> ${policy_lora}"

  echo "============================================================"
  echo "[3.4-SPV] ANeurIPS2024 SPV-MIA baseline"
  echo "============================================================"
  if [[ ! -s "${stage34_spv_dir}/spv_mia_summary.json" ]]; then
    for resume_run_dir in "${resume_run_dirs[@]}"; do
      local resume_stage34_dir="${resume_run_dir}/3.4-SPV-MIA-Baseline"
      if [[ -s "${resume_stage34_dir}/spv_mia_summary.json" ]]; then
        cp -a "${resume_stage34_dir}/." "${stage34_spv_dir}/"
        echo "[RESUME] Copied 3.4 SPV results: ${resume_stage34_dir} -> ${stage34_spv_dir}"
        break
      fi
    done
  fi

  if [[ -s "${stage34_spv_dir}/spv_mia_summary.json" ]]; then
    echo "[3.4-SPV] Found existing summary, skip SPV attack: ${stage34_spv_dir}/spv_mia_summary.json"
  else
    local local_files_flag=()
    if [[ "${SPV_LOCAL_FILES_ONLY}" == "1" ]]; then
      local_files_flag=(--local_files_only)
    fi

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${SPV_ATTACK_SCRIPT}" \
      --step2_path "${stage32_output}" \
      --target_base_model "${STAGE3_BASE_MODEL}" \
      --target_adapter_path "${policy_lora}" \
      --reference_base_model "${STAGE3_BASE_MODEL}" \
      --output_dir "${stage34_spv_dir}" \
      --model_tag "${model_tag}" \
      --response_field y_plus \
      --prompt_template alpaca \
      --member_size "${MEMBER_SIZE}" \
      --nonmember_size "${NONMEMBER_SIZE}" \
      --member_sample_mode first \
      --nonmember_sample_mode first \
      --seed "${SEED}" \
      --perturbation_source "${SPV_PERTURBATION_SOURCE}" \
      --mask_filling_model_name "${SPV_MASK_MODEL}" \
      --sample_number "${SPV_SAMPLE_NUMBER}" \
      --score_mode "${SPV_SCORE_MODE}" \
      --target_fpr "${FPR}" \
      --target_fprs "${FPRS}" \
      --max_length "${MAX_LENGTH}" \
      --torch_dtype "${SPV_TORCH_DTYPE}" \
      "${local_files_flag[@]}"
  fi

  echo "============================================================"
  echo "[DONE] ${run_name}"
  echo "============================================================"
}

IFS=',' read -r -a GPU_IDS <<< "${CUDA_DEVICES}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "[ERROR] CUDA_DEVICES cannot be empty"
  exit 1
fi
if (( MAX_PARALLEL > ${#GPU_IDS[@]} )); then
  MAX_PARALLEL="${#GPU_IDS[@]}"
fi

echo "============================================================"
echo "[BATCH] SPV-MIA baseline strictly following membership inference steps"
echo "[FIXED] reward=${RM_BASE_MODEL} + ${RM_ADAPTER}"
echo "[FIXED] stage3_base=${STAGE3_BASE_MODEL}"
echo "[VARY ] only stage2 --pretrained_llm_path"
echo "[GPUs ] ${CUDA_DEVICES}; MAX_PARALLEL=${MAX_PARALLEL}"
echo "[PYTHON] ${PYTHON_BIN}"
echo "[SPV  ] perturbation=${SPV_PERTURBATION_SOURCE}; sample_number=${SPV_SAMPLE_NUMBER}; score_mode=${SPV_SCORE_MODE}"
echo "[FPRS ] ${FPRS}"
if [[ -n "${RUN_MODEL_TAGS}" ]]; then
  echo "[RUN  ] only model tags: ${RUN_MODEL_TAGS}"
fi
echo "[OUT  ] ${OUTPUT_ROOT}"
echo "[LOG  ] ${LOG_ROOT}"
echo "[RESUME_ROOTS] ${RESUME_OUTPUT_ROOTS}"
echo "============================================================"

SCHEDULE_TAGS=()
SCHEDULE_PATHS=()

for i in "${!STAGE2_MODEL_TAGS[@]}"; do
  model_tag="${STAGE2_MODEL_TAGS[$i]}"
  if ! should_run_model "${model_tag}"; then
    echo "[SKIP] ${model_tag} not in RUN_MODEL_TAGS=${RUN_MODEL_TAGS}"
    continue
  fi

  SCHEDULE_TAGS+=("${model_tag}")
  SCHEDULE_PATHS+=("${STAGE2_MODEL_PATHS[$i]}")
done

if (( ${#SCHEDULE_TAGS[@]} == 0 )); then
  echo "[WARN] No models selected."
fi

FREE_GPUS=()
for ((i = 0; i < MAX_PARALLEL; i++)); do
  FREE_GPUS+=("${GPU_IDS[$i]}")
done

ACTIVE_PIDS=()
ACTIVE_NAMES=()
ACTIVE_GPUS=()
NEXT_MODEL_INDEX=0
SCHEDULER_POLL_SECONDS="${SCHEDULER_POLL_SECONDS:-5}"

compact_active_jobs() {
  ACTIVE_PIDS=("${ACTIVE_PIDS[@]}")
  ACTIVE_NAMES=("${ACTIVE_NAMES[@]}")
  ACTIVE_GPUS=("${ACTIVE_GPUS[@]}")
}

pid_in_running_jobs() {
  local needle="$1"
  local pid

  for pid in ${RUNNING_PIDS_TEXT:-}; do
    if [[ "${pid}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

launch_next_model() {
  local gpu_id="${FREE_GPUS[0]}"
  local model_tag="${SCHEDULE_TAGS[$NEXT_MODEL_INDEX]}"
  local model_path="${SCHEDULE_PATHS[$NEXT_MODEL_INDEX]}"
  local log_path="${LOG_ROOT}/${model_tag}.log"

  FREE_GPUS=("${FREE_GPUS[@]:1}")
  echo "[LAUNCH] ${model_tag} on visible GPU ${gpu_id}; log=${log_path}"
  run_one_model "${gpu_id}" "${model_tag}" "${model_path}" > "${log_path}" 2>&1 &
  ACTIVE_PIDS+=("$!")
  ACTIVE_NAMES+=("${model_tag}")
  ACTIVE_GPUS+=("${gpu_id}")
  NEXT_MODEL_INDEX=$((NEXT_MODEL_INDEX + 1))
}

while (( NEXT_MODEL_INDEX < ${#SCHEDULE_TAGS[@]} || ${#ACTIVE_PIDS[@]} > 0 )); do
  while (( NEXT_MODEL_INDEX < ${#SCHEDULE_TAGS[@]} && ${#ACTIVE_PIDS[@]} < MAX_PARALLEL && ${#FREE_GPUS[@]} > 0 )); do
    launch_next_model
  done

  if (( ${#ACTIVE_PIDS[@]} == 0 )); then
    continue
  fi

  RUNNING_PIDS_TEXT="$(jobs -r -p || true)"
  finished_index=-1
  for i in "${!ACTIVE_PIDS[@]}"; do
    if ! pid_in_running_jobs "${ACTIVE_PIDS[$i]}"; then
      finished_index="${i}"
      break
    fi
  done

  if (( finished_index >= 0 )); then
    finished_pid="${ACTIVE_PIDS[$finished_index]}"
    finished_name="${ACTIVE_NAMES[$finished_index]}"
    finished_gpu="${ACTIVE_GPUS[$finished_index]}"

    set +e
    wait "${finished_pid}"
    rc=$?
    set -e

    unset "ACTIVE_PIDS[$finished_index]"
    unset "ACTIVE_NAMES[$finished_index]"
    unset "ACTIVE_GPUS[$finished_index]"
    compact_active_jobs

    if [[ "${rc}" -ne 0 ]]; then
      print_failed_log_tail "${finished_name}" "${rc}"
      exit 1
    fi

    echo "[FINISH] ${finished_name} completed on visible GPU ${finished_gpu}"
    FREE_GPUS+=("${finished_gpu}")
    continue
  fi

  sleep "${SCHEDULER_POLL_SECONDS}"
done

REFRESH_SCRIPT="${SCRIPT_DIR}/refresh_spv_mia_low_fpr_summary.py"
if [[ -f "${REFRESH_SCRIPT}" ]]; then
  "${PYTHON_BIN}" "${REFRESH_SCRIPT}" --output_root "${OUTPUT_ROOT}" --target_fprs "${FPRS}"
else
  echo "[WARN] Cannot find low-FPR refresh script: ${REFRESH_SCRIPT}"
fi

SUMMARY_CSV="${OUTPUT_ROOT}/spv_mia_membership_steps_summary.csv"
"${PYTHON_BIN}" - "${SUMMARY_CSV}" "${OUTPUT_ROOT}" <<'PY'
import csv
import glob
import json
import os
import sys

summary_csv = sys.argv[1]
output_root = sys.argv[2]
summary_paths = sorted(glob.glob(os.path.join(output_root, "*", "3.4-SPV-MIA-Baseline", "spv_mia_summary.json")))

rows = []
for path in summary_paths:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    config = obj.get("config", {})
    roc = obj.get("roc_based_metrics", {})
    best = obj.get("best_threshold_metrics", {})
    rows.append({
        "model_tag": config.get("model_tag"),
        "auc": roc.get("auc"),
        "raw_auc": roc.get("raw_auc"),
        "score_flipped": roc.get("score_flipped"),
        "target_fpr": roc.get("target_fpr"),
        "tpr_at_target_fpr": roc.get("tpr_at_target_fpr"),
        "actual_fpr_at_target_fpr": roc.get("actual_fpr_at_target_fpr"),
        "tpr_at_1pct_fpr": roc.get("tpr_at_1pct_fpr"),
        "acc_at_1pct_fpr": roc.get("acc_at_1pct_fpr"),
        "actual_fpr_at_1pct_fpr": roc.get("actual_fpr_at_1pct_fpr"),
        "tpr_at_5pct_fpr": roc.get("tpr_at_5pct_fpr"),
        "acc_at_5pct_fpr": roc.get("acc_at_5pct_fpr"),
        "actual_fpr_at_5pct_fpr": roc.get("actual_fpr_at_5pct_fpr"),
        "best_accuracy": best.get("best_accuracy", {}).get("accuracy"),
        "best_balanced_accuracy": best.get("best_balanced_accuracy", {}).get("balanced_accuracy"),
        "best_f1": best.get("best_f1", {}).get("f1"),
        "summary_json": path,
    })

if rows:
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Summary CSV: {summary_csv}")
else:
    print(f"[WARN] No SPV summary files found under: {output_root}")
PY

echo "============================================================"
echo "[ALL DONE]"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log root   : ${LOG_ROOT}"
echo "Summary    : ${SUMMARY_CSV}"
echo "============================================================"
