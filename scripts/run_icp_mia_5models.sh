#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MI_DIR="${PROJECT_DIR}/membership inference"
ICP_DIR="${PROJECT_DIR}/baseline/ICP-MIA-main"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-/mnt/bai_data/yang-safe/huggingface_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/bai_data/yang-safe/baseline_icp_mia_membership_steps_5models}"
LOG_ROOT="${LOG_ROOT:-/mnt/bai_data/yang-safe/logs/baseline_icp_mia_membership_steps_5models}"
REMAINING_OUTPUT_ROOT="${REMAINING_OUTPUT_ROOT:-/mnt/bai_data/yang-safe/membership_inference/baseline_icp_mia_membership_steps_5models}"
REMAINING_LOG_ROOT="${REMAINING_LOG_ROOT:-/mnt/bai_data/yang-safe/membership_inference/logs/baseline_icp_mia_membership_steps_5models}"
REMAINING_MODEL_TAGS="${REMAINING_MODEL_TAGS:-llama32-3b,qwen3-8b,mistral-7b}"
RUN_MODEL_TAGS="${RUN_MODEL_TAGS:-}"

MEMBER_FILE="${MEMBER_FILE:-/mnt/bai_data/yang-safe/data/mia_overfit/rm_member_train_512.jsonl}"
NONMEMBER_FILE="${NONMEMBER_FILE:-/mnt/bai_data/yang-safe/data/mia_overfit/rm_nonmember_test_512.jsonl}"
if [[ ! -f "${NONMEMBER_FILE}" ]]; then
  echo "[WARN] Cannot find NONMEMBER_FILE: ${NONMEMBER_FILE}"
  echo "[WARN] Falling back to PKU test set."
  NONMEMBER_FILE="/mnt/bai_data/yang-safe/data/pku30k_filtered/pku30k_test_cosine_le_085_xyz.jsonl"
fi

RM_BASE_MODEL="${RM_BASE_MODEL:-/home/vipuser/Desktop/model/llama2-7b}"
RM_ADAPTER="${RM_ADAPTER:-/mnt/bai_data/yang-safe/output/rm_lora_overfit_member512_e80_lr1e4_ga1_r64_margin8}"
STAGE3_BASE_MODEL="${STAGE3_BASE_MODEL:-/home/vipuser/Desktop/model/llama2-7b}"

LLAMA2_13B="${LLAMA2_13B:-/mnt/model_data/models/llama2-13b-hf}"
LLAMA32_3B="${LLAMA32_3B:-/mnt/model_data/models/llama32-3b}"
QWEN3_8B="${QWEN3_8B:-/mnt/model_data/models/qwen3-8b}"
MISTRAL_7B="${MISTRAL_7B:-/mnt/model_data/models/mistral-7b-v0.1}"
if [[ -z "${LLAMA2_7B:-}" ]]; then
  if [[ -d "/mnt/model_data/models/llama2-7b" ]]; then
    LLAMA2_7B="/mnt/model_data/models/llama2-7b"
  else
    LLAMA2_7B="/home/vipuser/Desktop/model/llama2-7b"
  fi
fi

STAGE2_MODEL_TAGS=("llama2-7b" "llama2-13b" "llama32-3b" "qwen3-8b" "mistral-7b")
STAGE2_MODEL_PATHS=("${LLAMA2_7B}" "${LLAMA2_13B}" "${LLAMA32_3B}" "${QWEN3_8B}" "${MISTRAL_7B}")

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
ICP_MODE="${ICP_MODE:-sp}"
ICP_RESPONSE_FIELD="${ICP_RESPONSE_FIELD:-y_plus}"
ICP_PROMPT_TEMPLATE="${ICP_PROMPT_TEMPLATE:-alpaca}"
ICP_MAX_PROMPT_TOKENS="${ICP_MAX_PROMPT_TOKENS:-2048}"
ICP_TORCH_DTYPE="${ICP_TORCH_DTYPE:-float16}"
ICP_PREFIX_POOL_SOURCE="${ICP_PREFIX_POOL_SOURCE:-${ICP_DIR}/data/iCliniq/iCliniq.json}"
ICP_EMBEDDING_MODEL="${ICP_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
ICP_MAX_PREFIX_CANDIDATES="${ICP_MAX_PREFIX_CANDIDATES:-10}"
ICP_AGGREGATION_STRATEGY="${ICP_AGGREGATION_STRATEGY:-max}"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

STAGE32_SCRIPT="${MI_DIR}/3_2_candidate_response_generation.py"
STAGE33_SCRIPT="${MI_DIR}/3_3_llm_update_ppo_full.py"
ICP_ATTACK_SCRIPT="${ICP_DIR}/icp_mia_attack.py"
ICP_CONVERT_SCRIPT="${ICP_DIR}/prepare_safe_rlhf_data.py"

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
    echo "[HINT] Override model paths with LLAMA2_7B, LLAMA2_13B, LLAMA32_3B, QWEN3_8B, or MISTRAL_7B."
    exit 1
  fi
}

require_file "${STAGE32_SCRIPT}" "membership inference 3.2 script"
require_file "${STAGE33_SCRIPT}" "membership inference 3.3 script"
require_file "${ICP_ATTACK_SCRIPT}" "ICP-MIA attack script"
require_file "${ICP_CONVERT_SCRIPT}" "ICP-MIA converter"
require_file "${MEMBER_FILE}" "MEMBER_FILE"
require_file "${NONMEMBER_FILE}" "NONMEMBER_FILE"
require_dir "${RM_BASE_MODEL}" "RM_BASE_MODEL"
require_dir "${RM_ADAPTER}" "RM_ADAPTER"
require_dir "${STAGE3_BASE_MODEL}" "STAGE3_BASE_MODEL"

for i in "${!STAGE2_MODEL_TAGS[@]}"; do
  require_dir "${STAGE2_MODEL_PATHS[$i]}" "${STAGE2_MODEL_TAGS[$i]}"
done

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

model_output_root() {
  local model_tag="$1"
  if model_in_csv "${model_tag}" "${REMAINING_MODEL_TAGS}"; then
    echo "${REMAINING_OUTPUT_ROOT}"
  else
    echo "${OUTPUT_ROOT}"
  fi
}

model_log_root() {
  local model_tag="$1"
  if model_in_csv "${model_tag}" "${REMAINING_MODEL_TAGS}"; then
    echo "${REMAINING_LOG_ROOT}"
  else
    echo "${LOG_ROOT}"
  fi
}

should_run_model() {
  local model_tag="$1"
  [[ -z "${RUN_MODEL_TAGS}" ]] || model_in_csv "${model_tag}" "${RUN_MODEL_TAGS}"
}

move_tree_non_clobber() {
  local src="$1"
  local dst="$2"
  local item
  local base
  local target

  mkdir -p "${dst}"
  shopt -s dotglob nullglob
  for item in "${src}"/*; do
    base="$(basename "${item}")"
    target="${dst}/${base}"
    if [[ -d "${item}" ]] && [[ -d "${target}" ]]; then
      move_tree_non_clobber "${item}" "${target}"
      rmdir "${item}" 2>/dev/null || true
    elif [[ -e "${target}" ]]; then
      echo "[MIGRATE] keep destination, leave source item in old dir: ${item}"
    else
      mv "${item}" "${target}"
    fi
  done
  shopt -u dotglob nullglob
}

migrate_existing_run_dir() {
  local model_tag="$1"
  local run_name="$2"
  local output_root_for_model="$3"
  local old_run_dir="${OUTPUT_ROOT}/${run_name}"
  local new_run_dir="${output_root_for_model}/${run_name}"

  if [[ "${old_run_dir}" == "${new_run_dir}" ]] || [[ ! -d "${old_run_dir}" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "${new_run_dir}")"
  if [[ -d "${new_run_dir}" ]] && [[ -z "$(find "${new_run_dir}" -mindepth 1 -print -quit)" ]]; then
    rmdir "${new_run_dir}"
  fi

  if [[ ! -e "${new_run_dir}" ]]; then
    echo "[MIGRATE] ${model_tag} existing generated files"
    echo "[MIGRATE] from: ${old_run_dir}"
    echo "[MIGRATE] to  : ${new_run_dir}"
    mv "${old_run_dir}" "${new_run_dir}"
    return 0
  fi

  echo "[MIGRATE] ${model_tag} destination already exists, merging non-conflicting items."
  echo "[MIGRATE] from: ${old_run_dir}"
  echo "[MIGRATE] to  : ${new_run_dir}"
  move_tree_non_clobber "${old_run_dir}" "${new_run_dir}"
  rmdir "${old_run_dir}" 2>/dev/null || true
}

migrate_existing_log() {
  local model_tag="$1"
  local old_log="${LOG_ROOT}/${model_tag}.log"
  local new_log_root
  new_log_root="$(model_log_root "${model_tag}")"

  if [[ "${new_log_root}" == "${LOG_ROOT}" ]] || [[ ! -f "${old_log}" ]]; then
    return 0
  fi

  mkdir -p "${new_log_root}"
  local archived_log="${new_log_root}/${model_tag}.previous.log"
  if [[ -e "${archived_log}" ]]; then
    archived_log="${new_log_root}/${model_tag}.previous.$(date +%Y%m%d%H%M%S).log"
  fi
  echo "[MIGRATE] previous ${model_tag} log -> ${archived_log}"
  mv "${old_log}" "${archived_log}"
}

mkdir -p "${REMAINING_OUTPUT_ROOT}" "${REMAINING_LOG_ROOT}"

write_icp_config() {
  local mode="$1"
  local model_tag="$2"
  local policy_lora="$3"
  local data_dir="$4"
  local result_dir="$5"
  local config_path="$6"
  local train_json="${data_dir}/icp_${model_tag}_train.json"
  local test_json="${data_dir}/icp_${model_tag}_test.json"
  local perturbation_json="${data_dir}/icp_${model_tag}_perturbations.json"

  mkdir -p "${result_dir}"

  if [[ "${mode}" == "sp" ]]; then
    cat > "${config_path}" <<YAML
model:
  target_model_path: "${policy_lora}"
  reference_model_path: null
  device: "cuda:0"
  max_prompt_tokens: ${ICP_MAX_PROMPT_TOKENS}
  torch_dtype: "${ICP_TORCH_DTYPE}"

data:
  train_data_path: "${train_json}"
  test_data_path: "${test_json}"
  data_format: "instruction"
  prompt_template: "${ICP_PROMPT_TEMPLATE}"
  test_size: ${MEMBER_SIZE}
  random_seed: ${SEED}
  enable_shuffle: false
  member_detection_strategy: "auto"
  force_balanced_split: false

similarity_based_icp:
  enabled: false

self_perturbation_icp:
  enabled: true
  perturbation_file_path: "${perturbation_json}"
  perturbation_key: "mask_perturbations"
  top_k: ${NUM_CANDIDATES}
  aggregation_strategy: "mean"

experiment:
  output_dir: "${result_dir}"
  cache_dir: "${result_dir}/cache"
  experiment_name: "icp_sp_${model_tag}_membership_steps_seed${SEED}"
  save_detailed_results: true
  fpr_thresholds: [0.01, 0.05, 0.1]
YAML
  elif [[ "${mode}" == "ref" ]]; then
    cat > "${config_path}" <<YAML
model:
  target_model_path: "${policy_lora}"
  reference_model_path: null
  device: "cuda:0"
  max_prompt_tokens: ${ICP_MAX_PROMPT_TOKENS}
  torch_dtype: "${ICP_TORCH_DTYPE}"

data:
  train_data_path: "${train_json}"
  test_data_path: "${test_json}"
  data_format: "instruction"
  prompt_template: "${ICP_PROMPT_TEMPLATE}"
  test_size: ${MEMBER_SIZE}
  random_seed: ${SEED}
  enable_shuffle: false

similarity_based_icp:
  enabled: true
  prefix_pool_source: "${ICP_PREFIX_POOL_SOURCE}"
  top_k: 1
  max_prefix_candidates: ${ICP_MAX_PREFIX_CANDIDATES}
  aggregation_strategy: "${ICP_AGGREGATION_STRATEGY}"
  embedding_model: "${ICP_EMBEDDING_MODEL}"

self_perturbation_icp:
  enabled: false

experiment:
  output_dir: "${result_dir}"
  cache_dir: "${result_dir}/cache"
  experiment_name: "icp_ref_${model_tag}_membership_steps_seed${SEED}"
  save_detailed_results: true
  fpr_thresholds: [0.01, 0.05, 0.1]
YAML
  else
    echo "[ERROR] Unknown ICP mode: ${mode}"
    exit 1
  fi
}

run_icp_mode() {
  local gpu_id="$1"
  local mode="$2"
  local model_tag="$3"
  local policy_lora="$4"
  local data_dir="$5"
  local icp_dir="$6"
  local config_dir="${icp_dir}/configs"
  local result_dir="${icp_dir}/results-${mode}"
  local config_path="${config_dir}/config_icp_${mode}_${model_tag}.yaml"

  mkdir -p "${config_dir}"
  write_icp_config "${mode}" "${model_tag}" "${policy_lora}" "${data_dir}" "${result_dir}" "${config_path}"

  echo "============================================================"
  echo "[3.4-ICP] ${model_tag} mode=${mode}"
  echo "[3.4-ICP] policy_lora: ${policy_lora}"
  echo "[3.4-ICP] config     : ${config_path}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES="${gpu_id}" python "${ICP_ATTACK_SCRIPT}" --config "${config_path}"
}

run_one_model() {
  local gpu_id="$1"
  local model_tag="$2"
  local stage2_pretrained_llm_path="$3"

  local run_name="${model_tag}-stage2llm-r${LORA_R}-margin${MARGIN}-icp-baseline"
  local output_root_for_model
  output_root_for_model="$(model_output_root "${model_tag}")"
  local run_dir="${output_root_for_model}/${run_name}"

  migrate_existing_run_dir "${model_tag}" "${run_name}" "${output_root_for_model}"

  local stage31_dir="${run_dir}/3.1-Target-Data"
  local stage32_dir="${run_dir}/3.2-Candidate-Response-Generation"
  local stage33_dir="${run_dir}/3.3-LLM-Update-FullPPO"
  local stage34_icp_dir="${run_dir}/3.4-ICP-MIA-Baseline"
  local icp_data_dir="${stage34_icp_dir}/data"

  local stage31_member="${stage31_dir}/member.jsonl"
  local stage31_nonmember="${stage31_dir}/nonmember.jsonl"
  local stage32_output_name="3.2_candidate_response_generation.json"
  local stage32_output="${stage32_dir}/${stage32_output_name}"

  mkdir -p "${stage31_dir}" "${stage32_dir}" "${stage33_dir}" "${stage34_icp_dir}" "${icp_data_dir}"

  echo "============================================================"
  echo "[RUN] ${run_name} on visible GPU ${gpu_id}"
  echo "[STAGE2_LLM ] ${stage2_pretrained_llm_path}"
  echo "[STAGE3_BASE] ${STAGE3_BASE_MODEL}"
  echo "[REWARD_BASE] ${RM_BASE_MODEL}"
  echo "[REWARD_LORA] ${RM_ADAPTER}"
  echo "[RUN_DIR    ] ${run_dir}"
  echo "============================================================"

  echo "============================================================"
  echo "[3.1] Prepare target data"
  echo "============================================================"
  cp "${MEMBER_FILE}" "${stage31_member}"
  cp "${NONMEMBER_FILE}" "${stage31_nonmember}"
  echo "[3.1] member    -> ${stage31_member}"
  echo "[3.1] nonmember -> ${stage31_nonmember}"

  echo "============================================================"
  echo "[3.2] Candidate response generation"
  echo "============================================================"
  if [[ -s "${stage32_output}" ]]; then
    echo "[3.2] Found existing output, skip generation: ${stage32_output}"
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${STAGE32_SCRIPT}" \
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
  local policy_lora="${stage33_dir}/final_policy_lora"
  if [[ -f "${policy_lora}/adapter_config.json" ]]; then
    echo "[3.3] Found existing policy LoRA, skip PPO update: ${policy_lora}"
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${STAGE33_SCRIPT}" \
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

  if [[ ! -d "${policy_lora}" ]]; then
    local found_lora
    found_lora="$(find "${stage33_dir}" -type f -name adapter_config.json | head -n 1 || true)"
    if [[ -n "${found_lora}" ]]; then
      policy_lora="$(dirname "${found_lora}")"
    else
      echo "[ERROR] Cannot find policy LoRA adapter after 3.3 in: ${stage33_dir}"
      exit 1
    fi
  fi
  echo "[3.3] policy LoRA -> ${policy_lora}"

  echo "============================================================"
  echo "[3.4-ICP] Convert 3.2 data to ICP-MIA format"
  echo "============================================================"
  python "${ICP_CONVERT_SCRIPT}" \
    --step2_path "${stage32_output}" \
    --output_dir "${icp_data_dir}" \
    --file_prefix "icp_${model_tag}" \
    --member_size "${MEMBER_SIZE}" \
    --nonmember_size "${NONMEMBER_SIZE}" \
    --seed "${SEED}" \
    --member_sample_mode first \
    --nonmember_sample_mode first \
    --response_field "${ICP_RESPONSE_FIELD}"

  if [[ "${ICP_MODE}" == "both" ]]; then
    run_icp_mode "${gpu_id}" "sp" "${model_tag}" "${policy_lora}" "${icp_data_dir}" "${stage34_icp_dir}"
    run_icp_mode "${gpu_id}" "ref" "${model_tag}" "${policy_lora}" "${icp_data_dir}" "${stage34_icp_dir}"
  else
    run_icp_mode "${gpu_id}" "${ICP_MODE}" "${model_tag}" "${policy_lora}" "${icp_data_dir}" "${stage34_icp_dir}"
  fi

  echo "============================================================"
  echo "[DONE] ${run_name}"
  echo "============================================================"
}

wait_batch() {
  local -n pids_ref=$1
  local -n names_ref=$2
  local failed=0

  for i in "${!pids_ref[@]}"; do
    set +e
    wait "${pids_ref[$i]}"
    local rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      echo "[ERROR] ${names_ref[$i]} failed with exit code ${rc}. Log: $(model_log_root "${names_ref[$i]}")/${names_ref[$i]}.log"
      failed=1
    fi
  done

  pids_ref=()
  names_ref=()

  if [[ "${failed}" -ne 0 ]]; then
    exit 1
  fi
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
echo "[BATCH] ICP-MIA baseline strictly following membership inference steps"
echo "[MODE ] ICP_MODE=${ICP_MODE}"
echo "[FIXED] reward=${RM_BASE_MODEL} + ${RM_ADAPTER}"
echo "[FIXED] stage3_base=${STAGE3_BASE_MODEL}"
echo "[VARY ] only stage2 --pretrained_llm_path"
echo "[GPUs ] ${CUDA_DEVICES}; MAX_PARALLEL=${MAX_PARALLEL}"
if [[ -n "${RUN_MODEL_TAGS}" ]]; then
  echo "[RUN  ] only model tags: ${RUN_MODEL_TAGS}"
fi
echo "[OUT  ] first models: ${OUTPUT_ROOT}"
echo "[OUT2 ] remaining   : ${REMAINING_OUTPUT_ROOT}"
echo "[LOG  ] first models: ${LOG_ROOT}"
echo "[LOG2 ] remaining   : ${REMAINING_LOG_ROOT}"
echo "============================================================"

PIDS=()
NAMES=()

for i in "${!STAGE2_MODEL_TAGS[@]}"; do
  gpu_id="${GPU_IDS[$(( i % ${#GPU_IDS[@]} ))]}"
  model_tag="${STAGE2_MODEL_TAGS[$i]}"
  if ! should_run_model "${model_tag}"; then
    echo "[SKIP] ${model_tag} not in RUN_MODEL_TAGS=${RUN_MODEL_TAGS}"
    continue
  fi

  log_root_for_model="$(model_log_root "${model_tag}")"
  mkdir -p "${log_root_for_model}"
  migrate_existing_log "${model_tag}"
  log_path="${log_root_for_model}/${model_tag}.log"

  run_one_model "${gpu_id}" "${model_tag}" "${STAGE2_MODEL_PATHS[$i]}" > "${log_path}" 2>&1 &
  PIDS+=("$!")
  NAMES+=("${model_tag}")

  if (( ${#PIDS[@]} >= MAX_PARALLEL )); then
    wait_batch PIDS NAMES
  fi
done

if (( ${#PIDS[@]} > 0 )); then
  wait_batch PIDS NAMES
fi

SUMMARY_CSV="${REMAINING_OUTPUT_ROOT}/icp_mia_membership_steps_summary.csv"
python - "${SUMMARY_CSV}" "${OUTPUT_ROOT}" "${REMAINING_OUTPUT_ROOT}" <<'PY'
import csv
import glob
import os
import sys

summary_csv = sys.argv[1]
output_roots = sys.argv[2:]
files = []
for output_root in output_roots:
    files.extend(glob.glob(os.path.join(output_root, "*", "3.4-ICP-MIA-Baseline", "results-*", "*_results.csv")))
files = sorted(set(files))

rows = []
fieldnames = []
for path in files:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["source_csv"] = path
            rows.append(row)
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

if rows:
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Summary CSV: {summary_csv}")
else:
    print(f"[WARN] No ICP result CSV files found under: {', '.join(output_roots)}")
PY

echo "============================================================"
echo "[ALL DONE]"
echo "Output root        : ${OUTPUT_ROOT}"
echo "Remaining out root : ${REMAINING_OUTPUT_ROOT}"
echo "Log root           : ${LOG_ROOT}"
echo "Remaining log root : ${REMAINING_LOG_ROOT}"
echo "Summary    : ${SUMMARY_CSV}"
echo "============================================================"
