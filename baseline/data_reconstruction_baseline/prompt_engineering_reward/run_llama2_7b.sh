#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-${REPOSITORY_ROOT}}"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/data_reconstruction:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

MODEL_NAME="${MODEL_NAME:-llama2-7b}"
MODEL_PATH="${MODEL_PATH:-/home/vipuser/Desktop/model/llama2-7b}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-/home/vipuser/Desktop/yang-safe-rlhf/output/dual_reward_lora}"
EMBEDDER_PATH="${EMBEDDER_PATH:-/home/vipuser/Desktop/model/bge-small-en-v1.5}"

PROCESSED_DIR="${PROCESSED_DIR:-${ROOT_DIR}/data/pku_saferlhf_10k_triplet}"
EVAL_FILE="${EVAL_FILE:-${PROCESSED_DIR}/eval.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/model_data/yang_safe/data_reconstruction/prompt_engineering_reward_baseline}"

GPU_ID="${GPU_ID:-0}"
GENERATOR_DEVICE="${GENERATOR_DEVICE:-cuda:0}"
REWARD_DEVICE="${REWARD_DEVICE:-cuda:0}"
GENERATOR_DEVICE_MAP="${GENERATOR_DEVICE_MAP:-none}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"

MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
DELTA="${DELTA:-0.05}"
SCORE_HEAD="${SCORE_HEAD:-help}"
LAMBDA_SAFE="${LAMBDA_SAFE:-1.0}"
DELTA_LABEL="${DELTA//./p}"
DELTA_LABEL="${DELTA_LABEL//-/_}"
RUN_TAG="${RUN_TAG:-delta_${DELTA_LABEL}_${SCORE_HEAD}_rounds${MAX_ROUNDS}}"
OUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${RUN_TAG}"

GENERATOR_MAX_INPUT_LENGTH="${GENERATOR_MAX_INPUT_LENGTH:-2048}"
REWARD_MAX_LENGTH="${REWARD_MAX_LENGTH:-512}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-160}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.9}"
GEN_TOP_P="${GEN_TOP_P:-0.95}"
GEN_REPETITION_PENALTY="${GEN_REPETITION_PENALTY:-1.05}"
EMPTY_RETRIES="${EMPTY_RETRIES:-2}"
SEED="${SEED:-42}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-4}"
EMBED_MAX_LENGTH="${EMBED_MAX_LENGTH:-512}"
METRIC_TORCH_DTYPE="${METRIC_TORCH_DTYPE:-auto}"
BLEU_ORDER="${BLEU_ORDER:-1}"

BASELINE_FILE="${OUT_DIR}/prompt_engineering_reward_baseline.jsonl"
BASELINE_SUMMARY_FILE="${OUT_DIR}/prompt_engineering_reward_baseline_summary.json"
DETAIL_FILE="${OUT_DIR}/bleu_cosine_details.jsonl"
METRIC_SUMMARY_FILE="${OUT_DIR}/bleu_cosine_summary.json"

mkdir -p "${OUT_DIR}"

LOCK_DIR="${OUT_DIR}/.run.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "[ERROR] Another run is already writing to: ${OUT_DIR}" >&2
  echo "[ERROR] Use a different RUN_TAG, or wait for the current run to finish." >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -d "${REWARD_MODEL_PATH}" ]]; then
  echo "[ERROR] REWARD_MODEL_PATH does not exist: ${REWARD_MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${EVAL_FILE}" ]]; then
  echo "[ERROR] EVAL_FILE is not available: ${EVAL_FILE}" >&2
  echo "[ERROR] Set EVAL_FILE to a JSONL triplet file with x/y_plus/y_minus fields." >&2
  exit 1
fi

if [[ "${RUN_EVAL}" == "1" && ! -d "${EMBEDDER_PATH}" ]]; then
  echo "[ERROR] EMBEDDER_PATH does not exist: ${EMBEDDER_PATH}" >&2
  exit 1
fi

cat > "${OUT_DIR}/README.txt" <<EOF
Prompt-engineering reward baseline for data reconstruction.

Model: ${MODEL_NAME}
Generator path: ${MODEL_PATH}
Reward model path: ${REWARD_MODEL_PATH}
Input eval file: ${EVAL_FILE}
Max rounds: ${MAX_ROUNDS}
Delta: ${DELTA}
Score head: ${SCORE_HEAD}
Run tag: ${RUN_TAG}
EOF

echo "============================================================"
echo "[MODEL] ${MODEL_NAME}"
echo "[GEN  ] ${MODEL_PATH}"
echo "[REWARD] ${REWARD_MODEL_PATH}"
echo "[INPUT] ${EVAL_FILE}"
echo "[OUT  ] ${OUT_DIR}"
echo "[TAG  ] ${RUN_TAG}"
echo "============================================================"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "[STEP] Running iterative prompt-engineering baseline..."
  BASELINE_SCRIPT="${SCRIPT_DIR}/run_baseline.py"
  echo "[STEP] Baseline script: ${BASELINE_SCRIPT}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${BASELINE_SCRIPT}" \
    --base_model_path "${MODEL_PATH}" \
    --reward_model_path "${REWARD_MODEL_PATH}" \
    --triplet_file "${EVAL_FILE}" \
    --output_file "${BASELINE_FILE}" \
    --summary_file "${BASELINE_SUMMARY_FILE}" \
    --max_samples "${MAX_SAMPLES}" \
    --generator_device "${GENERATOR_DEVICE}" \
    --reward_device "${REWARD_DEVICE}" \
    --generator_device_map "${GENERATOR_DEVICE_MAP}" \
    --torch_dtype "${TORCH_DTYPE}" \
    --generator_max_input_length "${GENERATOR_MAX_INPUT_LENGTH}" \
    --reward_max_length "${REWARD_MAX_LENGTH}" \
    --max_rounds "${MAX_ROUNDS}" \
    --delta "${DELTA}" \
    --score_head "${SCORE_HEAD}" \
    --lambda_safe "${LAMBDA_SAFE}" \
    --max_new_tokens "${GEN_MAX_NEW_TOKENS}" \
    --temperature "${GEN_TEMPERATURE}" \
    --top_p "${GEN_TOP_P}" \
    --repetition_penalty "${GEN_REPETITION_PENALTY}" \
    --empty_retries "${EMPTY_RETRIES}" \
    --seed "${SEED}"
else
  echo "[STEP] Skipped baseline because RUN_BASELINE=${RUN_BASELINE}."
fi

if [[ ! -f "${BASELINE_FILE}" ]]; then
  echo "[ERROR] Baseline output file not found: ${BASELINE_FILE}" >&2
  exit 1
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[METRIC] Computing BLEU and cosine similarity..."
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${ROOT_DIR}/data_reconstruction/evaluation/eval_bleu_cosine.py" \
    --input_file "${BASELINE_FILE}" \
    --output_file "${DETAIL_FILE}" \
    --summary_file "${METRIC_SUMMARY_FILE}" \
    --ref_field real_y_minus \
    --pred_field selected_generated_y_minus \
    --embedder_path "${EMBEDDER_PATH}" \
    --device "${GENERATOR_DEVICE}" \
    --batch_size "${METRIC_BATCH_SIZE}" \
    --embed_max_length "${EMBED_MAX_LENGTH}" \
    --torch_dtype "${METRIC_TORCH_DTYPE}" \
    --bleu_order "${BLEU_ORDER}" \
    --max_samples "${MAX_SAMPLES}"
else
  echo "[METRIC] Skipped evaluation because RUN_EVAL=${RUN_EVAL}."
fi

echo "============================================================"
echo "[DONE] Prompt-engineering baseline finished."
echo "[DONE] Baseline rows: ${BASELINE_FILE}"
echo "[DONE] Baseline summary: ${BASELINE_SUMMARY_FILE}"
if [[ -f "${METRIC_SUMMARY_FILE}" ]]; then
  echo "[DONE] BLEU/Cosine summary: ${METRIC_SUMMARY_FILE}"
fi
echo "============================================================"
