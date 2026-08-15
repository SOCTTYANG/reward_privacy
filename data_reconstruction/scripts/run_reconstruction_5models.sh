#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${DR_DIR}/.." && pwd)}"
export PYTHONPATH="${DR_DIR}:${PYTHONPATH:-}"
cd "${ROOT_DIR}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/bai_data/yang-safe/data_reconstruction/reconstruction_eval}"
PRETRAINED_ADAPTER_ROOT="${PRETRAINED_ADAPTER_ROOT:-}"
EXTRACTOR_PATH="${EXTRACTOR_PATH:-/home/vipuser/Desktop/model/bge-small-en-v1.5}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${ROOT_DIR}/output/dual_reward_lora}"
PROCESSED_DIR="${PROCESSED_DIR:-${ROOT_DIR}/data/pku_saferlhf_10k_triplet}"
DATASET_NAME="${DATASET_NAME:-PKU-Alignment/PKU-SafeRLHF-10K}"
EVAL_FILE="${EVAL_FILE:-${PROCESSED_DIR}/eval.jsonl}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${DR_DIR}/configs/ds_zero2_lora_single_a100_40g.json}"

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29531}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_SELECT="${RUN_SELECT:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-320}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LAMBDA_MLE="${LAMBDA_MLE:-1.0}"
LAMBDA_COS="${LAMBDA_COS:-0.05}"
FORCE_REBUILD="${FORCE_REBUILD:-False}"

NUM_CANDIDATES="${NUM_CANDIDATES:-3}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-96}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-1.0}"
GEN_TOP_P="${GEN_TOP_P:-0.95}"
GEN_REPETITION_PENALTY="${GEN_REPETITION_PENALTY:-1.05}"
GEN_TORCH_DTYPE="${GEN_TORCH_DTYPE:-auto}"

REWARD_MAX_LENGTH="${REWARD_MAX_LENGTH:-512}"
REWARD_SCORE_HEAD="${REWARD_SCORE_HEAD:-help}"
LAMBDA_SAFE="${LAMBDA_SAFE:-1.0}"

METRIC_BATCH_SIZE="${METRIC_BATCH_SIZE:-4}"
EMBED_MAX_LENGTH="${EMBED_MAX_LENGTH:-512}"
METRIC_TORCH_DTYPE="${METRIC_TORCH_DTYPE:-auto}"
BLEU_ORDER="${BLEU_ORDER:-1}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"

MODEL_NAMES=(
  "llama2-7b"
  "mistral-7b-v0.1"
  "llama32-3b"
  "qwen3-8b"
  "llama2-13b-hf"
)

MODEL_PATHS=(
  "/home/vipuser/Desktop/model/llama2-7b"
  "/mnt/model_data/models/mistral-7b-v0.1"
  "/mnt/model_data/models/llama32-3b"
  "/mnt/model_data/models/qwen3-8b"
  "/mnt/model_data/models/llama2-13b-hf"
)

mkdir -p "${OUTPUT_ROOT}"
SUMMARY_TSV="${OUTPUT_ROOT}/bleu${BLEU_ORDER}_cosine_summary.tsv"
printf "model\tmodel_path\tbleu_order\tbleu_score\tcosine_similarity\tsummary_file\n" > "${SUMMARY_TSV}"

if [[ ! -d "${EXTRACTOR_PATH}" ]]; then
  echo "[ERROR] EXTRACTOR_PATH does not exist: ${EXTRACTOR_PATH}" >&2
  exit 1
fi

if [[ ! -d "${REWARD_MODEL_PATH}" ]]; then
  echo "[ERROR] REWARD_MODEL_PATH does not exist: ${REWARD_MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "[ERROR] DEEPSPEED_CONFIG is not a file: ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi

if [[ "${RUN_TRAIN}" == "1" ]] && ! command -v deepspeed >/dev/null 2>&1; then
  echo "[ERROR] deepspeed is required for RUN_TRAIN=1 but was not found in PATH." >&2
  exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
  MODEL_NAME="${MODEL_NAMES[$i]}"
  MODEL_PATH="${MODEL_PATHS[$i]}"
  MODEL_OUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}"
  TRAIN_ADAPTER_DIR="${MODEL_OUT_DIR}/reconstruction_lora"
  if [[ "${RUN_TRAIN}" == "0" && -n "${PRETRAINED_ADAPTER_ROOT}" ]]; then
    ADAPTER_DIR="${PRETRAINED_ADAPTER_ROOT}/${MODEL_NAME}/reconstruction_lora"
  else
    ADAPTER_DIR="${TRAIN_ADAPTER_DIR}"
  fi
  STAGE2_FILE="${MODEL_OUT_DIR}/stage2_candidates_k${NUM_CANDIDATES}.jsonl"
  STAGE3_FILE="${MODEL_OUT_DIR}/stage3_lowest_reward_selected.jsonl"
  DETAIL_FILE="${MODEL_OUT_DIR}/bleu_cosine_details.jsonl"
  SUMMARY_FILE="${MODEL_OUT_DIR}/bleu_cosine_summary.json"
  MASTER_PORT="$((MASTER_PORT_BASE + i))"

  mkdir -p "${MODEL_OUT_DIR}"

  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[ERROR] Model path does not exist for ${MODEL_NAME}: ${MODEL_PATH}" >&2
    exit 1
  fi

  echo "============================================================"
  echo "[MODEL] ${MODEL_NAME}"
  echo "[PATH ] ${MODEL_PATH}"
  echo "[LORA ] ${ADAPTER_DIR}"
  echo "[OUT  ] ${MODEL_OUT_DIR}"
  echo "============================================================"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo "[STEP 1] Fine-tuning seq2seq reconstruction model..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" deepspeed --master_port "${MASTER_PORT}" \
      --module safe_rlhf.reconstruction.__main___seq2seq \
      --model_name_or_path "${MODEL_PATH}" \
      --extractor_name_or_path "${EXTRACTOR_PATH}" \
      --dataset_name "${DATASET_NAME}" \
      --processed_dir "${PROCESSED_DIR}" \
      --max_length "${TRAIN_MAX_LENGTH}" \
      --eval_ratio 0.05 \
      --split_seed 42 \
      --force_rebuild "${FORCE_REBUILD}" \
      --output_dir "${TRAIN_ADAPTER_DIR}" \
      --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
      --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
      --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --learning_rate "${LEARNING_RATE}" \
      --lr_scheduler_type cosine \
      --warmup_ratio 0.03 \
      --weight_decay 0.0 \
      --logging_steps 10 \
      --save_strategy no \
      --eval_strategy no \
      --bf16 True \
      --tf32 False \
      --gradient_checkpointing True \
      --remove_unused_columns False \
      --dataloader_num_workers 0 \
      --report_to none \
      --seed 42 \
      --lambda_mle "${LAMBDA_MLE}" \
      --lambda_cos "${LAMBDA_COS}" \
      --deepspeed "${DEEPSPEED_CONFIG}"
  else
    echo "[STEP 1] Skipped fine-tuning because RUN_TRAIN=${RUN_TRAIN}."
  fi

  if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo "[ERROR] Reconstruction adapter not found: ${ADAPTER_DIR}" >&2
    exit 1
  fi

  if [[ ! -f "${EVAL_FILE}" ]]; then
    echo "[ERROR] EVAL_FILE is not available: ${EVAL_FILE}" >&2
    echo "[ERROR] Step 1 should create it under PROCESSED_DIR, or pass EVAL_FILE explicitly." >&2
    exit 1
  fi

  if [[ "${RUN_GENERATE}" == "1" ]]; then
    echo "[STEP 2] Generating ${NUM_CANDIDATES} candidate dispreferred responses..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${DR_DIR}/stage2_generate_candidates_k3.py" \
      --base_model_path "${MODEL_PATH}" \
      --adapter_dir "${ADAPTER_DIR}" \
      --input_file "${EVAL_FILE}" \
      --output_file "${STAGE2_FILE}" \
      --max_samples "${MAX_SAMPLES}" \
      --device "${DEVICE}" \
      --num_candidates "${NUM_CANDIDATES}" \
      --max_new_tokens "${GEN_MAX_NEW_TOKENS}" \
      --temperature "${GEN_TEMPERATURE}" \
      --top_p "${GEN_TOP_P}" \
      --repetition_penalty "${GEN_REPETITION_PENALTY}" \
      --torch_dtype "${GEN_TORCH_DTYPE}"
  else
    echo "[STEP 2] Skipped generation because RUN_GENERATE=${RUN_GENERATE}."
  fi

  if [[ ! -f "${STAGE2_FILE}" ]]; then
    echo "[ERROR] Stage 2 candidate file not found: ${STAGE2_FILE}" >&2
    exit 1
  fi

  if [[ "${RUN_SELECT}" == "1" ]]; then
    echo "[STEP 3] Selecting the lowest-reward candidate..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${DR_DIR}/stage3_select_lowest_reward.py" \
      --reward_model_path "${REWARD_MODEL_PATH}" \
      --triplet_file "${EVAL_FILE}" \
      --stage2_file "${STAGE2_FILE}" \
      --output_file "${STAGE3_FILE}" \
      --device "${DEVICE}" \
      --max_samples "${MAX_SAMPLES}" \
      --max_length "${REWARD_MAX_LENGTH}" \
      --score_head "${REWARD_SCORE_HEAD}" \
      --lambda_safe "${LAMBDA_SAFE}"
  else
    echo "[STEP 3] Skipped selection because RUN_SELECT=${RUN_SELECT}."
  fi

  if [[ ! -f "${STAGE3_FILE}" ]]; then
    echo "[ERROR] Stage 3 selected file not found: ${STAGE3_FILE}" >&2
    exit 1
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    echo "[METRIC] Computing BLEU and cosine similarity..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${DR_DIR}/evaluation/eval_bleu_cosine.py" \
      --input_file "${STAGE3_FILE}" \
      --output_file "${DETAIL_FILE}" \
      --summary_file "${SUMMARY_FILE}" \
      --ref_field real_y_minus \
      --pred_field lowest_reward_generated_y_minus \
      --embedder_path "${EXTRACTOR_PATH}" \
      --device "${DEVICE}" \
      --batch_size "${METRIC_BATCH_SIZE}" \
      --embed_max_length "${EMBED_MAX_LENGTH}" \
      --torch_dtype "${METRIC_TORCH_DTYPE}" \
      --bleu_order "${BLEU_ORDER}" \
      --max_samples "${MAX_SAMPLES}"
  else
    echo "[METRIC] Skipped evaluation because RUN_EVAL=${RUN_EVAL}."
  fi

  if [[ -f "${SUMMARY_FILE}" ]]; then
    read -r BLEU_SCORE COSINE_SIMILARITY < <(
      python -c 'import json, sys; d = json.load(open(sys.argv[1], encoding="utf-8")); print(d["bleu_score"], d["cosine_similarity"])' "${SUMMARY_FILE}"
    )
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${MODEL_NAME}" \
      "${MODEL_PATH}" \
      "${BLEU_ORDER}" \
      "${BLEU_SCORE}" \
      "${COSINE_SIMILARITY}" \
      "${SUMMARY_FILE}" >> "${SUMMARY_TSV}"
  fi
done

echo "============================================================"
echo "[DONE] Data reconstruction pipeline finished for all models."
echo "[DONE] Aggregate metrics: ${SUMMARY_TSV}"
echo "[DONE] Per-model outputs: ${OUTPUT_ROOT}/<model_name>/"
echo "============================================================"
