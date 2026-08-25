#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use a configurable cache directory.
export HF_HOME=/path/to/cache/huggingface
export HF_DATASETS_CACHE=/path/to/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/path/to/cache/huggingface/transformers
export TORCH_HOME=/path/to/cache/torch

PROJECT_DIR=/path/to/code

STUDENT_MODEL=${PROJECT_DIR}/models/roberta-base
ATTACKER_PREFERENCE_TRAIN=${PROJECT_DIR}/data/attacker_preference_dataset_train.jsonl
ATTACKER_PREFERENCE_EVAL=${PROJECT_DIR}/data/attacker_preference_dataset_test.jsonl
SCORED_AUX=${PROJECT_DIR}/data/scored_aux_llama2_7b.jsonl
Defender Evaluation_EVAL=${PROJECT_DIR}/data/test.jsonl

ONLY_PRETRAIN_OUTPUT=${PROJECT_DIR}/output/ablation_llama2_7b_roberta_only_pretrain
ONLY_DISTILL_OUTPUT=${PROJECT_DIR}/output/ablation_llama2_7b_roberta_only_distill

LOG_DIR=${PROJECT_DIR}/logs/ablation_llama2_7b_roberta

mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_DIR}/output"
mkdir -p "${HF_HOME}"
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${TRANSFORMERS_CACHE}"
mkdir -p "${TORCH_HOME}"

cd "${PROJECT_DIR}"

echo "======================================================"
echo "Ablation experiments"
echo "Teacher RM : LLaMA2-7B"
echo "Student RM : RoBERTa-base"
echo "Project    : ${PROJECT_DIR}"
echo "======================================================"

# 检查所有必要文件，避免跑到一半才发现路径错误
for REQUIRED_PATH in \
    "${STUDENT_MODEL}" \
    "${ATTACKER_PREFERENCE_TRAIN}" \
    "${ATTACKER_PREFERENCE_EVAL}" \
    "${SCORED_AUX}" \
    "${Defender Evaluation_EVAL}"
do
    if [ ! -e "${REQUIRED_PATH}" ]; then
        echo "[ERROR] Required path does not exist:"
        echo "        ${REQUIRED_PATH}"
        exit 1
    fi
done

echo "[OK] All required model and data paths exist."

echo
echo "======================================================"
echo "Ablation 1: Only Pretrain"
echo "Stage A: Attacker Preference preference pretraining     ENABLED"
echo "Stage B: auxiliary distillation        DISABLED"
echo "Output : ${ONLY_PRETRAIN_OUTPUT}"
echo "======================================================"

python -m src.train_extracted_rm_two_stage \
  --student_model_path "${STUDENT_MODEL}" \
  --attacker_preference_dataset_train_path "${ATTACKER_PREFERENCE_TRAIN}" \
  --attacker_preference_dataset_eval_path "${ATTACKER_PREFERENCE_EVAL}" \
  --scored_aux_path "${SCORED_AUX}" \
  --defender_eval_eval_path "${Defender Evaluation_EVAL}" \
  --output_dir "${ONLY_PRETRAIN_OUTPUT}" \
  --max_attacker_preference_train_samples 5000 \
  --max_attacker_preference_eval_samples 1000 \
  --max_aux_samples 5000 \
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
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --grad_clip 1.0 \
  --seed 42 \
  --device cuda \
  --skip_stage_b \
  2>&1 | tee "${LOG_DIR}/only_pretrain.log"

if [ ! -f "${ONLY_PRETRAIN_OUTPUT}/model.safetensors" ] && \
   [ ! -f "${ONLY_PRETRAIN_OUTPUT}/pytorch_model.bin" ]; then
    echo "[ERROR] Only Pretrain model was not saved successfully."
    exit 1
fi

echo "[OK] Only Pretrain experiment completed successfully."

echo
echo "======================================================"
echo "Ablation 2: Only Distill"
echo "Stage A: Attacker Preference preference pretraining     DISABLED"
echo "Stage B: auxiliary distillation        ENABLED"
echo "Output : ${ONLY_DISTILL_OUTPUT}"
echo "======================================================"

# 这是一个新的 Python 进程，会重新从原始 RoBERTa-base 初始化，
# 不会加载上一个 Only Pretrain 实验的模型。
python -m src.train_extracted_rm_two_stage \
  --student_model_path "${STUDENT_MODEL}" \
  --attacker_preference_dataset_train_path "${ATTACKER_PREFERENCE_TRAIN}" \
  --attacker_preference_dataset_eval_path "${ATTACKER_PREFERENCE_EVAL}" \
  --scored_aux_path "${SCORED_AUX}" \
  --defender_eval_eval_path "${Defender Evaluation_EVAL}" \
  --output_dir "${ONLY_DISTILL_OUTPUT}" \
  --max_attacker_preference_train_samples 5000 \
  --max_attacker_preference_eval_samples 1000 \
  --max_aux_samples 5000 \
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
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --grad_clip 1.0 \
  --seed 42 \
  --device cuda \
  --skip_stage_a \
  2>&1 | tee "${LOG_DIR}/only_distill.log"

if [ ! -f "${ONLY_DISTILL_OUTPUT}/model.safetensors" ] && \
   [ ! -f "${ONLY_DISTILL_OUTPUT}/pytorch_model.bin" ]; then
    echo "[ERROR] Only Distill model was not saved successfully."
    exit 1
fi

echo "[OK] Only Distill experiment completed successfully."

echo
echo "======================================================"
echo "All ablation training experiments finished."
echo
echo "Only Pretrain:"
echo "${ONLY_PRETRAIN_OUTPUT}"
echo
echo "Only Distill:"
echo "${ONLY_DISTILL_OUTPUT}"
echo
echo "Logs:"
echo "${LOG_DIR}"
echo "======================================================"
