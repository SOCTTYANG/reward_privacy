#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 所有缓存统一放到空间充足的 /mnt/model_data
export HF_HOME=/mnt/model_data/cache/huggingface
export HF_DATASETS_CACHE=/mnt/model_data/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/mnt/model_data/cache/huggingface/transformers
export TORCH_HOME=/mnt/model_data/cache/torch

PROJECT_DIR=/mnt/model_data/projects/bai-rm-extraction-exp

STUDENT_MODEL=${PROJECT_DIR}/models/roberta-base
HH_TRAIN=${PROJECT_DIR}/data/hh_pref_train.jsonl
HH_EVAL=${PROJECT_DIR}/data/hh_pref_test.jsonl
SCORED_AUX=${PROJECT_DIR}/data/scored_aux_llama2_7b.jsonl
PKU_EVAL=${PROJECT_DIR}/data/test.jsonl

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
    "${HH_TRAIN}" \
    "${HH_EVAL}" \
    "${SCORED_AUX}" \
    "${PKU_EVAL}"
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
echo "Stage A: HH preference pretraining     ENABLED"
echo "Stage B: auxiliary distillation        DISABLED"
echo "Output : ${ONLY_PRETRAIN_OUTPUT}"
echo "======================================================"

python -m src.train_extracted_rm_two_stage \
  --student_model_path "${STUDENT_MODEL}" \
  --hh_pref_train_path "${HH_TRAIN}" \
  --hh_pref_eval_path "${HH_EVAL}" \
  --scored_aux_path "${SCORED_AUX}" \
  --pku_pref_eval_path "${PKU_EVAL}" \
  --output_dir "${ONLY_PRETRAIN_OUTPUT}" \
  --max_hh_train_samples 5000 \
  --max_hh_eval_samples 1000 \
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
echo "Stage A: HH preference pretraining     DISABLED"
echo "Stage B: auxiliary distillation        ENABLED"
echo "Output : ${ONLY_DISTILL_OUTPUT}"
echo "======================================================"

# 这是一个新的 Python 进程，会重新从原始 RoBERTa-base 初始化，
# 不会加载上一个 Only Pretrain 实验的模型。
python -m src.train_extracted_rm_two_stage \
  --student_model_path "${STUDENT_MODEL}" \
  --hh_pref_train_path "${HH_TRAIN}" \
  --hh_pref_eval_path "${HH_EVAL}" \
  --scored_aux_path "${SCORED_AUX}" \
  --pku_pref_eval_path "${PKU_EVAL}" \
  --output_dir "${ONLY_DISTILL_OUTPUT}" \
  --max_hh_train_samples 5000 \
  --max_hh_eval_samples 1000 \
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
