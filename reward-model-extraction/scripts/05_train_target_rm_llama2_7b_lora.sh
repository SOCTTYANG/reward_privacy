#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================
# Cache paths on data disk
# =========================
export HF_HOME=/path/to/code-data/cache/huggingface
export HF_DATASETS_CACHE=/path/to/code-data/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/path/to/code-data/cache/huggingface/transformers
export TORCH_HOME=/path/to/code-data/cache/torch

mkdir -p ${HF_HOME}
mkdir -p ${HF_DATASETS_CACHE}
mkdir -p ${TRANSFORMERS_CACHE}
mkdir -p ${TORCH_HOME}

# =========================
# Project paths
# =========================
PROJECT_DIR=/path/to/code
LLAMA2_BASE=/path/to/target-model/llama2-7b

cd ${PROJECT_DIR}

python -m src.train_target_rm_llama_lora \
  --model_name_or_path ${LLAMA2_BASE} \
  --train_path ${PROJECT_DIR}/data/train.jsonl \
  --eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${PROJECT_DIR}/output/target_rm_llama2_7b_lora_10k_e1 \
  --max_train_samples 30000 \
  --max_eval_samples 1000 \
  --epochs 1 \
  --batch_size 1 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --max_length 768 \
  --lr 8e-6 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --grad_clip 1.0 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --margin 10.0 \
  --bf16 \
  --gradient_checkpointing \
  --seed 42
