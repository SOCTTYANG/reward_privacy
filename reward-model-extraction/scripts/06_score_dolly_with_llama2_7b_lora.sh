#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME=/mnt/bai_data/cache/huggingface
export HF_DATASETS_CACHE=/mnt/bai_data/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/mnt/bai_data/cache/huggingface/transformers
export TORCH_HOME=/mnt/bai_data/cache/torch

mkdir -p ${HF_HOME}
mkdir -p ${HF_DATASETS_CACHE}
mkdir -p ${TRANSFORMERS_CACHE}
mkdir -p ${TORCH_HOME}

PROJECT_DIR=/mnt/bai_data/projects/bai-rm-extraction-exp
LLAMA2_BASE=/home/vipuser/Desktop/model/llama2-7b
LLAMA2_RM=${PROJECT_DIR}/output/target_rm_llama2_7b_lora_10k_e2

cd ${PROJECT_DIR}

python -m src.score_aux_with_llama_lora \
  --base_model_path ${LLAMA2_BASE} \
  --lora_adapter_path ${LLAMA2_RM} \
  --aux_path ${PROJECT_DIR}/data/aux_dolly.jsonl \
  --output_path ${PROJECT_DIR}/data/scored_aux_llama2_7b.jsonl \
  --max_samples 5000 \
  --batch_size 2 \
  --max_length 512 \
  --bf16