#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME=/path/to/code-data/cache/huggingface
export HF_DATASETS_CACHE=/path/to/code-data/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/path/to/code-data/cache/huggingface/transformers
export TORCH_HOME=/path/to/code-data/cache/torch

mkdir -p ${HF_HOME}
mkdir -p ${HF_DATASETS_CACHE}
mkdir -p ${TRANSFORMERS_CACHE}
mkdir -p ${TORCH_HOME}

PROJECT_DIR=/path/to/code
LLAMA2_BASE=/path/to/target-model/llama2-7b
LLAMA2_RM=${PROJECT_DIR}/output/target_rm_llama2_7b_lora_10k_e2

cd ${PROJECT_DIR}

python -m src.score_aux_with_llama_lora \
  --base_model_path ${LLAMA2_BASE} \
  --lora_adapter_path ${LLAMA2_RM} \
  --aux_path ${PROJECT_DIR}/data/attacker_auxiliary_dataset.jsonl \
  --output_path ${PROJECT_DIR}/data/scored_aux_llama2_7b.jsonl \
  --max_samples 5000 \
  --batch_size 2 \
  --max_length 512 \
  --bf16