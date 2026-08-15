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
export TMPDIR=/mnt/bai_data/tmp

mkdir -p ${HF_HOME} ${HF_DATASETS_CACHE} ${TRANSFORMERS_CACHE} ${TORCH_HOME} ${TMPDIR}

PROJECT_DIR=/mnt/bai_data/projects/bai-rm-extraction-exp

MODEL_ID=${1}
RUN_NAME=${2}
MAX_TRAIN=${3:-2000}
MAX_EVAL=${4:-500}
EPOCHS=${5:-1}
LR=${6:-1e-5}
MARGIN=${7:-5.0}
MAX_LENGTH=${8:-512}

cd ${PROJECT_DIR}

python -m src.train_target_rm_llama_lora \
  --model_name_or_path ${MODEL_ID} \
  --train_path ${PROJECT_DIR}/data/train.jsonl \
  --eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${PROJECT_DIR}/output/target_rm_${RUN_NAME} \
  --max_train_samples ${MAX_TRAIN} \
  --max_eval_samples ${MAX_EVAL} \
  --epochs ${EPOCHS} \
  --batch_size 1 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length ${MAX_LENGTH} \
  --lr ${LR} \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --grad_clip 1.0 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --margin ${MARGIN} \
  --bf16 \
  --gradient_checkpointing \
  --seed 42
