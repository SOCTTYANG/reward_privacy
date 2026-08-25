#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=/path/to/code

cd ${PROJECT_DIR}

python -m src.train_target_rm \
  --model_name_or_path ${PROJECT_DIR}/models/roberta-base \
  --train_path ${PROJECT_DIR}/data/train.jsonl \
  --eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${PROJECT_DIR}/output/target_rm_roberta \
  --max_train_samples 5000 \
  --max_eval_samples 1000 \
  --epochs 1 \
  --batch_size 8 \
  --eval_batch_size 8 \
  --max_length 512 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --grad_clip 1.0 \
  --seed 42