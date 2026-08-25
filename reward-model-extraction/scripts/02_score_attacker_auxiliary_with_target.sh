#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=/path/to/code

cd ${PROJECT_DIR}

python -m src.score_aux_with_target \
  --target_model_path ${PROJECT_DIR}/output/target_rm_roberta \
  --aux_path ${PROJECT_DIR}/data/attacker_auxiliary_dataset.jsonl \
  --output_path ${PROJECT_DIR}/data/scored_aux_exp1.jsonl \
  --max_samples 5000 \
  --batch_size 32 \
  --max_length 512