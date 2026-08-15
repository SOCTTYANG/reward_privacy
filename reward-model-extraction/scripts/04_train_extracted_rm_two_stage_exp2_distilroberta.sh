#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=/run/media/vipuser/data/projects/bai-rm-extraction-exp

cd ${PROJECT_DIR}

python -m src.train_extracted_rm_two_stage \
  --student_model_path ${PROJECT_DIR}/models/distilroberta-base \
  --hh_pref_train_path ${PROJECT_DIR}/data/hh_pref_train.jsonl \
  --hh_pref_eval_path ${PROJECT_DIR}/data/hh_pref_test.jsonl \
  --scored_aux_path ${PROJECT_DIR}/data/scored_aux_exp1.jsonl \
  --pku_pref_eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${PROJECT_DIR}/output/extracted_rm_exp2_roberta_to_distilroberta \
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
  --seed 42