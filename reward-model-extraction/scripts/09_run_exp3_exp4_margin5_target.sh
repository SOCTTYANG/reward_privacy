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
export TMPDIR=/path/to/code-data/tmp

mkdir -p ${HF_HOME}
mkdir -p ${HF_DATASETS_CACHE}
mkdir -p ${TRANSFORMERS_CACHE}
mkdir -p ${TORCH_HOME}
mkdir -p ${TMPDIR}

PROJECT_DIR=/path/to/code
SCORED_AUX=${PROJECT_DIR}/data/scored_aux_llama2_7b_margin5_e2.jsonl

cd ${PROJECT_DIR}

echo "======================================================"
echo "Exp3-M5: T3 LLaMA2-7B margin5 target RM -> RoBERTa-base substitute"
echo "======================================================"

python -m src.train_extracted_rm_two_stage \
  --student_model_path ${PROJECT_DIR}/models/roberta-base \
  --attacker_preference_dataset_train_path ${PROJECT_DIR}/data/attacker_preference_dataset_train.jsonl \
  --attacker_preference_dataset_eval_path ${PROJECT_DIR}/data/attacker_preference_dataset_test.jsonl \
  --scored_aux_path ${SCORED_AUX} \
  --defender_eval_eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${PROJECT_DIR}/output/extracted_rm_exp3_margin5_llama2_7b_to_roberta \
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
  --seed 42

echo "======================================================"
echo "Exp4-M5: T3 LLaMA2-7B margin5 target RM -> DistilRoBERTa-base substitute"
echo "======================================================"

python -m src.train_extracted_rm_two_stage \
  --student_model_path ${PROJECT_DIR}/models/distilroberta-base \
  --attacker_preference_dataset_train_path ${PROJECT_DIR}/data/attacker_preference_dataset_train.jsonl \
  --attacker_preference_dataset_eval_path ${PROJECT_DIR}/data/attacker_preference_dataset_test.jsonl \
  --scored_aux_path ${SCORED_AUX} \
  --defender_eval_eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${PROJECT_DIR}/output/extracted_rm_exp4_margin5_llama2_7b_to_distilroberta \
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
  --seed 42

echo "======================================================"
echo "Exp3-M5 and Exp4-M5 finished."
echo "======================================================"
