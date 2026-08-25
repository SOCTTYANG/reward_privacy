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

cd ${PROJECT_DIR}

python -m src.train_extracted_rm_llama_lora_two_stage \
  --student_model_path ${LLAMA2_BASE} \
  --attacker_preference_dataset_train_path ${PROJECT_DIR}/data/attacker_preference_dataset_train.jsonl \
  --attacker_preference_dataset_eval_path ${PROJECT_DIR}/data/attacker_preference_dataset_test.jsonl \
  --scored_aux_path ${PROJECT_DIR}/data/scored_aux_llama2_7b.jsonl \
  --defender_eval_eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${PROJECT_DIR}/output/extracted_rm_exp5_llama2_7b_to_llama2_7b_lora \
  --max_attacker_preference_train_samples 5000 \
  --max_attacker_preference_eval_samples 1000 \
  --max_aux_samples 5000 \
  --max_defender_evaluation_eval_samples 1000 \
  --aux_train_ratio 0.9 \
  --pref_epochs 1 \
  --distill_epochs 1 \
  --pref_batch_size 1 \
  --distill_batch_size 1 \
  --eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --max_length 512 \
  --pref_lr 2e-5 \
  --distill_lr 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --grad_clip 1.0 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --bf16 \
  --gradient_checkpointing \
  --seed 42
