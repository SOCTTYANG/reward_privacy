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


PROJECT_DIR=/path/to/code

cd ${PROJECT_DIR}


echo "======================================================"
echo "Defender Evaluation-10K Ablation"
echo "Teacher: Llama2-7B"
echo "Student: RoBERTa-base"
echo "======================================================"


python -m src.train_extracted_rm_two_stage \
  --student_model_path ${PROJECT_DIR}/models/roberta-base \
  --attacker_preference_dataset_train_path ${PROJECT_DIR}/data/defender_evaluation_pref_train.jsonl \
  --attacker_preference_dataset_eval_path ${PROJECT_DIR}/data/defender_evaluation_pref_eval.jsonl \
  --scored_aux_path ${PROJECT_DIR}/data/scored_aux_llama2_7b.jsonl \
  --defender_eval_eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${PROJECT_DIR}/output/defender_evaluation_exp3_llama2_7b_to_roberta \
  --max_attacker_preference_train_samples 9500 \
  --max_attacker_preference_eval_samples 500 \
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
echo "Defender Evaluation-10K Ablation"
echo "Teacher: Llama2-7B"
echo "Student: DistilRoBERTa"
echo "======================================================"


python -m src.train_extracted_rm_two_stage \
  --student_model_path ${PROJECT_DIR}/models/distilroberta-base \
  --attacker_preference_dataset_train_path ${PROJECT_DIR}/data/defender_evaluation_pref_train.jsonl \
  --attacker_preference_dataset_eval_path ${PROJECT_DIR}/data/defender_evaluation_pref_eval.jsonl \
  --scored_aux_path ${PROJECT_DIR}/data/scored_aux_llama2_7b.jsonl \
  --defender_eval_eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${PROJECT_DIR}/output/defender_evaluation_exp4_llama2_7b_to_distilroberta \
  --max_attacker_preference_train_samples 9500 \
  --max_attacker_preference_eval_samples 500 \
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
echo "Defender Evaluation-10K Exp3 and Exp4 finished."
echo "======================================================"
