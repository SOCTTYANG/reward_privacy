#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME=/path/to/code-data/cache/huggingface
export HF_DATASETS_CACHE=/path/to/code-data/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/path/to/code-data/cache/huggingface/transformers
export TORCH_HOME=/path/to/code-data/cache/torch


PROJECT_DIR=/path/to/code

cd ${PROJECT_DIR}


echo "=========================================="
echo "Defender Evaluation10K DeBERTa-v3-large Training"
echo "=========================================="


python -m src.train_extracted_rm_two_stage \
 --student_model_path /path/to/models/deberta-v3-large \
 --attacker_preference_dataset_train_path ${PROJECT_DIR}/data/defender_evaluation_pref_train.jsonl \
 --attacker_preference_dataset_eval_path ${PROJECT_DIR}/data/defender_evaluation_pref_eval.jsonl \
 --scored_aux_path ${PROJECT_DIR}/data/scored_aux_llama2_7b.jsonl \
 --defender_eval_eval_path ${PROJECT_DIR}/data/test.jsonl \
 --output_dir ${PROJECT_DIR}/output/defender_evaluation_llama2_7b_deberta_v3_large \
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



echo "=========================================="
echo "Defender Evaluation10K DeBERTa-v2-xlarge Training"
echo "=========================================="


python -m src.train_extracted_rm_two_stage \
 --student_model_path /path/to/models/deberta-v2-xlarge \
 --attacker_preference_dataset_train_path ${PROJECT_DIR}/data/defender_evaluation_pref_train.jsonl \
 --attacker_preference_dataset_eval_path ${PROJECT_DIR}/data/defender_evaluation_pref_eval.jsonl \
 --scored_aux_path ${PROJECT_DIR}/data/scored_aux_llama2_7b.jsonl \
 --defender_eval_eval_path ${PROJECT_DIR}/data/test.jsonl \
 --output_dir ${PROJECT_DIR}/output/defender_evaluation_llama2_7b_deberta_v2_xlarge \
 --max_attacker_preference_train_samples 9500 \
 --max_attacker_preference_eval_samples 500 \
 --max_aux_samples 5000 \
 --max_defender_evaluation_eval_samples 1000 \
 --aux_train_ratio 0.9 \
 --pref_epochs 8 \
 --distill_epochs 1 \
 --pref_batch_size 4 \
 --distill_batch_size 8 \
 --eval_batch_size 16 \
 --max_length 512 \
 --pref_lr 2e-5 \
 --distill_lr 1e-5 \
 --weight_decay 0.01 \
 --warmup_ratio 0.03 \
 --grad_clip 1.0 \
 --seed 42



echo "=========================================="
echo "Start Diff Evaluation"
echo "=========================================="


TARGET_BASE=/path/to/target-model/llama2-7b
TARGET_ADAPTER=${PROJECT_DIR}/output/target_rm_llama2_7b_lora_full_margin5_e2


####################################
# RoBERTa Defender Evaluation10K Diff
####################################

python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model ${TARGET_BASE} \
 --target_adapter_path ${TARGET_ADAPTER} \
 --substitute_model_path ${PROJECT_DIR}/output/defender_evaluation_exp3_llama2_7b_to_roberta \
 --train_path ${PROJECT_DIR}/data/train.jsonl \
 --test_path ${PROJECT_DIR}/data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir ${PROJECT_DIR}/output/diff_defender_evaluation_roberta \
 --max_length 512 \
 --target_batch_size 1 \
 --substitute_batch_size 16 \
 --bf16



####################################
# DeBERTa-v3 Defender Evaluation10K Diff
####################################

python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model ${TARGET_BASE} \
 --target_adapter_path ${TARGET_ADAPTER} \
 --substitute_model_path ${PROJECT_DIR}/output/defender_evaluation_llama2_7b_deberta_v3_large \
 --train_path ${PROJECT_DIR}/data/train.jsonl \
 --test_path ${PROJECT_DIR}/data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir ${PROJECT_DIR}/output/diff_defender_evaluation_deberta_v3_large \
 --max_length 512 \
 --target_batch_size 1 \
 --substitute_batch_size 16 \
 --bf16



####################################
# DeBERTa-v2 Defender Evaluation10K Diff
####################################

python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model ${TARGET_BASE} \
 --target_adapter_path ${TARGET_ADAPTER} \
 --substitute_model_path ${PROJECT_DIR}/output/defender_evaluation_llama2_7b_deberta_v2_xlarge \
 --train_path ${PROJECT_DIR}/data/train.jsonl \
 --test_path ${PROJECT_DIR}/data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir ${PROJECT_DIR}/output/diff_defender_evaluation_deberta_v2_xlarge \
 --max_length 384 \
 --target_batch_size 1 \
 --substitute_batch_size 8 \
 --bf16



echo "=========================================="
echo "ALL Defender Evaluation10K EXP FINISHED"
echo "=========================================="
