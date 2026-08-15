#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME=/mnt/bai_data/cache/huggingface
export HF_DATASETS_CACHE=/mnt/bai_data/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/mnt/bai_data/cache/huggingface/transformers
export TORCH_HOME=/mnt/bai_data/cache/torch


PROJECT_DIR=/mnt/model_data/projects/bai-rm-extraction-exp

cd ${PROJECT_DIR}


echo "=========================================="
echo "PKU10K DeBERTa-v3-large Training"
echo "=========================================="


python -m src.train_extracted_rm_two_stage \
 --student_model_path /mnt/model_data/models/deberta-v3-large \
 --hh_pref_train_path ${PROJECT_DIR}/data/pku10k_pref_train.jsonl \
 --hh_pref_eval_path ${PROJECT_DIR}/data/pku10k_pref_eval.jsonl \
 --scored_aux_path ${PROJECT_DIR}/data/scored_aux_llama2_7b.jsonl \
 --pku_pref_eval_path ${PROJECT_DIR}/data/test.jsonl \
 --output_dir ${PROJECT_DIR}/output/pku10k_llama2_7b_deberta_v3_large \
 --max_hh_train_samples 9500 \
 --max_hh_eval_samples 500 \
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



echo "=========================================="
echo "PKU10K DeBERTa-v2-xlarge Training"
echo "=========================================="


python -m src.train_extracted_rm_two_stage \
 --student_model_path /mnt/model_data/models/deberta-v2-xlarge \
 --hh_pref_train_path ${PROJECT_DIR}/data/pku10k_pref_train.jsonl \
 --hh_pref_eval_path ${PROJECT_DIR}/data/pku10k_pref_eval.jsonl \
 --scored_aux_path ${PROJECT_DIR}/data/scored_aux_llama2_7b.jsonl \
 --pku_pref_eval_path ${PROJECT_DIR}/data/test.jsonl \
 --output_dir ${PROJECT_DIR}/output/pku10k_llama2_7b_deberta_v2_xlarge \
 --max_hh_train_samples 9500 \
 --max_hh_eval_samples 500 \
 --max_aux_samples 5000 \
 --max_pku_eval_samples 1000 \
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


TARGET_BASE=/home/vipuser/Desktop/model/llama2-7b
TARGET_ADAPTER=${PROJECT_DIR}/output/target_rm_llama2_7b_lora_full_margin5_e2


####################################
# RoBERTa PKU10K Diff
####################################

python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model ${TARGET_BASE} \
 --target_adapter_path ${TARGET_ADAPTER} \
 --substitute_model_path ${PROJECT_DIR}/output/pku10k_exp3_llama2_7b_to_roberta \
 --train_path ${PROJECT_DIR}/data/train.jsonl \
 --test_path ${PROJECT_DIR}/data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir ${PROJECT_DIR}/output/diff_pku10k_roberta \
 --max_length 512 \
 --target_batch_size 1 \
 --substitute_batch_size 16 \
 --bf16



####################################
# DeBERTa-v3 PKU10K Diff
####################################

python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model ${TARGET_BASE} \
 --target_adapter_path ${TARGET_ADAPTER} \
 --substitute_model_path ${PROJECT_DIR}/output/pku10k_llama2_7b_deberta_v3_large \
 --train_path ${PROJECT_DIR}/data/train.jsonl \
 --test_path ${PROJECT_DIR}/data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir ${PROJECT_DIR}/output/diff_pku10k_deberta_v3_large \
 --max_length 512 \
 --target_batch_size 1 \
 --substitute_batch_size 16 \
 --bf16



####################################
# DeBERTa-v2 PKU10K Diff
####################################

python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model ${TARGET_BASE} \
 --target_adapter_path ${TARGET_ADAPTER} \
 --substitute_model_path ${PROJECT_DIR}/output/pku10k_llama2_7b_deberta_v2_xlarge \
 --train_path ${PROJECT_DIR}/data/train.jsonl \
 --test_path ${PROJECT_DIR}/data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir ${PROJECT_DIR}/output/diff_pku10k_deberta_v2_xlarge \
 --max_length 384 \
 --target_batch_size 1 \
 --substitute_batch_size 8 \
 --bf16



echo "=========================================="
echo "ALL PKU10K EXP FINISHED"
echo "=========================================="
