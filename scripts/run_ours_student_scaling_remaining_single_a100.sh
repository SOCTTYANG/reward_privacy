#!/bin/bash

set -e

PROJECT=/mnt/model_data/projects/bai-rm-extraction-exp

cd $PROJECT

mkdir -p logs


run_one_target(){

TARGET_NAME=$1
TARGET_BASE=$2
TARGET_ADAPTER=$3
AUX_DATA=$4


echo "======================================"
echo "Running Target: $TARGET_NAME"
echo "======================================"


########################################
# DeBERTa-v3-large
########################################

echo "======== Train DeBERTa-v3-large ========"


CUDA_VISIBLE_DEVICES=0 python -m src.train_extracted_rm_two_stage \
 --student_model_path /mnt/model_data/models/deberta-v3-large \
 --hh_pref_train_path data/hh_pref_train.jsonl \
 --hh_pref_eval_path data/hh_pref_test.jsonl \
 --scored_aux_path $AUX_DATA \
 --pku_pref_eval_path data/test.jsonl \
 --output_dir output/ours_${TARGET_NAME}_deberta_v3_large \
 --max_hh_train_samples 5000 \
 --max_hh_eval_samples 1000 \
 --max_aux_samples 5000 \
 --max_pku_eval_samples 1000 \
 --aux_train_ratio 0.9 \
 --pref_epochs 1 \
 --distill_epochs 1 \
 --pref_batch_size 2 \
 --distill_batch_size 2 \
 --eval_batch_size 16 \
 --max_length 384 \
 --pref_lr 2e-5 \
 --distill_lr 1e-5 \
 > logs/${TARGET_NAME}_deberta_v3_train.log 2>&1



echo "======== Diff DeBERTa-v3-large ========"


CUDA_VISIBLE_DEVICES=0 python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model $TARGET_BASE \
 --target_adapter_path $TARGET_ADAPTER \
 --substitute_model_path output/ours_${TARGET_NAME}_deberta_v3_large \
 --train_path data/train.jsonl \
 --test_path data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir output/diff_ours_${TARGET_NAME}_deberta_v3_large \
 --max_length 512 \
 --bf16



########################################
# DeBERTa-v2-xlarge
########################################


echo "======== Train DeBERTa-v2-xlarge ========"


CUDA_VISIBLE_DEVICES=0 python -m src.train_extracted_rm_two_stage \
 --student_model_path /mnt/model_data/models/deberta-v2-xlarge \
 --hh_pref_train_path data/hh_pref_train.jsonl \
 --hh_pref_eval_path data/hh_pref_test.jsonl \
 --scored_aux_path $AUX_DATA \
 --pku_pref_eval_path data/test.jsonl \
 --output_dir output/ours_${TARGET_NAME}_deberta_v2_xlarge \
 --max_hh_train_samples 5000 \
 --max_hh_eval_samples 1000 \
 --max_aux_samples 5000 \
 --max_pku_eval_samples 1000 \
 --aux_train_ratio 0.9 \
 --pref_epochs 1 \
 --distill_epochs 1 \
 --pref_batch_size 2 \
 --distill_batch_size 2 \
 --eval_batch_size 16 \
 --max_length 384 \
 --pref_lr 2e-5 \
 --distill_lr 1e-5 \
 > logs/${TARGET_NAME}_deberta_v2_train.log 2>&1



echo "======== Diff DeBERTa-v2-xlarge ========"


CUDA_VISIBLE_DEVICES=0 python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model $TARGET_BASE \
 --target_adapter_path $TARGET_ADAPTER \
 --substitute_model_path output/ours_${TARGET_NAME}_deberta_v2_xlarge \
 --train_path data/train.jsonl \
 --test_path data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir output/diff_ours_${TARGET_NAME}_deberta_v2_xlarge \
 --max_length 384 \
 --bf16


echo "Finished $TARGET_NAME"

}



########################################
# Remaining targets only
########################################


# Qwen3-8B

run_one_target \
qwen3_8b \
/mnt/model_data/models/qwen3-8b \
output/target_rm_qwen3_8b_full_margin5_e1 \
data/scored_aux_qwen3_8b_full_margin5_e1.jsonl



# Llama3.2-3B

run_one_target \
llama32_3b \
/mnt/model_data/models/llama32-3b \
output/target_rm_llama32_3b_full_margin5_e1 \
data/scored_aux_llama32_3b_full_margin5_e1.jsonl



# Llama2-13B

run_one_target \
llama2_13b \
/mnt/model_data/models/llama2-13b-hf \
output/target_rm_llama2_13b_full_margin5_e1 \
data/scored_aux_llama2_13b_full_margin5_e1.jsonl



echo "======================================"
echo "ALL REMAINING TARGETS FINISHED"
echo "======================================"
