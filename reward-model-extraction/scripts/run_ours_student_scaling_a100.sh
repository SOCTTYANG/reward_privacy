#!/bin/bash

set -e

PROJECT=/path/to/code

cd $PROJECT


mkdir -p logs


TARGET_BASE=/path/to/target-model/llama2-7b

TARGET_ADAPTER=/path/to/code/output/target_rm_llama2_7b_lora_full_margin5_e2

AUX_DATA=data/scored_aux_llama2_7b_margin5_e2.jsonl


OUT=/path/to/code/output



echo "===================================="
echo "Start training extracted RM"
echo "===================================="


########################################
# GPU0 DeBERTa-v3-large
########################################

CUDA_VISIBLE_DEVICES=0 python -m src.train_extracted_rm_two_stage \
 --student_model_path /path/to/models/deberta-v3-large \
 --attacker_preference_dataset_train_path data/attacker_preference_dataset_train.jsonl \
 --attacker_preference_dataset_eval_path data/attacker_preference_dataset_test.jsonl \
 --scored_aux_path $AUX_DATA \
 --defender_eval_eval_path data/test.jsonl \
 --output_dir $OUT/ours_llama2_7b_deberta_v3_large \
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
 > logs/ours_deberta_v3_train.log 2>&1 &


PID1=$!



########################################
# GPU1 DeBERTa-v2-xlarge
########################################

CUDA_VISIBLE_DEVICES=1 python -m src.train_extracted_rm_two_stage \
 --student_model_path /path/to/models/deberta-v2-xlarge \
 --attacker_preference_dataset_train_path data/attacker_preference_dataset_train.jsonl \
 --attacker_preference_dataset_eval_path data/attacker_preference_dataset_test.jsonl \
 --scored_aux_path $AUX_DATA \
 --defender_eval_eval_path data/test.jsonl \
 --output_dir $OUT/ours_llama2_7b_deberta_v2_xlarge \
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
 > logs/ours_deberta_v2_train.log 2>&1 &


PID2=$!



echo "Training started..."
echo "PID1=$PID1"
echo "PID2=$PID2"


########################################
# Wait training
########################################

wait $PID1
STATUS1=$?

wait $PID2
STATUS2=$?


if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then

    echo "===================================="
    echo "Training failed!"
    echo "Check logs:"
    echo "logs/ours_deberta_v3_train.log"
    echo "logs/ours_deberta_v2_train.log"
    echo "===================================="

    exit 1

fi


echo "===================================="
echo "Training finished"
echo "Start Diff evaluation"
echo "===================================="



########################################
# Diff DeBERTa-v3-large
########################################


CUDA_VISIBLE_DEVICES=0 python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model $TARGET_BASE \
 --target_adapter_path $TARGET_ADAPTER \
 --substitute_model_path $OUT/ours_llama2_7b_deberta_v3_large \
 --train_path data/train.jsonl \
 --test_path data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir $OUT/diff_ours_deberta_v3_large \
 --max_length 512 \
 --target_batch_size 1 \
 --substitute_batch_size 16 \
 --bf16



########################################
# Diff DeBERTa-v2-xlarge
########################################


CUDA_VISIBLE_DEVICES=1 python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model $TARGET_BASE \
 --target_adapter_path $TARGET_ADAPTER \
 --substitute_model_path $OUT/ours_llama2_7b_deberta_v2_xlarge \
 --train_path data/train.jsonl \
 --test_path data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir $OUT/diff_ours_deberta_v2_xlarge \
 --max_length 512 \
 --target_batch_size 1 \
 --substitute_batch_size 16 \
 --bf16



echo "===================================="
echo "All experiments finished!"
echo "===================================="
