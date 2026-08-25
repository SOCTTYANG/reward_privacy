#!/bin/bash

set -e


cd /path/to/code


export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


mkdir -p logs


OUT=output


echo "=============================="
echo "Train DeBERTa-v3-large"
echo "=============================="


python -m src.train_extracted_rm_two_stage \
 --student_model_path /path/to/models/deberta-v3-large \
 --attacker_preference_dataset_train_path data/attacker_preference_dataset_train.jsonl \
 --attacker_preference_dataset_eval_path data/attacker_preference_dataset_test.jsonl \
 --scored_aux_path data/scored_aux_llama2_7b_margin5_e2.jsonl \
 --defender_eval_eval_path data/test.jsonl \
 --output_dir $OUT/ours_llama2_7b_deberta_v3_large \
 --max_attacker_preference_train_samples 5000 \
 --max_attacker_preference_eval_samples 1000 \
 --max_aux_samples 5000 \
 --max_defender_evaluation_eval_samples 1000 \
 --aux_train_ratio 0.9 \
 --pref_epochs 1 \
 --distill_epochs 1 \
 --pref_batch_size 2 \
 --distill_batch_size 2 \
 --eval_batch_size 4 \
 --max_length 512 \
 --pref_lr 2e-5 \
 --distill_lr 1e-5 \
 > logs/ours_deberta_v3_train.log 2>&1



echo "=============================="
echo "Diff DeBERTa-v3"
echo "=============================="


python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model /path/to/target-model/llama2-7b \
 --target_adapter_path output/target_rm_llama2_7b_lora_full_margin5_e2 \
 --substitute_model_path output/ours_llama2_7b_deberta_v3_large \
 --train_path data/train.jsonl \
 --test_path data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir output/diff_ours_deberta_v3_large \
 --max_length 512 \
 --target_batch_size 1 \
 --substitute_batch_size 16 \
 --bf16



echo "=============================="
echo "Train DeBERTa-v2-xlarge"
echo "=============================="


python -m src.train_extracted_rm_two_stage \
 --student_model_path /path/to/models/deberta-v2-xlarge \
 --attacker_preference_dataset_train_path data/attacker_preference_dataset_train.jsonl \
 --attacker_preference_dataset_eval_path data/attacker_preference_dataset_test.jsonl \
 --scored_aux_path data/scored_aux_llama2_7b_margin5_e2.jsonl \
 --defender_eval_eval_path data/test.jsonl \
 --output_dir $OUT/ours_llama2_7b_deberta_v2_xlarge \
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
 --max_length 384 \
 --pref_lr 2e-5 \
 --distill_lr 1e-5 \
 > logs/ours_deberta_v2_train.log 2>&1



echo "=============================="
echo "Diff DeBERTa-v2"
echo "=============================="


python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model /path/to/target-model/llama2-7b \
 --target_adapter_path output/target_rm_llama2_7b_lora_full_margin5_e2 \
 --substitute_model_path output/ours_llama2_7b_deberta_v2_xlarge \
 --train_path data/train.jsonl \
 --test_path data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir output/diff_ours_deberta_v2_xlarge \
 --max_length 384 \
 --target_batch_size 1 \
 --substitute_batch_size 8 \
 --bf16


echo "=============================="
echo "ALL DONE"
echo "=============================="
