#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=/path/to/code

cd ${PROJECT_DIR}


STUDENT=/path/to/code/models/roberta-base


run_exp(){

TEACHER=$1
AUX=$2
NAME=$3


echo "========================================="
echo ${NAME}
echo "========================================="


# Only Pretrain
python -m src.train_extracted_rm_two_stage \
 --student_model_path ${STUDENT} \
 --attacker_preference_dataset_train_path ${PROJECT_DIR}/data/attacker_preference_dataset_train.jsonl \
 --attacker_preference_dataset_eval_path ${PROJECT_DIR}/data/attacker_preference_dataset_test.jsonl \
 --scored_aux_path ${PROJECT_DIR}/data/${AUX} \
 --defender_eval_eval_path ${PROJECT_DIR}/data/test.jsonl \
 --output_dir ${PROJECT_DIR}/output/${NAME}_only_pretrain \
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
 --seed 42 \
 --skip_stage_b



# Only Distill
python -m src.train_extracted_rm_two_stage \
 --student_model_path ${STUDENT} \
 --attacker_preference_dataset_train_path ${PROJECT_DIR}/data/attacker_preference_dataset_train.jsonl \
 --attacker_preference_dataset_eval_path ${PROJECT_DIR}/data/attacker_preference_dataset_test.jsonl \
 --scored_aux_path ${PROJECT_DIR}/data/${AUX} \
 --defender_eval_eval_path ${PROJECT_DIR}/data/test.jsonl \
 --output_dir ${PROJECT_DIR}/output/${NAME}_only_distill \
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
 --seed 42 \
 --skip_stage_a

}



run_exp \
target_rm_llama2_13b_full_margin5_e1 \
scored_aux_llama2_13b_full_margin5_e1.jsonl \
ablation_attacker_preference_llama2_13b_roberta


run_exp \
target_rm_llama32_3b_full_margin5_e1 \
scored_aux_llama32_3b_full_margin5_e1.jsonl \
ablation_attacker_preference_llama32_3b_roberta


run_exp \
target_rm_mistral_7b_full_margin5_e1 \
scored_aux_mistral_7b_full_margin5_e1.jsonl \
ablation_attacker_preference_mistral_7b_roberta


run_exp \
target_rm_qwen3_8b_full_margin5_e1 \
scored_aux_qwen3_8b_full_margin5_e1.jsonl \
ablation_attacker_preference_qwen3_8b_roberta


echo "ALL ABLATIONS DONE"
