#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME=/mnt/model_data/cache/huggingface
export HF_DATASETS_CACHE=/mnt/model_data/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/mnt/model_data/cache/huggingface/transformers
export TORCH_HOME=/mnt/model_data/cache/torch
export TMPDIR=/mnt/model_data/tmp

mkdir -p ${HF_HOME} ${HF_DATASETS_CACHE} ${TRANSFORMERS_CACHE} ${TORCH_HOME} ${TMPDIR}

PROJECT_DIR=/mnt/bai_data/projects/bai-rm-extraction-exp

MODEL_PATH=$1
RUN_NAME=$2
EXP_ROBERTA=$3

TARGET_DIR=${PROJECT_DIR}/output/target_rm_${RUN_NAME}
SCORED_AUX=${PROJECT_DIR}/data/scored_aux_${RUN_NAME}.jsonl

cd ${PROJECT_DIR}

echo "======================================================"
echo "MODEL_PATH   = ${MODEL_PATH}"
echo "RUN_NAME     = ${RUN_NAME}"
echo "TARGET_DIR   = ${TARGET_DIR}"
echo "SCORED_AUX   = ${SCORED_AUX}"
echo "EXP_ROBERTA  = ${EXP_ROBERTA}"
echo "======================================================"

echo "======================================================"
echo "Step 1: Train target reward model"
echo "======================================================"

python -m src.train_target_rm_llama_lora \
  --model_name_or_path ${MODEL_PATH} \
  --train_path ${PROJECT_DIR}/data/train.jsonl \
  --eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${TARGET_DIR} \
  --max_train_samples 30000 \
  --max_eval_samples 1000 \
  --epochs 1 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_length 512 \
  --lr 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --grad_clip 1.0 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --margin 5.0 \
  --bf16 \
  --gradient_checkpointing \
  --seed 42

echo "======================================================"
echo "Step 2: Score Dolly auxiliary data with target RM"
echo "======================================================"

test -f ${TARGET_DIR}/adapter_config.json || { echo "[ERROR] adapter_config.json not found in ${TARGET_DIR}"; exit 1; }
test -f ${TARGET_DIR}/adapter_model.safetensors || { echo "[ERROR] adapter_model.safetensors not found in ${TARGET_DIR}"; exit 1; }

python -m src.score_aux_with_llama_lora \
  --base_model_path ${MODEL_PATH} \
  --lora_adapter_path ${TARGET_DIR} \
  --aux_path ${PROJECT_DIR}/data/aux_dolly.jsonl \
  --output_path ${SCORED_AUX} \
  --max_samples 5000 \
  --batch_size 1 \
  --max_length 512 \
  --bf16

echo "======================================================"
echo "Step 3: Eval target RM on PKU and HH"
echo "======================================================"

python -m src.eval_rm_misclassified \
  --base_model_path ${MODEL_PATH} \
  --adapter_path ${TARGET_DIR} \
  --eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${PROJECT_DIR}/output/error_analysis/${RUN_NAME}_on_pku \
  --max_samples 1000 \
  --batch_size 1 \
  --max_length 512 \
  --bf16

python -m src.eval_rm_misclassified \
  --base_model_path ${MODEL_PATH} \
  --adapter_path ${TARGET_DIR} \
  --eval_path ${PROJECT_DIR}/data/hh_pref_test.jsonl \
  --output_dir ${PROJECT_DIR}/output/error_analysis/${RUN_NAME}_on_hh \
  --max_samples 1000 \
  --batch_size 1 \
  --max_length 512 \
  --bf16

echo "======================================================"
echo "Step 4: ${EXP_ROBERTA}: Target RM -> RoBERTa-base substitute"
echo "======================================================"

python -m src.train_extracted_rm_two_stage \
  --student_model_path ${PROJECT_DIR}/models/roberta-base \
  --hh_pref_train_path ${PROJECT_DIR}/data/hh_pref_train.jsonl \
  --hh_pref_eval_path ${PROJECT_DIR}/data/hh_pref_test.jsonl \
  --scored_aux_path ${SCORED_AUX} \
  --pku_pref_eval_path ${PROJECT_DIR}/data/test.jsonl \
  --output_dir ${PROJECT_DIR}/output/extracted_rm_${EXP_ROBERTA} \
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

echo "======================================================"
echo "Finished: ${RUN_NAME}"
echo "Target dir: ${TARGET_DIR}"
echo "Scored aux: ${SCORED_AUX}"
echo "RoBERTa substitute: ${PROJECT_DIR}/output/extracted_rm_${EXP_ROBERTA}"
echo "======================================================"
