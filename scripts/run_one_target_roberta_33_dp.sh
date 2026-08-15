#!/usr/bin/env bash
# Full 3.3 pipeline for one epsilon=8 DP-trained causal-LM reward-model teacher.
# Do not enable nounset: the cluster's /etc/bashrc references optional variables.
set -eo pipefail

source ~/.bashrc
conda activate rm_extract

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROJECT_DIR="${PROJECT_DIR:-/mnt/model_data/projects/bai-rm-extraction-exp}"
# The clean RoBERTa-base checkpoint is retained on the original data mount.
STUDENT_MODEL="${STUDENT_MODEL:-/mnt/bai_data/projects/bai-rm-extraction-exp/models/roberta-base}"

MODEL_PATH="$1"
RUN_NAME="$2"
cd "${PROJECT_DIR}"
TARGET_DIR="${PROJECT_DIR}/output/target_rm_dp_eps8_${RUN_NAME}"
SCORED_AUX="${PROJECT_DIR}/data/scored_aux_dp_eps8_${RUN_NAME}.jsonl"
STUDENT_DIR="${PROJECT_DIR}/output/extracted_rm_dp_eps8_${RUN_NAME}_to_roberta"
DIFF_DIR="${PROJECT_DIR}/output/diff_dp_eps8_${RUN_NAME}_to_roberta"

python -m src.train_target_rm_llama_lora_dp \
  --model_name_or_path "${MODEL_PATH}" --train_path "${PROJECT_DIR}/data/train.jsonl" \
  --eval_path "${PROJECT_DIR}/data/test.jsonl" --output_dir "${TARGET_DIR}" \
  --max_train_samples 30000 --max_eval_samples 1000 --epochs 1 --lot_size 16 \
  --max_length 512 --lr 1e-5 --weight_decay 0.01 --warmup_ratio 0.03 --margin 5.0 \
  --target_epsilon 8 --delta 1e-5 --max_grad_norm 1.0 --lora_r 16 --lora_alpha 32 \
  --lora_dropout 0.05 --lora_target_modules q_proj,k_proj,v_proj,o_proj --bf16 --gradient_checkpointing --seed 42

python -m src.score_aux_with_llama_lora \
  --base_model_path "${MODEL_PATH}" --lora_adapter_path "${TARGET_DIR}" \
  --aux_path "${PROJECT_DIR}/data/aux_dolly.jsonl" --output_path "${SCORED_AUX}" \
  --target_model_name "dp_eps8_${RUN_NAME}" --max_samples 5000 --batch_size 1 --max_length 512 --bf16

python -m src.train_extracted_rm_two_stage \
  --student_model_path "${STUDENT_MODEL}" --hh_pref_train_path "${PROJECT_DIR}/data/hh_pref_train.jsonl" \
  --hh_pref_eval_path "${PROJECT_DIR}/data/hh_pref_test.jsonl" --scored_aux_path "${SCORED_AUX}" \
  --pku_pref_eval_path "${PROJECT_DIR}/data/test.jsonl" --output_dir "${STUDENT_DIR}" \
  --max_hh_train_samples 5000 --max_hh_eval_samples 1000 --max_aux_samples 5000 --max_pku_eval_samples 1000 \
  --aux_train_ratio 0.9 --pref_epochs 1 --distill_epochs 1 --pref_batch_size 8 --distill_batch_size 8 \
  --eval_batch_size 16 --max_length 512 --pref_lr 2e-5 --distill_lr 1e-5 --weight_decay 0.01 \
  --warmup_ratio 0.03 --grad_clip 1.0 --seed 42

python scripts/eval_target_vs_substitute_diff.py \
  --target_base_model "${MODEL_PATH}" --target_adapter_path "${TARGET_DIR}" --substitute_model_path "${STUDENT_DIR}" \
  --train_path "${PROJECT_DIR}/data/train.jsonl" --test_path "${PROJECT_DIR}/data/test.jsonl" \
  --sample_train 500 --sample_test 500 --output_dir "${DIFF_DIR}" --max_length 512 \
  --target_batch_size 1 --substitute_batch_size 16 --bf16

echo "[OK] DP epsilon=8 3.3 pipeline completed: ${RUN_NAME}"
