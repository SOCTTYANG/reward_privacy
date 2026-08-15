#!/usr/bin/env bash
set -e

cd /mnt/bai_data/projects/bai-rm-extraction-exp

echo "======================================================"
echo "Exp8: LLaMA3.2-3B target RM -> RoBERTa-base"
echo "======================================================"

bash scripts/run_one_target_roberta_33.sh \
  /mnt/model_data/models/llama32-3b \
  llama32_3b_full_margin5_e1 \
  exp8_llama32_3b_to_roberta

echo "======================================================"
echo "Exp10: Qwen3-8B target RM -> RoBERTa-base"
echo "======================================================"

bash scripts/run_one_target_roberta_33.sh \
  /mnt/model_data/models/qwen3-8b \
  qwen3_8b_full_margin5_e1 \
  exp10_qwen3_8b_to_roberta

echo "======================================================"
echo "Exp12: Mistral-7B target RM -> RoBERTa-base"
echo "======================================================"

bash scripts/run_one_target_roberta_33.sh \
  /mnt/model_data/models/mistral-7b-v0.1 \
  mistral_7b_full_margin5_e1 \
  exp12_mistral_7b_to_roberta

echo "======================================================"
echo "Exp6: LLaMA2-13B target RM -> RoBERTa-base"
echo "======================================================"

bash scripts/run_one_target_roberta_33.sh \
  /mnt/model_data/models/llama2-13b-hf \
  llama2_13b_full_margin5_e1 \
  exp6_llama2_13b_to_roberta

echo "======================================================"
echo "All selected RoBERTa extraction experiments finished."
echo "======================================================"
