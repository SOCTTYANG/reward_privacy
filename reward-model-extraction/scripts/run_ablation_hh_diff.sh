#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=/mnt/model_data/projects/bai-rm-extraction-exp

cd ${PROJECT_DIR}


run_diff(){

TARGET_ADAPTER=$1
STUDENT=$2
NAME=$3


echo "=============================================="
echo ${NAME}
echo "=============================================="


python scripts/eval_target_vs_substitute_diff.py \
 --target_base_model $4 \
 --target_adapter_path ${TARGET_ADAPTER} \
 --substitute_model_path ${STUDENT} \
 --train_path data/train.jsonl \
 --test_path data/test.jsonl \
 --sample_train 500 \
 --sample_test 500 \
 --output_dir output/${NAME} \
 --max_length 512 \
 --target_batch_size 1 \
 --substitute_batch_size 16 \
 --bf16

}


# ==========================
# Llama2-13B
# ==========================

run_diff \
output/target_rm_llama2_13b_full_margin5_e1 \
output/ablation_hh_llama2_13b_roberta_only_pretrain \
diff_ablation_hh_llama2_13b_only_pretrain \
/mnt/model_data/models/llama2-13b-hf

run_diff \
output/target_rm_llama2_13b_full_margin5_e1 \
output/ablation_hh_llama2_13b_roberta_only_distill \
diff_ablation_hh_llama2_13b_only_distill \
/mnt/model_data/models/llama2-13b-hf



# ==========================
# Llama3.2-3B
# ==========================

run_diff \
output/target_rm_llama32_3b_full_margin5_e1 \
output/ablation_hh_llama32_3b_roberta_only_pretrain \
diff_ablation_hh_llama32_3b_only_pretrain \
/mnt/model_data/models/llama32-3b


run_diff \
output/target_rm_llama32_3b_full_margin5_e1 \
output/ablation_hh_llama32_3b_roberta_only_distill \
diff_ablation_hh_llama32_3b_only_distill \
/mnt/model_data/models/llama32-3b



# ==========================
# Mistral-7B
# ==========================

run_diff \
output/target_rm_mistral_7b_full_margin5_e1 \
output/ablation_hh_mistral_7b_roberta_only_pretrain \
diff_ablation_hh_mistral_7b_only_pretrain \
/mnt/model_data/models/mistral-7b-v0.1


run_diff \
output/target_rm_mistral_7b_full_margin5_e1 \
output/ablation_hh_mistral_7b_roberta_only_distill \
diff_ablation_hh_mistral_7b_only_distill \
/mnt/model_data/models/mistral-7b-v0.1



# ==========================
# Qwen3-8B
# ==========================

run_diff \
output/target_rm_qwen3_8b_full_margin5_e1 \
output/ablation_hh_qwen3_8b_roberta_only_pretrain \
diff_ablation_hh_qwen3_8b_only_pretrain \
/mnt/model_data/models/qwen3-8b


run_diff \
output/target_rm_qwen3_8b_full_margin5_e1 \
output/ablation_hh_qwen3_8b_roberta_only_distill \
diff_ablation_hh_qwen3_8b_only_distill \
/mnt/model_data/models/qwen3-8b


echo "=============================================="
echo "ALL ABLATION DIFF FINISHED"
echo "=============================================="
