#!/usr/bin/env bash
set -e

source ~/.bashrc
conda activate rm_extract

export HF_HOME=/mnt/bai_data/cache/huggingface
export HF_DATASETS_CACHE=/mnt/bai_data/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/mnt/bai_data/cache/huggingface/transformers
export TORCH_HOME=/mnt/bai_data/cache/torch
export TMPDIR=/mnt/bai_data/tmp

# 新版 huggingface hub 推荐这个
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p ${HF_HOME}
mkdir -p ${HF_DATASETS_CACHE}
mkdir -p ${TRANSFORMERS_CACHE}
mkdir -p ${TORCH_HOME}
mkdir -p ${TMPDIR}
mkdir -p /mnt/bai_data/models
mkdir -p /mnt/bai_data/projects/bai-rm-extraction-exp/logs

echo "======================================================"
echo "Check Hugging Face login"
echo "======================================================"
hf auth whoami || true

echo "======================================================"
echo "Download LLaMA2-13B"
echo "======================================================"
hf download meta-llama/Llama-2-13b-hf \
  --local-dir /mnt/bai_data/models/llama2-13b-hf \
  2>&1 | tee /mnt/bai_data/projects/bai-rm-extraction-exp/logs/download_llama2_13b.log

echo "======================================================"
echo "Download LLaMA3.2-3B"
echo "======================================================"
hf download meta-llama/Llama-3.2-3B \
  --local-dir /mnt/bai_data/models/llama32-3b \
  2>&1 | tee /mnt/bai_data/projects/bai-rm-extraction-exp/logs/download_llama32_3b.log

echo "======================================================"
echo "Download Qwen3-8B"
echo "======================================================"
hf download Qwen/Qwen3-8B \
  --local-dir /mnt/bai_data/models/qwen3-8b \
  2>&1 | tee /mnt/bai_data/projects/bai-rm-extraction-exp/logs/download_qwen3_8b.log

echo "======================================================"
echo "Download Mistral-7B"
echo "======================================================"
hf download mistralai/Mistral-7B-v0.1 \
  --local-dir /mnt/bai_data/models/mistral-7b-v0.1 \
  2>&1 | tee /mnt/bai_data/projects/bai-rm-extraction-exp/logs/download_mistral_7b.log

echo "======================================================"
echo "All downloads finished."
echo "Model directories:"
echo "======================================================"

du -h --max-depth=1 /mnt/bai_data/models | sort -h

echo "======================================================"
echo "Done."
echo "======================================================"
