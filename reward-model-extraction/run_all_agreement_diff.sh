#!/bin/bash

set -e

echo "========== baseline2 =========="
CUDA_VISIBLE_DEVICES=0 bash scripts/run_baseline2_all_train_and_diff_seq.sh


echo "========== PKU10K =========="
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pku10k_v2_v3_and_diff.sh


echo "========== Ours remaining single A100 =========="
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ours_student_scaling_remaining_single_a100.sh


echo "========== Ablation HH =========="
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ablation_hh_diff.sh


echo "========== DONE =========="
