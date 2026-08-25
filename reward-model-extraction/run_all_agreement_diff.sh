#!/bin/bash

set -e

echo "========== baseline2 =========="
CUDA_VISIBLE_DEVICES=0 bash scripts/run_baseline2_all_train_and_diff_seq.sh


echo "========== Defender Evaluation10K =========="
CUDA_VISIBLE_DEVICES=0 bash scripts/run_defender_evaluation_v2_v3_and_diff.sh


echo "========== Ours remaining single A100 =========="
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ours_student_scaling_remaining_single_a100.sh


echo "========== Ablation Attacker Preference =========="
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ablation_attacker_preference_diff.sh


echo "========== DONE =========="
