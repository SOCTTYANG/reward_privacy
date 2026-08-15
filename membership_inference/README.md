# Membership Inference for Reward Models

This repository contains the implementation and experiment launchers for a membership-inference method based on reward gaps and PPO update signals.

## Repository layout

```text
method/                         Core four-stage implementation
ablation_experiments/           The two ablation experiments
  reward_gap_only/              Reward-Gap-Only Ablation
  ppo_gradient_only/            PPO-Gradient-Only Ablation
utilities/                      Metric collection and plotting scripts
results/                        Small result tables and figures
```

## Method pipeline

1. `method/3_1_target_data_decomposition.py`: prepare member/non-member target data.
2. `method/3_2_candidate_response_generation.py`: generate candidate responses and obtain reward scores.
3. `method/3_3_llm_update_ppo_full.py`: update the policy with PPO while keeping a frozen old policy.
4. `method/3_4_membership_inference.py`: combine the configured signals and evaluate membership inference.

## Ablation experiments

The two ablations are kept together under `ablation_experiments/`:

### 1. Reward-Gap-Only Ablation

Uses only the reward-gap signal and skips the PPO/policy-update stage. Run:

```bash
bash ablation_experiments/reward_gap_only/run_mia_reward_gap_ablation_5models.sh
```

### 2. PPO-Gradient-Only Ablation

Uses only the PPO gradient-norm signal and removes the reward-gap contribution from the final score. Run:

```bash
bash ablation_experiments/ppo_gradient_only/run_mia_ppo_gradient_only_5models_cuda0.sh
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The launchers contain machine-specific default model and output paths from the original experiments. Override them with environment variables such as `RM_BASE_MODEL`, `RM_ADAPTER`, `OUTPUT_ROOT`, `CUDA_DEVICE`, and `PYTHON_BIN` before running on another machine.

## Notes

- Large checkpoints, logs, caches, and generated experiment output directories are excluded through `.gitignore`.
- Existing small CSV summaries and figures are retained in `results/` for reference.
