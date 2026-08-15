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

## Equation-faithful implementation

The default implementation follows the paper equations directly:

- `A(y_i) = r_i - mean_j(r_j)` with no additional advantage normalization.
- `log pi(y|x)` is the sum of response-token log-probabilities, so `rho(y)` is the complete-sequence probability ratio.
- The reference policy is reset to the current policy at the beginning of every PPO step and remains frozen inside that step.
- The update uses plain SGD, implementing `phi <- phi - eta * grad_phi L_f`; there is no AdamW, weight decay, or gradient clipping.
- Policy updates and gradient norms cover all policy parameters, rather than only LoRA parameters.
- The membership score is exactly `I = lambda1 * (r_plus - r_minus) - lambda2 * ||g_phi||_2`.
- `--delta` is required and must be chosen on an independent calibration set. Test labels are never used to flip score direction or select the final decision threshold.

This formulation requires substantially more GPU memory than a LoRA approximation because both the current and reference policies are full models.

## Ablation experiments

The two ablations are kept together under `ablation_experiments/`:

### 1. Reward-Gap-Only Ablation

Uses only the reward-gap signal and skips the PPO/policy-update stage. Run:

```bash
bash ablation_experiments/reward_gap_only/run_mia_reward_gap_ablation_5models.sh
```

Set `DELTA` to a threshold obtained from an independent calibration set before running.

### 2. PPO-Gradient-Only Ablation

Uses only the PPO gradient-norm signal and removes the reward-gap contribution from the final score. Run:

```bash
bash ablation_experiments/ppo_gradient_only/run_mia_ppo_gradient_only_5models_cuda0.sh
```

This launcher also requires a calibrated `DELTA` environment variable.

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
