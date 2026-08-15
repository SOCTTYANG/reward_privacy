# Data Reconstruction

This directory contains only the implementation required for the paper's three-step data-reconstruction method. Analysis utilities, qualitative samples, unrelated experiment launchers, archives, and ablation studies are intentionally excluded.

## Method

Given a preference dataset

```text
D_pref = {(x_i, y_i_plus, y_i_minus)}
```

the implementation performs:

1. Fine-tune the reconstruction model with

   ```text
   L_s = lambda_MLE * L_MLE + lambda_cos * L_cos
   L_cos = -cos(E(y_hat_minus), E(y_minus))
   ```

   During every training step, `y_hat_minus` is sampled from the model. Because discrete sampling is non-differentiable, the implementation uses a REINFORCE score-function estimator to propagate the cosine objective to the generator.

2. Generate `K` candidate dispreferred responses from `(x, y_plus)`.

3. Compute `r_hat_i = R(x, y_hat_i_minus)`, obtain

   ```text
   j = argmin_i R(x, y_hat_i_minus)
   y_hat_star_minus = y_hat_j_minus
   ```

   and return the lowest-reward candidate.

## Included files

```text
data_reconstruction/
├── safe_rlhf/reconstruction/       # Step 1 dataset, model, loss, and trainer
├── safe_rlhf/dual_reward/           # minimal reward-model code needed by Step 3
├── stage2_generate_candidates_k3.py # Step 2 candidate generation
├── stage3_select_lowest_reward.py   # Step 3 reward scoring and argmin selection
├── data_processing/                 # construction of D_pref triplets
├── evaluation/eval_bleu_cosine.py   # BLEU-1 and cosine evaluation
├── configs/                         # DeepSpeed training configuration
└── scripts/run_reconstruction_5models.sh
```

## Installation

```bash
pip install -r data_reconstruction/requirements.txt
```

## Run

Set local model, reward-model, data, and output paths through the environment variables in the launcher, then run from the repository root:

```bash
bash data_reconstruction/scripts/run_reconstruction_5models.sh
```

For a small smoke test:

```bash
MAX_SAMPLES=2 METRIC_BATCH_SIZE=1 \
bash data_reconstruction/scripts/run_reconstruction_5models.sh
```
