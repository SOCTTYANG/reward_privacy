# Reward Model Extraction

This directory implements the model-extraction procedure in Section 4.3:

1. Train a surrogate reward model on `attacker-preference-dataset` with the
   pairwise ranking objective.
2. Query the target reward model on `attacker-auxiliary-dataset` to construct
   scored auxiliary records.
3. Fine-tune the surrogate by regressing on the target scores.

The `scripts/query_budget_ablation/` launchers repeat this procedure with 50%
and 25% query budgets. The `scripts/open_source_reward_models/` launchers
apply the same workflow to a `public-reward-model`. The latter also retains an
explicitly labeled joint-distillation experiment variant.

All paths are placeholders. Set the relevant environment variables or command
line paths to your local `address` before running an experiment. Datasets,
checkpoints, logs, and generated outputs are intentionally excluded.

## Data format

Preference records are JSONL objects with `prompt`, `chosen`, and `rejected`
fields. Auxiliary records use `prompt` and `response`; after querying, they
also contain `target_score`.
