# Ablation Experiments

This directory contains the two named ablation studies used to isolate the contribution of each membership-inference signal.

1. **Reward-Gap-Only Ablation** (`reward_gap_only/`): retains the reward-gap signal and removes the PPO/policy-update contribution.
2. **PPO-Gradient-Only Ablation** (`ppo_gradient_only/`): retains the PPO gradient-norm signal and removes the reward-gap contribution.

Each subdirectory contains its experiment launcher. The reward-gap-only variant also has a dedicated evaluator; the PPO-gradient-only variant uses the core evaluator with `--score_signal ppo_gradient_only`.

Both launchers require `DELTA`, a fixed threshold selected using an independent calibration set. They do not infer the final decision threshold from test labels or reverse the score direction after seeing test labels.
