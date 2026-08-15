# Reward Privacy

This repository groups privacy attacks, reward-model extraction experiments,
baselines, and defenses.

## Layout

- `reward-model-extraction/`: core reward-model extraction implementation,
  data preparation, scoring, evaluation, and tests.
- `baseline/`: prior and comparative baselines. The reward-model extraction
  baselines are in `baseline/reward-model-extraction/`.
- `defense/`: defense implementations. The DP-SGD reward-model training code
  and its launch scripts are in `defense/reward-model-extraction/`.
- `membership_inference/` and `data_reconstruction/`: existing privacy attack
  implementations.

Datasets, checkpoints, logs, and generated experiment outputs are excluded from
version control.
