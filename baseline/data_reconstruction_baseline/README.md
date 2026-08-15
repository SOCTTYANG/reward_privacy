# Data Reconstruction Baseline

This directory contains the baseline used for comparison with the paper's data-reconstruction method.

```text
data_reconstruction_baseline/
├── README.md
└── prompt_engineering_reward/
    ├── README.md
    ├── baseline_components.py
    ├── run_baseline.py
    └── run_llama2_7b.sh
```

The baseline iteratively prompts a generator using reward feedback. It does not fine-tune the reconstruction model.

Run it from the repository root:

```bash
MAX_SAMPLES=20 \
bash baseline/data_reconstruction_baseline/prompt_engineering_reward/run_llama2_7b.sh
```
