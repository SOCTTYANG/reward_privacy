# Reward Model Extraction Experiments

This repository contains the implementation and experiment launch scripts for
reward-model extraction and preference-model training experiments. The code
supports training target reward models, scoring auxiliary preference data with
target models, and training extracted reward models using two-stage objectives.

## Repository layout

- `src/`: training, data preparation, scoring, and metric implementations.
- `scripts/`: shell launchers for the experiments and evaluations.
  - `scripts/ablation/query_budget/`: 50% and 25% query-budget ablations.
  - `scripts/open_source_reward_models/`: attacks against public reward models.
- `tests/`: unit tests for extraction metrics.
- `check_saferlhf_overlap.py`: utility for dataset-overlap checks.

Generated datasets, model checkpoints, logs, and experiment outputs are not
included. They are ignored by Git to keep the repository lightweight and to
avoid redistributing source data or model artifacts.

## Setup

Use Python 3.10+ and a CUDA-enabled PyTorch installation appropriate for your
hardware. Then install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA version before
or instead of the generic `torch` dependency above.

## Data format

The preference-training scripts expect JSONL records with these fields:

```json
{"prompt": "...", "chosen": "preferred response", "rejected": "other response"}
```

Place local data under `data/` (or pass absolute paths through the command-line
arguments). Data files are deliberately excluded from version control.

## Running experiments

Each launcher in `scripts/` defines `PROJECT_DIR`, model locations, data paths,
and output paths. Set `PROJECT_DIR` to the path of your clone and adjust the
model/data paths for your environment before running a script. For example:

```bash
bash scripts/01_train_target_rm_roberta.sh
bash scripts/03_train_extracted_rm_two_stage_exp1.sh
```

Alternatively, invoke modules directly; see the argument definitions in the
corresponding files under `src/`.

### Query-budget ablation

The launchers under `scripts/ablation/query_budget/` reduce the original 5,000
distillation queries to 2,500 (50%) and 1,250 (25%), then train and evaluate
the student model with the same data split and random seed.

### Public reward-model extraction

The launchers under `scripts/open_source_reward_models/` apply the extraction
pipeline to public reward models. The Skywork workflow uses the target
tokenizer's chat template, queries the target model, and trains a RoBERTa
substitute. For example:

```bash
SKYWORK_MODEL=/path/to/Skywork-Reward-Llama-3.1-8B \
STUDENT_MODEL=/path/to/roberta-base \
bash scripts/open_source_reward_models/run_skywork_reward_extraction.sh
```

## Tests

```bash
python -m pytest -q
```

## Reproducibility notes

The launchers set random seeds and CUDA environment variables. Exact results
can still vary with GPU hardware, CUDA/cuDNN versions, and model/data releases.
