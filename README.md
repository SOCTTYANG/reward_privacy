# Reward Model Extraction Experiments

This repository contains the implementation and experiment launch scripts for
reward-model extraction and preference-model training experiments. The code
supports training target reward models, scoring auxiliary preference data with
target models, and training extracted reward models using two-stage objectives.

## Repository layout

- `src/`: training, data preparation, scoring, and metric implementations.
- `scripts/`: shell launchers for the experiments and evaluations.
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

## Tests

```bash
python -m pytest -q
```

## Reproducibility notes

The launchers set random seeds and CUDA environment variables. Exact results
can still vary with GPU hardware, CUDA/cuDNN versions, and model/data releases.
