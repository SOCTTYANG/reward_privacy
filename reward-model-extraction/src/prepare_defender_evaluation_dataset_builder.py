"""Build safety preference pairs from DefenderEvaluationDataset's per-response labels.

For each prompt, a safe refusal (chosen) is paired with a harmful compliance
(rejected).  The labels are used only while constructing the evaluation file;
the resulting JSONL contains just prompt/chosen/rejected fields.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_id",
        required=True,
        help="Dataset identifier for defender-evaluation-dataset.",
    )
    parser.add_argument(
        "--dataset_config",
        default=None,
        help="Optional configuration name for defender-evaluation-dataset.",
    )
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    dataset = load_dataset(args.dataset_id, args.dataset_config, split=args.dataset_split)
    by_prompt = defaultdict(list)
    for row in dataset:
        by_prompt[row["prompt"]].append(row)

    pairs = []
    for prompt, rows in by_prompt.items():
        safe_refusals = [
            row for row in rows
            if row["response_refusal_label"] == "refusal"
            and row["response_harm_label"] == "unharmful"
        ]
        harmful_compliances = [
            row for row in rows
            if row["response_refusal_label"] == "compliance"
            and row["response_harm_label"] == "harmful"
        ]
        if safe_refusals and harmful_compliances:
            pairs.append({
                "prompt": prompt,
                "chosen": safe_refusals[0]["response"],
                "rejected": harmful_compliances[0]["response"],
            })

    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[:args.max_samples]
    if not pairs:
        raise ValueError("No defender-evaluation preference pairs were constructed.")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {args.output_path}, n={len(pairs)}")


if __name__ == "__main__":
    main(parse_args())
