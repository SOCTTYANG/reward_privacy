from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_prompt(example: Dict[str, Any]) -> str:
    for key in ["prompt", "question", "instruction", "input"]:
        if key in example and example[key] is not None:
            return str(example[key])
    raise KeyError(f"Cannot find prompt field. Keys: {list(example.keys())}")


def convert_one(example: Dict[str, Any], preference_type: str) -> Dict[str, str]:
    prompt = pick_prompt(example)

    if "response_0" not in example or "response_1" not in example:
        raise KeyError(f"Cannot find response_0/response_1. Keys: {list(example.keys())}")

    response_0 = str(example["response_0"])
    response_1 = str(example["response_1"])

    if preference_type == "helpful":
        pref_keys = ["better_response_id"]
    elif preference_type == "harmless":
        pref_keys = ["safer_response_id"]
    else:
        raise ValueError(f"Unknown preference_type: {preference_type}")

    pref_value = None
    for key in pref_keys:
        if key in example and example[key] is not None:
            pref_value = example[key]
            break

    if pref_value is None:
        raise KeyError(f"Cannot find preference key. Keys: {list(example.keys())}")

    chosen_id = int(pref_value)

    if chosen_id == 0:
        chosen = response_0
        rejected = response_1
    elif chosen_id == 1:
        chosen = response_1
        rejected = response_0
    else:
        raise ValueError(f"Preference id must be 0 or 1, got {chosen_id}")

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Defender Evaluation-Alignment/defender-evaluation-dataset-30K")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--preference_type", type=str, default="helpful", choices=["helpful", "harmless"])
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    print(f"[INFO] Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)

    print(ds)

    train_raw = ds["train"]
    test_raw = ds["test"] if "test" in ds else None

    print("[INFO] Train columns:", train_raw.column_names)
    if test_raw is not None:
        print("[INFO] Test columns:", test_raw.column_names)

    train_rows = []
    for ex in tqdm(train_raw, desc="Converting train"):
        train_rows.append(convert_one(ex, args.preference_type))
        if args.max_train_samples is not None and len(train_rows) >= args.max_train_samples:
            break

    if test_raw is not None:
        test_rows = []
        for ex in tqdm(test_raw, desc="Converting test"):
            test_rows.append(convert_one(ex, args.preference_type))
            if args.max_test_samples is not None and len(test_rows) >= args.max_test_samples:
                break
    else:
        split_idx = int(len(train_rows) * 0.9)
        test_rows = train_rows[split_idx:]
        train_rows = train_rows[:split_idx]

    train_path = os.path.join(args.output_dir, "train.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")

    save_jsonl(train_rows, train_path)
    save_jsonl(test_rows, test_path)

    print(f"[OK] Saved train: {train_path}, n={len(train_rows)}")
    print(f"[OK] Saved test:  {test_path}, n={len(test_rows)}")

    if train_rows:
        print("[INFO] First example:")
        print(json.dumps(train_rows[0], ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()