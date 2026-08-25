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


def build_prompt(instruction: str, context: str) -> str:
    instruction = instruction.strip()
    context = context.strip()

    if context:
        return f"{instruction}\n\nContext:\n{context}"

    return instruction


def convert_one(example: Dict[str, Any]) -> Dict[str, str]:
    """
    attacker-auxiliary-dataset 15K 常见字段：
      - instruction
      - context
      - response
      - category

    我们转换成：
      {"prompt": ..., "response": ...}
    """

    if "instruction" not in example:
        raise KeyError(f"Missing instruction field. Keys: {list(example.keys())}")

    if "response" not in example:
        raise KeyError(f"Missing response field. Keys: {list(example.keys())}")

    instruction = str(example["instruction"])
    context = str(example.get("context", "") or "")
    response = str(example["response"])

    prompt = build_prompt(instruction, context)

    return {
        "prompt": prompt,
        "response": response,
        "category": str(example.get("category", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="attacker-auxiliary-dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--remove_empty",
        action="store_true",
    )

    args = parser.parse_args()

    print(f"[INFO] Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)

    print(ds)

    if args.split not in ds:
        raise KeyError(f"Split {args.split} not found. Available splits: {list(ds.keys())}")

    raw = ds[args.split]
    print("[INFO] Columns:", raw.column_names)

    rows = []

    for ex in tqdm(raw, desc="Converting attacker-auxiliary-dataset aux data"):
        item = convert_one(ex)

        if args.remove_empty:
            if not item["prompt"].strip():
                continue
            if not item["response"].strip():
                continue

        rows.append(item)

        if args.max_samples is not None and len(rows) >= args.max_samples:
            break

    save_jsonl(rows, args.output_path)

    print(f"[OK] Saved aux data to: {args.output_path}")
    print(f"[OK] Total samples: {len(rows)}")

    if rows:
        print("[INFO] First example:")
        print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()