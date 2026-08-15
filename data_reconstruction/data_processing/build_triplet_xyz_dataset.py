from __future__ import annotations

import json
import os
from typing import Dict

from datasets import load_dataset



DATASET_NAME = "Dahoas/full-hh-rlhf"

OUTPUT_DIR = "/home/vipuser/Desktop/yang-safe-rlhf/data/xyz_triplet_dataset"


def normalize_text(x):
    if x is None:
        return ""
    return str(x).strip()


def convert_item(item: Dict) -> Dict | None:
    prompt = normalize_text(item.get("prompt", ""))
    chosen = normalize_text(item.get("chosen", ""))
    rejected = normalize_text(item.get("rejected", ""))

    if not prompt or not chosen or not rejected:
        return None

    return {
        "x": prompt,
        "y_plus": chosen,
        "y_minus": rejected,
    }


def export_split(dataset, split_name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")

    kept = 0
    dropped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset[split_name]:
            row = convert_item(item)
            if row is None:
                dropped += 1
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] split={split_name}")
    print(f"     output={output_path}")
    print(f"     kept={kept}")
    print(f"     dropped={dropped}")


def preview_jsonl(path: str, n: int = 2):
    print("\n" + "=" * 100)
    print(f"Preview: {path}")
    print("=" * 100)

    if not os.path.exists(path):
        print("File not found.")
        return

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            print(f"\nSample #{i+1}")
            print("-" * 100)
            print("x:")
            print(obj["x"])
            print("\ny_plus:")
            print(obj["y_plus"])
            print("\ny_minus:")
            print(obj["y_minus"])
            print("-" * 100)


def main():
    dataset = load_dataset(DATASET_NAME)

    for split_name in dataset.keys():
        export_split(dataset, split_name)

    for split_name in dataset.keys():
        preview_jsonl(os.path.join(OUTPUT_DIR, f"{split_name}.jsonl"), n=2)


if __name__ == "__main__":
    main()
