from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

from datasets import load_dataset


DATASET_NAME = "Anthropic/hh-rlhf"
OUTPUT_DIR = "/home/vipuser/Desktop/yang-safe-rlhf/data"

TRAIN_OUT = os.path.join(OUTPUT_DIR, "hh_rlhf_triplet_train.jsonl")
TEST_OUT = os.path.join(OUTPUT_DIR, "hh_rlhf_triplet_test.jsonl")


def longest_common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def split_prompt_and_last_assistant_response(text: str) -> Tuple[str, str]:

    marker = "\n\nAssistant:"
    idx = text.rfind(marker)

    if idx == -1:
        marker = "Assistant:"
        idx = text.rfind(marker)
        if idx == -1:
            return "", text.strip()
        prompt = text[: idx + len(marker)].strip()
        response = text[idx + len(marker):].strip()
        return prompt, response

    prompt = text[: idx + len(marker)].strip()
    response = text[idx + len(marker):].strip()
    return prompt, response


def build_triplet_from_pair(chosen: str, rejected: str) -> Optional[Dict]:

    common = longest_common_prefix(chosen, rejected).rstrip()


    prompt_common, _ = split_prompt_and_last_assistant_response(common)


    chosen_prompt, chosen_resp = split_prompt_and_last_assistant_response(chosen)
    rejected_prompt, rejected_resp = split_prompt_and_last_assistant_response(rejected)


    x = prompt_common if prompt_common.strip() else chosen_prompt.strip()

    y_plus = chosen_resp.strip()
    y_minus = rejected_resp.strip()

    if not x or not y_plus or not y_minus:
        return None

    return {
        "x": x,
        "y_plus": y_plus,
        "y_minus": y_minus,
        "chosen_full": chosen,
        "rejected_full": rejected,
    }


def export_split(split_name: str, output_path: str) -> None:
    ds = load_dataset(DATASET_NAME, split=split_name)

    total = 0
    kept = 0
    dropped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            total += 1
            chosen = item["chosen"]
            rejected = item["rejected"]

            triplet = build_triplet_from_pair(chosen, rejected)
            if triplet is None:
                dropped += 1
                continue

            f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] Exported split={split_name}")
    print(f"     output={output_path}")
    print(f"     total={total}")
    print(f"     kept={kept}")
    print(f"     dropped={dropped}")


def preview_file(file_path: str, n: int = 3) -> None:
    print("\n" + "=" * 100)
    print(f"Preview: {file_path}")
    print("=" * 100)

    with open(file_path, "r", encoding="utf-8") as f:
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    export_split("train", TRAIN_OUT)
    export_split("test", TEST_OUT)

    preview_file(TRAIN_OUT, n=3)
    preview_file(TEST_OUT, n=3)


if __name__ == "__main__":
    main()
