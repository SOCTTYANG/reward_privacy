from __future__ import annotations

import json
import os
import random


INPUT_TRAIN = "/home/vipuser/Desktop/yang-safe-rlhf/data/hh_rlhf_triplet_train.jsonl"
INPUT_TEST = "/home/vipuser/Desktop/yang-safe-rlhf/data/hh_rlhf_triplet_test.jsonl"

OUTPUT_DIR = "/home/vipuser/Desktop/yang-safe-rlhf/data"

OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "hh_rlhf_triplet_train_2000.jsonl")
OUTPUT_EVAL = os.path.join(OUTPUT_DIR, "hh_rlhf_triplet_eval_200.jsonl")

TRAIN_SIZE = 2000
EVAL_SIZE = 200
SEED = 42


def read_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_jsonl(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(SEED)

    train_data = read_jsonl(INPUT_TRAIN)
    test_data = read_jsonl(INPUT_TEST)

    if len(train_data) < TRAIN_SIZE:
        raise ValueError(f"Train data only has {len(train_data)} samples, less than {TRAIN_SIZE}")

    if len(test_data) < EVAL_SIZE:
        raise ValueError(f"Test data only has {len(test_data)} samples, less than {EVAL_SIZE}")

    sampled_train = random.sample(train_data, TRAIN_SIZE)
    sampled_eval = random.sample(test_data, EVAL_SIZE)

    write_jsonl(sampled_train, OUTPUT_TRAIN)
    write_jsonl(sampled_eval, OUTPUT_EVAL)

    print(f"[OK] train -> {OUTPUT_TRAIN} ({len(sampled_train)} samples)")
    print(f"[OK] eval  -> {OUTPUT_EVAL} ({len(sampled_eval)} samples)")


if __name__ == "__main__":
    main()
