"""Flatten a disjoint slice of HH preference triples into teacher queries."""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def read_valid_pairs(path: str) -> List[Dict[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row: Dict[str, Any] = json.loads(line)
            if not {"prompt", "chosen", "rejected"} <= row.keys():
                raise KeyError("HH records must contain prompt, chosen, and rejected.")
            pair = {key: str(row[key]) for key in ("prompt", "chosen", "rejected")}
            if all(value.strip() for value in pair.values()):
                pairs.append(pair)
    return pairs


def main(args) -> None:
    pairs = read_valid_pairs(args.input_path)
    selected = pairs[args.skip_pairs : args.skip_pairs + args.max_pairs]
    if len(selected) != args.max_pairs:
        raise ValueError(
            f"Requested {args.max_pairs} disjoint pairs after skipping {args.skip_pairs}, "
            f"but only {len(selected)} are available from {len(pairs)} valid HH pairs."
        )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for source_index, pair in enumerate(selected, start=args.skip_pairs):
            # Do not carry the preference role into either the teacher query or
            # the distillation record.  The two responses are independent
            # (X, Y) examples after this point.
            for response in (pair["chosen"], pair["rejected"]):
                handle.write(json.dumps({
                    "prompt": pair["prompt"],
                    "response": response,
                    "source_pair_index": source_index,
                }, ensure_ascii=False) + "\n")
    print(f"[OK] selected {len(selected)} disjoint HH triples")
    print(f"[OK] wrote {2 * len(selected)} teacher-query pairs to {args.output_path}")
    print(f"[INFO] source pair indices: [{args.skip_pairs}, {args.skip_pairs + len(selected)})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--skip_pairs", type=int, required=True,
                        help="Number of valid HH triples reserved for Stage A.")
    parser.add_argument("--max_pairs", type=int, required=True,
                        help="Number of subsequent HH triples to flatten for Stage B.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
