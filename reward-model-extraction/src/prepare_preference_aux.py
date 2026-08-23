"""Expand preference triples into label-free prompt/response teacher queries."""
from __future__ import annotations

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_pairs", type=int, default=None)
    return parser.parse_args()


def get_pair(row):
    if {"prompt", "chosen", "rejected"} <= row.keys():
        return str(row["prompt"]), str(row["chosen"]), str(row["rejected"])
    if {"x", "y_plus", "y_minus"} <= row.keys():
        return str(row["x"]), str(row["y_plus"]), str(row["y_minus"])
    raise KeyError("Each input row must have prompt/chosen/rejected or x/y_plus/y_minus.")


def main(args):
    queries = []
    pair_count = 0
    with open(args.input_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            prompt, first_response, second_response = get_pair(json.loads(line))
            if not prompt.strip() or not first_response.strip() or not second_response.strip():
                continue
            # The output intentionally contains no chosen/rejected roles.  The
            # teacher receives two independent (X,Y) inputs and determines
            # their ordering from its rewards after scoring.
            queries.extend((
                {"prompt": prompt, "response": first_response, "source_pair_index": pair_count},
                {"prompt": prompt, "response": second_response, "source_pair_index": pair_count},
            ))
            pair_count += 1
            if args.max_pairs is not None and pair_count >= args.max_pairs:
                break
    if not queries:
        raise ValueError("No valid preference pairs were found.")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(query, ensure_ascii=False) + "\n")
    print(f"[OK] expanded {pair_count} preference pairs into {len(queries)} label-free teacher queries: {args.output_path}")


if __name__ == "__main__":
    main(parse_args())
