#!/usr/bin/env python3
"""Check exact row overlap between two Hugging Face datasets.

By default this compares PKU-Alignment/PKU-SafeRLHF-10K and
PKU-Alignment/PKU-SafeRLHF-30K.  A row matches only when every shared
column has exactly the same value (including prompt, both responses, and
preference/safety labels).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_LEFT = "PKU-Alignment/PKU-SafeRLHF-10K"
DEFAULT_RIGHT = "PKU-Alignment/PKU-SafeRLHF-30K"


def canonical(value: Any) -> str:
    """Serialize a value deterministically without altering its text."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(row: dict[str, Any], fields: list[str]) -> str:
    payload = {field: row[field] for field in fields}
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def load_hf_dataset(dataset_id: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `python -m pip install datasets`."
        ) from exc

    try:
        return load_dataset(dataset_id, split=split)
    except Exception as exc:
        raise SystemExit(
            f"Could not load {dataset_id!r} (split={split!r}): {exc}"
        ) from exc


def count_hashes(rows: Iterable[dict[str, Any]], fields: list[str]) -> collections.Counter[str]:
    return collections.Counter(digest(row, fields) for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", default=DEFAULT_LEFT, help="Left Hugging Face dataset ID.")
    parser.add_argument("--right", default=DEFAULT_RIGHT, help="Right Hugging Face dataset ID.")
    parser.add_argument("--left-split", default="train", help="Split to read from the left dataset.")
    parser.add_argument("--right-split", default="train", help="Split to read from the right dataset.")
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Columns to compare. Defaults to every column shared by the two datasets.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        metavar="N",
        help="Print up to N matching hashes (not raw potentially-sensitive content).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left = load_hf_dataset(args.left, args.left_split)
    right = load_hf_dataset(args.right, args.right_split)

    left_columns = set(left.column_names)
    right_columns = set(right.column_names)
    fields = args.fields or sorted(left_columns & right_columns)
    if not fields:
        raise SystemExit("The datasets have no shared columns; specify --fields explicitly.")

    missing_left = set(fields) - left_columns
    missing_right = set(fields) - right_columns
    if missing_left or missing_right:
        raise SystemExit(
            f"Requested fields are unavailable. left missing={sorted(missing_left)}, "
            f"right missing={sorted(missing_right)}"
        )

    left_counts = count_hashes(left, fields)
    right_counts = count_hashes(right, fields)
    common_hashes = left_counts.keys() & right_counts.keys()

    # Unique overlap answers “how many distinct identical examples?”.
    # Multiset overlap additionally accounts for duplicated rows in either source.
    unique_overlap = len(common_hashes)
    multiset_overlap = sum(min(left_counts[key], right_counts[key]) for key in common_hashes)

    result = {
        "left": {"dataset": args.left, "split": args.left_split, "rows": len(left), "unique_rows": len(left_counts)},
        "right": {"dataset": args.right, "split": args.right_split, "rows": len(right), "unique_rows": len(right_counts)},
        "comparison_fields": fields,
        "exact_overlap": {
            "unique_identical_rows": unique_overlap,
            "identical_rows_counting_duplicates": multiset_overlap,
            "percent_of_left_rows": round(100 * multiset_overlap / len(left), 6) if len(left) else 0,
            "percent_of_right_rows": round(100 * multiset_overlap / len(right), 6) if len(right) else 0,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.show:
        print("\nExample matching SHA-256 identifiers:")
        for key in sorted(common_hashes)[: args.show]:
            print(f"  {key}  (left={left_counts[key]}, right={right_counts[key]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
