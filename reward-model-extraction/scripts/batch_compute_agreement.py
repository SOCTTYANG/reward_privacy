#!/usr/bin/env python
"""Backfill Agreement for every saved diff experiment without model inference.

Agreement is the fraction of labelled preference pairs for which both the
target and substitute reward models score ``chosen`` strictly above
``rejected``.  All inputs come from existing ``diff_records.csv`` files; this
script never imports training code or loads a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def compute_agreement(records: list[dict[str, str]]) -> dict[str, float | int | None]:
    pairs: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for record in records:
        response_name = record.get("response_name")
        if response_name in {"chosen", "rejected"}:
            pairs[(record["split"], record["row_id"])][response_name] = record

    agreed = 0
    count = 0
    for pair in pairs.values():
        if not {"chosen", "rejected"}.issubset(pair):
            continue
        chosen, rejected = pair["chosen"], pair["rejected"]
        target_correct = float(chosen["target_score"]) > float(rejected["target_score"])
        substitute_correct = float(chosen["substitute_score"]) > float(
            rejected["substitute_score"]
        )
        agreed += int(target_correct and substitute_correct)
        count += 1

    return {
        "Agreement": agreed / count if count else None,
        "agreement_count": count,
        "agreed_count": agreed,
    }


def process_experiment(diff_dir: Path, update_summaries: bool) -> dict:
    records_path = diff_dir / "diff_records.csv"
    summary_path = diff_dir / "diff_summary.json"
    with records_path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))

    by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_split[record["split"]].append(record)

    split_metrics = {name: compute_agreement(rows) for name, rows in by_split.items()}
    all_metrics = compute_agreement(records)

    if update_summaries:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        for split in summary.get("splits", []):
            split.update(split_metrics.get(split.get("split"), {}))
        summary.setdefault("all", {}).update(all_metrics)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    return {
        "experiment": diff_dir.name,
        **all_metrics,
        "splits": split_metrics,
        "records_path": str(records_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/agreement"))
    parser.add_argument("--expected-count", type=int, default=38)
    parser.add_argument(
        "--no-update-summaries",
        action="store_true",
        help="Do not add Agreement fields to the existing diff_summary.json files.",
    )
    args = parser.parse_args()

    diff_dirs = sorted(
        path
        for path in args.output_root.iterdir()
        if path.is_dir()
        and path.name.startswith("diff_")
        and (path / "diff_records.csv").is_file()
        and (path / "diff_summary.json").is_file()
    )
    if len(diff_dirs) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} complete diff experiments, found {len(diff_dirs)}. "
            "Refusing to silently produce an incomplete table; pass --expected-count to override."
        )

    results = [
        process_experiment(path, update_summaries=not args.no_update_summaries)
        for path in diff_dirs
    ]
    args.results_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.results_dir / "agreement_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    csv_path = args.results_dir / "agreement_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment", "Agreement", "agreement_count", "agreed_count"],
        )
        writer.writeheader()
        writer.writerows(
            {key: result[key] for key in writer.fieldnames} for result in results
        )

    print(f"Computed Agreement for {len(results)} experiments (checkpoint-free).")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
