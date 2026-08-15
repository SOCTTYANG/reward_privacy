#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


M_VALUE = 3
METRICS = ["ASR", "AUC", "T@1%F", "T@5%F"]
MODEL_ORDER = [
    "llama2-7b",
    "llama32-3b",
    "llama2-13b-hf",
    "mistral-7b-v0.1",
    "qwen3-8b",
]
MODEL_ALIASES = {
    "llama2-7b": ["llama2-7b", "llama-2-7b", "llama_2_7b"],
    "llama32-3b": ["llama32-3b", "llama3.2-3b", "llama-3.2-3b", "llama_3.2_3b"],
    "llama2-13b-hf": ["llama2-13b-hf", "llama2-13b", "llama-2-13b"],
    "mistral-7b-v0.1": ["mistral-7b-v0.1", "mistral-7b", "mistral"],
    "qwen3-8b": ["qwen3-8b", "qwen-3-8b", "qwen3_8b"],
}
DEFAULT_M3_ROOTS = [
    Path(
        "/run/media/vipuser/data/yang-safe/"
        "membership_inference_multi_models_2gpu_best_rm_local"
    )
]
DEFAULT_EXTRA_RESULT_DIRS = [
    (
        "llama2-7b",
        Path(
            "/run/media/vipuser/data/yang-safe/membership inference/"
            "3.4-Membership Inference-r64-margin8-fullppo-fpr1"
        ),
    ),
    (
        "llama2-7b",
        Path(
            "/run/media/vipuser/data/yang-safe/membership inference/"
            "3.4-Membership Inference-r64-margin8-fullppo-fpr5"
        ),
    ),
]
RESULT_NAME_PATTERNS = [
    "3.4*table_metrics*.csv",
    "3.4*summary*.json",
    "3.4*summary*.txt",
]


def parse_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return math.nan
    return float(text)


def metric_name(name: str) -> str | None:
    compact = name.strip().lower().replace(" ", "")
    aliases = {
        "asr": "ASR",
        "accuracy": "ASR",
        "auc": "AUC",
        "t@1%f": "T@1%F",
        "t@1%fpr": "T@1%F",
        "tpr_at_1pct_fpr": "T@1%F",
        "t@5%f": "T@5%F",
        "t@5%fpr": "T@5%F",
        "tpr_at_5pct_fpr": "T@5%F",
    }
    return aliases.get(compact)


def empty_row(model_tag: str) -> dict:
    return {"model_tag": model_tag, "m": M_VALUE, **{metric: math.nan for metric in METRICS}}


def update_max(row: dict, values: Dict[str, float]):
    for metric in METRICS:
        value = values.get(metric, math.nan)
        if math.isnan(value):
            continue
        old_value = row.get(metric, math.nan)
        if math.isnan(old_value) or value > old_value:
            row[metric] = value


def infer_model_tag(path: Path) -> str | None:
    text = str(path).lower().replace("\\", "/")
    for model_tag, aliases in MODEL_ALIASES.items():
        if any(alias in text for alias in aliases):
            return model_tag
    return None


def result_files_under(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        return []
    files: List[Path] = []
    for pattern in RESULT_NAME_PATTERNS:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(files))


def read_csv_metrics(path: Path) -> Dict[str, float]:
    best: Dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = {}
            for key, value in row.items():
                metric = metric_name(key or "")
                if metric is not None:
                    values[metric] = parse_float(value)
            update_max(best, values)
    return best


def visit_json(obj: object, values: Dict[str, float]):
    if isinstance(obj, dict):
        for key, value in obj.items():
            metric = metric_name(str(key))
            if metric is not None and not isinstance(value, (dict, list)):
                values[metric] = parse_float(value)
            visit_json(value, values)
    elif isinstance(obj, list):
        for item in obj:
            visit_json(item, values)


def read_json_metrics(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    values: Dict[str, float] = {}
    visit_json(obj, values)
    return values


def read_txt_metrics(path: Path) -> Dict[str, float]:
    values: Dict[str, float] = {}
    pattern = re.compile(r"^\s*([^:]+?)\s*:\s*([-+0-9.eE]+)\s*$")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.match(line)
            if not match:
                continue
            metric = metric_name(match.group(1))
            if metric is not None:
                values[metric] = parse_float(match.group(2))
    return values


def read_metric_file(path: Path) -> Dict[str, float]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_metrics(path)
    if suffix == ".json":
        return read_json_metrics(path)
    if suffix == ".txt":
        return read_txt_metrics(path)
    return {}


def collect_from_roots(
    roots: Sequence[Path],
    extra_result_dirs: Sequence[Tuple[str, Path]],
) -> Dict[str, dict]:
    rows = {model_tag: empty_row(model_tag) for model_tag in MODEL_ORDER}

    for root in roots:
        for result_file in result_files_under(root):
            model_tag = infer_model_tag(result_file)
            if model_tag is None:
                continue
            update_max(rows[model_tag], read_metric_file(result_file))

    for model_tag, result_dir in extra_result_dirs:
        for result_file in result_files_under(result_dir):
            update_max(rows[model_tag], read_metric_file(result_file))

    return rows


def format_value(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.10g}"


def write_by_model(rows: Dict[str, dict], output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_tag", "m", *METRICS])
        writer.writeheader()
        for model_tag in MODEL_ORDER:
            row = rows[model_tag]
            writer.writerow(
                {
                    "model_tag": model_tag,
                    "m": M_VALUE,
                    **{metric: format_value(row[metric]) for metric in METRICS},
                }
            )


def write_by_metric(rows: Dict[str, dict], output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *MODEL_ORDER])
        for metric in METRICS:
            writer.writerow(
                [metric, *[format_value(rows[model_tag][metric]) for model_tag in MODEL_ORDER]]
            )


def write_long(rows: Dict[str, dict], output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "metric", "model_tag", "value"])
        for metric in METRICS:
            for model_tag in MODEL_ORDER:
                writer.writerow([M_VALUE, metric, model_tag, format_value(rows[model_tag][metric])])


def parse_extra_result(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected MODEL_TAG=/path/to/result_dir")
    model_tag, path = value.split("=", 1)
    model_tag = model_tag.strip()
    if model_tag not in MODEL_ORDER:
        raise argparse.ArgumentTypeError(
            f"Unknown model tag {model_tag!r}. Valid tags: {', '.join(MODEL_ORDER)}"
        )
    return model_tag, Path(path.strip())


def warn_missing(rows: Dict[str, dict]):
    for model_tag in MODEL_ORDER:
        missing = [metric for metric in METRICS if math.isnan(rows[model_tag][metric])]
        if missing:
            print(f"[WARN] {model_tag} missing: {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect M=3 metrics for five models into CSV files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=[],
        help="M=3 root containing per-model result folders. Can be used multiple times.",
    )
    parser.add_argument(
        "--extra-result",
        type=parse_extra_result,
        action="append",
        default=[],
        help="Extra result directory in MODEL_TAG=/path format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output CSV files.",
    )
    args = parser.parse_args()

    roots = args.root or DEFAULT_M3_ROOTS
    extra_result_dirs = list(DEFAULT_EXTRA_RESULT_DIRS)
    extra_result_dirs.extend(args.extra_result)

    rows = collect_from_roots(roots, extra_result_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_metric = args.output_dir / "m3_metrics_by_metric.csv"
    by_model = args.output_dir / "m3_metrics_by_model.csv"
    long_csv = args.output_dir / "m3_metrics_long.csv"

    write_by_metric(rows, by_metric)
    write_by_model(rows, by_model)
    write_long(rows, long_csv)
    warn_missing(rows)

    print(f"[DONE] Wrote metric-row table: {by_metric}")
    print(f"[DONE] Wrote model-row table : {by_model}")
    print(f"[DONE] Wrote long table      : {long_csv}")


if __name__ == "__main__":
    main()
