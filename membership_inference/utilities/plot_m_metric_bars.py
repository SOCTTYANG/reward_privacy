#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


METRICS = ["ASR", "AUC", "T@1%F", "T@5%F"]
M_VALUES = [1, 3, 5]
MODEL_ORDER = [
    "llama2-7b",
    "llama32-3b",
    "llama2-13b-hf",
    "mistral-7b-v0.1",
    "qwen3-8b",
]
MODEL_LABELS = {
    "llama2-7b": "Llama2-7B",
    "llama32-3b": "Llama3.2-3B",
    "llama2-13b-hf": "Llama2-13B",
    "mistral-7b-v0.1": "Mistral-7B",
    "qwen3-8b": "Qwen3-8B",
}
DEFAULT_SEARCH_ROOTS = [
    Path("/mnt/model_data/yang_safe"),
    Path("/mnt/bai_data/yang-safe"),
    Path("/run/media/vipuser/data/yang-safe"),
]
DEFAULT_FULL_RUN_ROOTS = {
    3: [
        Path(
            "/run/media/vipuser/data/yang-safe/"
            "membership_inference_multi_models_2gpu_best_rm_local"
        )
    ]
}
DEFAULT_FULL_EXTRA_RESULT_DIRS = {
    3: [
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
}
METRIC_FILE_NAMES = {
    "3.4_mia_table_metrics.csv",
    "3.4_membership_inference_summary.json",
    "3.4_membership_inference_summary.txt",
    "3.4_reward_gap_ablation_table_metrics.csv",
    "3.4_reward_gap_ablation_summary.json",
    "3.4_reward_gap_ablation_summary.txt",
}
MODEL_ALIASES = {
    "llama2-7b": ["llama2-7b", "llama-2-7b", "llama_2_7b"],
    "llama32-3b": ["llama32-3b", "llama3.2-3b", "llama-3.2-3b", "llama_3.2_3b"],
    "llama2-13b-hf": ["llama2-13b-hf", "llama2-13b", "llama-2-13b"],
    "mistral-7b-v0.1": ["mistral-7b-v0.1", "mistral-7b", "mistral"],
    "qwen3-8b": ["qwen3-8b", "qwen-3-8b", "qwen3_8b"],
}


def default_paths(source: str, score_mode: str) -> Dict[int, Path]:
    if source == "reward-gap":
        return {
            m: Path(
                f"/mnt/model_data/yang_safe/"
                f"mia_reward_gap_ablation_{score_mode}_m{m}_5models/"
                f"mia_reward_gap_ablation_m{m}_table_metrics.csv"
            )
            for m in M_VALUES
        }
    return {
        m: Path(
            f"/mnt/model_data/yang_safe/"
            f"mia_full_m{m}_5models/"
            f"mia_full_m{m}_table_metrics.csv"
        )
        for m in M_VALUES
    }


def summary_filename(source: str, m_value: int) -> str:
    if source == "reward-gap":
        return f"mia_reward_gap_ablation_m{m_value}_table_metrics.csv"
    return f"mia_full_m{m_value}_table_metrics.csv"


def find_summary_csv(source: str, m_value: int, search_roots: Sequence[Path]) -> Path | None:
    filename = summary_filename(source, m_value)
    candidates: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.name == filename:
            candidates.append(root)
            continue
        if root.is_dir():
            candidates.extend(root.rglob(filename))
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_missing_paths(
    paths: Dict[int, Path],
    source: str,
    search_roots: Sequence[Path],
) -> Dict[int, Path]:
    resolved = dict(paths)
    for m_value, path in paths.items():
        if path.is_file():
            continue
        found = find_summary_csv(source, m_value, search_roots)
        if found is not None:
            print(f"[INFO] Found M={m_value} CSV: {found}")
            resolved[m_value] = found
    return resolved


def parse_float(value: str) -> float:
    value = str(value).strip()
    if value == "" or value.lower() in {"nan", "none", "null"}:
        return math.nan
    return float(value)


def update_metric_max(target: dict, metric_values: Dict[str, float]):
    for metric in METRICS:
        value = metric_values.get(metric, math.nan)
        if math.isnan(value):
            continue
        old_value = target.get(metric, math.nan)
        if math.isnan(old_value) or value > old_value:
            target[metric] = value


def read_metrics(path: Path, m_value: int) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            model_tag = row.get("model_tag", "").strip()
            if not model_tag:
                continue
            item = {"model_tag": model_tag, "m": m_value}
            for metric in METRICS:
                item[metric] = parse_float(row.get(metric, ""))
            rows.append(item)
    return rows


def normalize_metric_name(name: str) -> str | None:
    compact = name.strip().lower().replace(" ", "")
    aliases = {
        "asr": "ASR",
        "auc": "AUC",
        "t@1%f": "T@1%F",
        "t@1%fpr": "T@1%F",
        "tpr_at_1pct_fpr": "T@1%F",
        "t@5%f": "T@5%F",
        "t@5%fpr": "T@5%F",
        "tpr_at_5pct_fpr": "T@5%F",
    }
    return aliases.get(compact)


def read_metric_file(path: Path) -> Dict[str, float]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            best: Dict[str, float] = {}
            for row in reader:
                metric_values = {}
                for key, value in row.items():
                    metric = normalize_metric_name(key or "")
                    if metric is not None:
                        metric_values[metric] = parse_float(value)
                update_metric_max(best, metric_values)
            return best

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        metric_values = {}
        if isinstance(obj, dict):
            table = obj.get("table_metrics", {})
            if isinstance(table, dict):
                for key, value in table.items():
                    metric = normalize_metric_name(key)
                    if metric is not None:
                        metric_values[metric] = parse_float(value)
            metrics = obj.get("metrics", {})
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    metric = normalize_metric_name(key)
                    if metric is not None:
                        metric_values[metric] = parse_float(value)
        return metric_values

    metric_values = {}
    pattern = re.compile(r"^\s*([^:]+?)\s*:\s*([-+0-9.eE]+)\s*$")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.match(line)
            if not match:
                continue
            metric = normalize_metric_name(match.group(1))
            if metric is not None:
                metric_values[metric] = parse_float(match.group(2))
    return metric_values


def infer_model_tag(path: Path) -> str | None:
    text = str(path).lower().replace("\\", "/")
    for model_tag, aliases in MODEL_ALIASES.items():
        if any(alias in text for alias in aliases):
            return model_tag
    return None


def metric_files_under(path: Path) -> List[Path]:
    if path.is_file():
        return [path] if path.name in METRIC_FILE_NAMES else []
    if not path.is_dir():
        return []
    return [
        item
        for item in path.rglob("*")
        if item.is_file() and item.name in METRIC_FILE_NAMES
    ]


def read_result_dir(path: Path, m_value: int, model_tag: str | None = None) -> List[dict]:
    grouped: Dict[str, dict] = {}
    for metric_file in metric_files_under(path):
        inferred_model = model_tag or infer_model_tag(metric_file)
        if inferred_model is None:
            continue
        row = grouped.setdefault(
            inferred_model,
            {
                "model_tag": inferred_model,
                "m": m_value,
                **{metric: math.nan for metric in METRICS},
            },
        )
        update_metric_max(row, read_metric_file(metric_file))
    return list(grouped.values())


def load_all(paths: Dict[int, Path]) -> Tuple[List[dict], List[Tuple[int, Path]]]:
    rows: List[dict] = []
    missing: List[Tuple[int, Path]] = []
    for m_value, path in paths.items():
        if path.is_file():
            rows.extend(read_metrics(path, m_value))
        else:
            missing.append((m_value, path))
    return rows, missing


def ordered_models(rows: Iterable[dict]) -> List[str]:
    present = {row["model_tag"] for row in rows}
    ordered = [model for model in MODEL_ORDER if model in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def value_lookup(rows: Iterable[dict]) -> Dict[Tuple[str, int, str], float]:
    values: Dict[Tuple[str, int, str], float] = {}
    for row in rows:
        for metric in METRICS:
            value = row[metric]
            if math.isnan(value):
                continue
            key = (row["model_tag"], row["m"], metric)
            old_value = values.get(key, math.nan)
            if math.isnan(old_value) or value > old_value:
                values[key] = value
    return values


def add_value_label(ax, bar, value: float):
    if math.isnan(value):
        return
    ax.annotate(
        f"{value:.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=90,
    )


def plot_matplotlib(rows: List[dict], output_path: Path, title: str):
    import matplotlib.pyplot as plt

    models = ordered_models(rows)
    values = value_lookup(rows)
    x_positions = list(range(len(models)))
    bar_width = 0.23
    offsets = {1: -bar_width, 3: 0.0, 5: bar_width}
    colors = {1: "#6b7280", 3: "#2563eb", 5: "#d97706"}
    hatches = {1: "///", 3: "---", 5: "oo"}

    fig, axes = plt.subplots(4, 1, figsize=(13, 15), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

    for ax, metric in zip(axes, METRICS):
        metric_values = [
            values.get((model, m_value, metric), math.nan)
            for model in models
            for m_value in M_VALUES
        ]
        ymax = max([v for v in metric_values if not math.isnan(v)] or [1.0])
        ytop = min(1.05, max(0.1, ymax * 1.18))

        best_key = None
        candidates = [
            (values.get((model, m_value, metric), math.nan), model, m_value)
            for model in models
            for m_value in M_VALUES
        ]
        candidates = [item for item in candidates if not math.isnan(item[0])]
        if candidates:
            best_key = max(candidates, key=lambda item: item[0])

        for m_value in M_VALUES:
            bar_values = [
                values.get((model, m_value, metric), math.nan) for model in models
            ]
            bars = ax.bar(
                [x + offsets[m_value] for x in x_positions],
                bar_values,
                width=bar_width,
                label=f"M={m_value}",
                color=colors[m_value],
                edgecolor="black",
                linewidth=0.8,
                hatch=hatches[m_value],
                alpha=0.82,
            )
            for model, bar, value in zip(models, bars, bar_values):
                add_value_label(ax, bar, value)
                if (
                    best_key is not None
                    and model == best_key[1]
                    and m_value == best_key[2]
                ):
                    ax.annotate(
                        "MAX",
                        xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 16),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color="#b91c1c",
                        fontweight="bold",
                    )

        ax.set_ylabel(metric)
        ax.set_ylim(0, ytop)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if best_key is not None:
            ax.set_title(
                f"{metric} highest: {MODEL_LABELS.get(best_key[1], best_key[1])}, "
                f"M={best_key[2]}, {best_key[0]:.4f}",
                loc="left",
                fontsize=11,
            )
        else:
            ax.set_title(metric, loc="left", fontsize=11)

    axes[0].legend(ncol=3, frameon=False, loc="upper right")
    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(
        [MODEL_LABELS.get(model, model) for model in models],
        rotation=18,
        ha="right",
    )
    axes[-1].set_xlabel("Model")
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[DONE] Saved plot: {output_path}")


def nice_ymax(values: Iterable[float]) -> float:
    finite_values = [v for v in values if not math.isnan(v)]
    if not finite_values:
        return 1.0
    ymax = max(finite_values)
    if ymax <= 1.0:
        return 1.0
    return math.ceil(ymax * 1.15 * 10) / 10


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    anchor: str = "start",
    weight: str = "400",
    rotate: float | None = None,
    fill: str = "#111827",
) -> str:
    rotate_attr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}"{rotate_attr}>'
        f"{html.escape(text)}</text>"
    )


def plot_svg(rows: List[dict], output_path: Path, title: str):
    models = ordered_models(rows)
    values = value_lookup(rows)

    width = 1500
    plot_height = 270
    top = 90
    left = 95
    right = 40
    gap = 82
    bottom_margin = 150
    height = top + len(METRICS) * plot_height + (len(METRICS) - 1) * gap + bottom_margin
    plot_width = width - left - right
    group_width = plot_width / max(1, len(models))
    bar_width = min(54, group_width * 0.18)
    offsets = {1: -bar_width * 1.18, 3: 0.0, 5: bar_width * 1.18}
    colors = {1: "#6b7280", 3: "#2563eb", 5: "#d97706"}
    patterns = {1: "diag", 3: "horiz", 5: "dots"}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<defs>",
        '<pattern id="diag" patternUnits="userSpaceOnUse" width="8" height="8">',
        '<rect width="8" height="8" fill="#6b7280" opacity="0.20"/>',
        '<path d="M-2,8 L8,-2 M0,10 L10,0" stroke="#111827" stroke-width="1.1"/>',
        "</pattern>",
        '<pattern id="horiz" patternUnits="userSpaceOnUse" width="8" height="8">',
        '<rect width="8" height="8" fill="#2563eb" opacity="0.20"/>',
        '<path d="M0,2 H8 M0,6 H8" stroke="#111827" stroke-width="1.1"/>',
        "</pattern>",
        '<pattern id="dots" patternUnits="userSpaceOnUse" width="9" height="9">',
        '<rect width="9" height="9" fill="#d97706" opacity="0.20"/>',
        '<circle cx="2.5" cy="2.5" r="1.25" fill="#111827"/>',
        '<circle cx="6.5" cy="6.5" r="1.25" fill="#111827"/>',
        "</pattern>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 35, title, size=22, anchor="middle", weight="500"),
    ]

    legend_x = width - 250
    legend_y = 24
    for index, m_value in enumerate(M_VALUES):
        y = legend_y + index * 26
        parts.append(
            f'<rect x="{legend_x}" y="{y}" width="46" height="16" '
            f'fill="url(#{patterns[m_value]})" stroke="#111827" stroke-width="1"/>'
        )
        parts.append(svg_text(legend_x + 58, y + 13, f"M={m_value}", size=15))

    for metric_index, metric in enumerate(METRICS):
        y0 = top + metric_index * (plot_height + gap)
        y1 = y0 + plot_height
        metric_values = [
            values.get((model, m_value, metric), math.nan)
            for model in models
            for m_value in M_VALUES
        ]
        ymax = nice_ymax(metric_values)

        best_key = None
        candidates = [
            (values.get((model, m_value, metric), math.nan), model, m_value)
            for model in models
            for m_value in M_VALUES
        ]
        candidates = [item for item in candidates if not math.isnan(item[0])]
        if candidates:
            best_key = max(candidates, key=lambda item: item[0])

        if best_key is None:
            parts.append(svg_text(left, y0 - 24, metric, size=19, weight="500"))
        else:
            best_label = MODEL_LABELS.get(best_key[1], best_key[1])
            parts.append(
                svg_text(
                    left,
                    y0 - 24,
                    f"{metric}  highest: {best_label}, M={best_key[2]}, {best_key[0]:.4f}",
                    size=19,
                    weight="500",
                )
            )

        for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
            if tick > ymax:
                continue
            y = y1 - tick / ymax * plot_height
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                f'stroke="#d1d5db" stroke-width="1" stroke-dasharray="4 5"/>'
            )
            parts.append(svg_text(left - 12, y + 5, f"{tick:.2g}", size=12, anchor="end", fill="#4b5563"))

        parts.append(
            f'<line x1="{left}" y1="{y1}" x2="{width - right}" y2="{y1}" '
            f'stroke="#111827" stroke-width="1.5"/>'
        )
        parts.append(
            f'<line x1="{left}" y1="{y0}" x2="{left}" y2="{y1}" '
            f'stroke="#111827" stroke-width="1.5"/>'
        )
        parts.append(svg_text(left - 62, y0 + plot_height / 2, metric, size=15, anchor="middle", rotate=-90))

        for model_index, model in enumerate(models):
            center_x = left + group_width * (model_index + 0.5)
            for m_value in M_VALUES:
                value = values.get((model, m_value, metric), math.nan)
                if math.isnan(value):
                    continue
                bar_height = value / ymax * plot_height
                x = center_x + offsets[m_value] - bar_width / 2
                y = y1 - bar_height
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                    f'height="{bar_height:.1f}" fill="url(#{patterns[m_value]})" '
                    f'stroke="#111827" stroke-width="1"/>'
                )
                parts.append(
                    svg_text(
                        x + bar_width / 2,
                        y - 5,
                        f"{value:.3f}",
                        size=11,
                        anchor="middle",
                        rotate=-90,
                        fill="#374151",
                    )
                )
                if best_key is not None and model == best_key[1] and m_value == best_key[2]:
                    parts.append(svg_text(x + bar_width / 2, y - 21, "MAX", size=13, anchor="middle", weight="500", fill="#b91c1c"))

            if metric_index == len(METRICS) - 1:
                parts.append(
                    svg_text(
                        center_x,
                        y1 + 42,
                        MODEL_LABELS.get(model, model),
                        size=14,
                        anchor="end",
                        rotate=-22,
                    )
                )

    parts.append(svg_text(width / 2, height - 35, "Model", size=16, anchor="middle", weight="500"))
    parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[DONE] Saved plot: {output_path}")


def plot(rows: List[dict], output_path: Path, title: str):
    if output_path.suffix.lower() == ".svg":
        plot_svg(rows, output_path, title)
        return
    plot_matplotlib(rows, output_path, title)


def parse_extra_result(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected format MODEL_TAG=/path/to/result_dir, for example "
            "llama2-7b=/path/with/results"
        )
    model_tag, path = value.split("=", 1)
    model_tag = model_tag.strip()
    if model_tag not in MODEL_ALIASES:
        raise argparse.ArgumentTypeError(
            f"Unknown model tag {model_tag!r}. Valid tags: {', '.join(MODEL_ORDER)}"
        )
    return model_tag, Path(path.strip())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Draw grouped bar charts for ASR, AUC, T@1%F, and T@5%F across "
            "models and M values."
        )
    )
    parser.add_argument(
        "--source",
        choices=["reward-gap", "full"],
        default="reward-gap",
        help="Which default result directories to read.",
    )
    parser.add_argument(
        "--score-mode",
        default="candidate_mean_gap",
        help="Reward-gap directory score mode name.",
    )
    parser.add_argument(
        "--m1-csv",
        type=Path,
        default=None,
        help="Override the M=1 summary CSV path.",
    )
    parser.add_argument(
        "--m3-csv",
        type=Path,
        default=None,
        help="Override the M=3 summary CSV path.",
    )
    parser.add_argument(
        "--m5-csv",
        type=Path,
        default=None,
        help="Override the M=5 summary CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mia_m_value_metric_bars.svg"),
        help="Output image path.",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Directory to search when default CSV paths are missing. "
            "Can be used multiple times."
        ),
    )
    parser.add_argument(
        "--m3-run-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "M=3 result root containing per-model folders. The script searches "
            "inside each model folder for 3.4 table/summary files."
        ),
    )
    parser.add_argument(
        "--m3-extra-result",
        type=parse_extra_result,
        action="append",
        default=[],
        help=(
            "Extra M=3 result directory in MODEL_TAG=/path format. Use this for "
            "separate llama2-7b fpr1/fpr5 folders."
        ),
    )
    args = parser.parse_args()

    paths = default_paths(args.source, args.score_mode)
    overrides = {1: args.m1_csv, 3: args.m3_csv, 5: args.m5_csv}
    for m_value, path in overrides.items():
        if path is not None:
            paths[m_value] = path

    explicit_m_paths = any(path is not None for path in overrides.values())
    if not explicit_m_paths:
        search_roots = args.search_root or DEFAULT_SEARCH_ROOTS
        paths = resolve_missing_paths(paths, args.source, search_roots)

    rows, missing = load_all(paths)
    if args.source == "full":
        m3_roots = list(args.m3_run_root)
        if args.m3_csv is None:
            m3_roots.extend(DEFAULT_FULL_RUN_ROOTS.get(3, []))
        for root in m3_roots:
            loaded = read_result_dir(root, 3)
            if loaded:
                print(f"[INFO] Loaded M=3 per-model results from: {root}")
                rows.extend(loaded)

        m3_extra_results = list(args.m3_extra_result)
        if args.m3_csv is None:
            m3_extra_results.extend(DEFAULT_FULL_EXTRA_RESULT_DIRS.get(3, []))
        for model_tag, result_dir in m3_extra_results:
            loaded = read_result_dir(result_dir, 3, model_tag=model_tag)
            if loaded:
                print(f"[INFO] Loaded M=3 extra result for {model_tag}: {result_dir}")
                rows.extend(loaded)

    loaded_m_values = {row["m"] for row in rows}
    missing = [(m_value, path) for m_value, path in missing if m_value not in loaded_m_values]
    if missing:
        print("[WARN] Missing CSV files:")
        for m_value, path in missing:
            print(f"  M={m_value}: {path}")
    if not rows:
        raise SystemExit(
            "No metrics were loaded. Please run this script on the machine "
            "where the result CSV files exist, or pass --m1-csv/--m3-csv/--m5-csv."
        )

    title = f"MIA metrics by model and M ({args.source})"
    plot(rows, args.output, title)


if __name__ == "__main__":
    main()
