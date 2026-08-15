#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from typing import Dict, List, Tuple


METRICS = ["ASR", "AUC", "T@1%F", "T@5%F"]
M_INPUTS = {
    1: Path("m1_yellow_ours_metrics_by_metric.csv"),
    3: Path("metric_bar_charts/m3_yellow_ours_metrics_by_metric.csv"),
    5: Path("m5_yellow_ours_metrics_by_metric.csv"),
}
MODEL_ORDER = ["Llama2-7b", "Llama3.2-3b", "Llama2-13b", "mistral-7b", "QWen3-8b"]
METRIC_FILE_STEMS = {
    "ASR": "asr",
    "AUC": "auc",
    "T@1%F": "t_at_1f",
    "T@5%F": "t_at_5f",
}


def read_by_metric_csv(path: Path) -> Dict[str, Dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        data: Dict[str, Dict[str, float]] = {}
        for row in reader:
            metric = row["metric"].strip()
            data[metric] = {
                model: float(row[model])
                for model in MODEL_ORDER
                if row.get(model, "").strip()
            }
    return data


def load_all(inputs: Dict[int, Path]) -> Dict[Tuple[int, str, str], float]:
    values: Dict[Tuple[int, str, str], float] = {}
    for m_value, path in inputs.items():
        data = read_by_metric_csv(path)
        for metric in METRICS:
            for model in MODEL_ORDER:
                values[(m_value, metric, model)] = data[metric][model]
    return values


def nice_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 0.001
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for multiplier in [1, 2, 5, 10]:
        step = multiplier * magnitude
        if step >= raw_step:
            return step
    return 10 * magnitude


def zoom_axis(values: List[float]) -> Tuple[float, float, float]:
    low = min(values)
    high = max(values)
    span = high - low
    if span == 0:
        span = max(abs(high) * 0.01, 0.001)
    padding = span * 0.35
    raw_min = max(0.0, low - padding)
    raw_max = high + padding
    step = nice_step((raw_max - raw_min) / 5)
    y_min = math.floor(raw_min / step) * step
    y_max = math.ceil(raw_max / step) * step
    if y_max <= y_min:
        y_max = y_min + step
    return y_min, y_max, step


def axis_ticks(y_min: float, y_max: float, step: float) -> List[float]:
    ticks = []
    current = math.ceil(y_min / step) * step
    while current <= y_max + step / 2:
        ticks.append(round(current, 8))
        current += step
    return ticks


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 16,
    anchor: str = "start",
    weight: str = "400",
    rotate: float | None = None,
    fill: str = "#111827",
) -> str:
    rotate_attr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}"{rotate_attr}>{html.escape(text)}</text>'
    )


def format_tick(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if abs(value) < 0.8:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.4f}".rstrip("0").rstrip(".")


def chart_svg(metric: str, values: Dict[Tuple[int, str, str], float]) -> str:
    width = 1280
    height = 720
    left = 124
    right = 50
    top = 116
    bottom = 138
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / len(MODEL_ORDER)
    bar_width = min(54, group_width * 0.18)
    offsets = {1: -bar_width * 1.18, 3: 0.0, 5: bar_width * 1.18}
    patterns = {1: "diag", 3: "horiz", 5: "dots"}

    metric_values = [
        values[(m_value, metric, model)]
        for model in MODEL_ORDER
        for m_value in [1, 3, 5]
    ]
    best_value = max(metric_values)
    y_min, y_max, y_step = zoom_axis(metric_values)
    y_base = top + plot_height
    y_span = y_max - y_min

    best_items = {
        (m_value, model)
        for model in MODEL_ORDER
        for m_value in [1, 3, 5]
        if abs(values[(m_value, metric, model)] - best_value) < 1e-12
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<defs>",
        '<pattern id="diag" patternUnits="userSpaceOnUse" width="8" height="8">',
        '<rect width="8" height="8" fill="#6b7280" opacity="0.22"/>',
        '<path d="M-2,8 L8,-2 M0,10 L10,0" stroke="#111827" stroke-width="1.1"/>',
        "</pattern>",
        '<pattern id="horiz" patternUnits="userSpaceOnUse" width="8" height="8">',
        '<rect width="8" height="8" fill="#2563eb" opacity="0.22"/>',
        '<path d="M0,2 H8 M0,6 H8" stroke="#111827" stroke-width="1.1"/>',
        "</pattern>",
        '<pattern id="dots" patternUnits="userSpaceOnUse" width="9" height="9">',
        '<rect width="9" height="9" fill="#d97706" opacity="0.22"/>',
        '<circle cx="2.5" cy="2.5" r="1.25" fill="#111827"/>',
        '<circle cx="6.5" cy="6.5" r="1.25" fill="#111827"/>',
        "</pattern>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 42, f"{metric} comparison by model and M", size=24, anchor="middle", weight="500"),
        svg_text(width / 2, 72, f"Zoomed y-axis: {format_tick(y_min)} to {format_tick(y_max)}", size=15, anchor="middle", fill="#4b5563"),
    ]

    legend_x = width - 330
    legend_y = 28
    for index, m_value in enumerate([1, 3, 5]):
        x = legend_x + index * 100
        parts.append(
            f'<rect x="{x}" y="{legend_y}" width="44" height="16" '
            f'fill="url(#{patterns[m_value]})" stroke="#111827" stroke-width="1"/>'
        )
        parts.append(svg_text(x + 53, legend_y + 13, f"M={m_value}", size=15))

    for tick in axis_ticks(y_min, y_max, y_step):
        y = y_base - (tick - y_min) / y_span * plot_height
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="#d1d5db" stroke-width="1" stroke-dasharray="4 5"/>'
        )
        parts.append(svg_text(left - 12, y + 5, format_tick(tick), size=17, anchor="end", fill="#4b5563"))

    parts.append(
        f'<line x1="{left}" y1="{y_base}" x2="{width - right}" y2="{y_base}" '
        f'stroke="#111827" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{y_base}" '
        f'stroke="#111827" stroke-width="1.5"/>'
    )
    parts.append(svg_text(left - 88, top + plot_height / 2, metric, size=30, anchor="middle", weight="500", rotate=-90))

    for model_index, model in enumerate(MODEL_ORDER):
        center_x = left + group_width * (model_index + 0.5)
        for m_value in [1, 3, 5]:
            value = values[(m_value, metric, model)]
            bar_height = (value - y_min) / y_span * plot_height
            x = center_x + offsets[m_value] - bar_width / 2
            y = y_base - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="url(#{patterns[m_value]})" '
                f'stroke="#111827" stroke-width="1"/>'
            )
            parts.append(
                svg_text(
                    x + bar_width / 2,
                    max(top - 10, y - 8),
                    f"{value:.4f}".rstrip("0").rstrip("."),
                    size=14,
                    anchor="middle",
                    fill="#374151",
                )
            )
            if (m_value, model) in best_items:
                parts.append(
                    svg_text(
                        x + bar_width / 2,
                        max(top - 30, y - 28),
                        "MAX",
                        size=13,
                        anchor="middle",
                        weight="500",
                        fill="#b91c1c",
                    )
                )

        parts.append(
            svg_text(
                center_x,
                y_base + 38,
                model,
                size=15,
                anchor="end",
                rotate=-22,
            )
        )

    parts.append(svg_text(width / 2, height - 28, "Model", size=16, anchor="middle", weight="500"))
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Generate one grouped SVG bar chart per metric from metric-row CSVs."
    )
    parser.add_argument("--m1-csv", type=Path, default=M_INPUTS[1])
    parser.add_argument("--m3-csv", type=Path, default=M_INPUTS[3])
    parser.add_argument("--m5-csv", type=Path, default=M_INPUTS[5])
    parser.add_argument("--output-dir", type=Path, default=Path("metric_bar_charts"))
    args = parser.parse_args()

    inputs = {1: args.m1_csv, 3: args.m3_csv, 5: args.m5_csv}
    for m_value, path in inputs.items():
        if not path.is_file():
            raise SystemExit(f"Missing M={m_value} CSV: {path}")

    values = load_all(inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        output = args.output_dir / f"{METRIC_FILE_STEMS[metric]}_by_model_m.svg"
        output.write_text(chart_svg(metric, values), encoding="utf-8")
        print(f"[DONE] {metric}: {output}")


if __name__ == "__main__":
    main()
