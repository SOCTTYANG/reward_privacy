import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BASE_DIR = Path(__file__).resolve().parent

MODELS = ["Llama2-7b", "mistral-7b", "QWen3-8b", "Llama3.2-3b", "Llama2-13b"]
DISPLAY_MODELS = ["Llama2-7B", "Mistral-7B", "Qwen3-8B", "Llama3.2-3B", "Llama2-13B"]

DATA_FILES = {
    "M=2": BASE_DIR / "m2_yellow_ours_metrics_by_model.csv",
    "M=3": BASE_DIR / "m3_yellow_ours_metrics_by_metric.csv",
    "M=4": BASE_DIR / "m4_yellow_ours_metrics_by_model.csv",
}

METRICS = [
    {
        "name": "AUC",
        "ylabel": "AUC",
        "output": "auc_by_model_m.pdf",
        "ylim": 0.85,
        "yticks": np.arange(0.0, 0.81, 0.2),
        "label_offset": 0.004,
    },
    {
        "name": "T@1%F",
        "ylabel": "TPR@1%FPR",
        "output": "t_at_1f_by_model_m.pdf",
        "ylim": 0.04,
        "yticks": np.arange(0.0, 0.041, 0.01),
        "label_offset": 0.001,
    },
    {
        "name": "T@5%F",
        "ylabel": "TPR@5%FPR",
        "output": "t_at_5f_by_model_m.pdf",
        "ylim": 0.22,
        "yticks": np.arange(0.0, 0.201, 0.05),
        "label_offset": 0.004,
    },
]


def load_metric_by_model(path, metric):
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = csv.DictReader(csv_file)
        return {row["model"]: float(row[metric]) for row in rows}


def load_metric_by_metric(path, metric):
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["metric"] == metric:
                return {model: float(row[model]) for model in MODELS}
    raise ValueError(f"{metric} row not found in {path}")


def load_metric(metric):
    return {
        "M=2": load_metric_by_model(DATA_FILES["M=2"], metric),
        "M=3": load_metric_by_metric(DATA_FILES["M=3"], metric),
        "M=4": load_metric_by_model(DATA_FILES["M=4"], metric),
    }


def fmt_value(value):
    rounded = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{rounded:.3f}"


def plot_metric(config):
    colors = ["#b3b3b3", "#66a265", "#6666b4"]
    bar_width = 0.19
    inner_gap = 0.0
    group_gap = 0.74
    group_centers = np.arange(len(MODELS)) * group_gap
    metric_values = load_metric(config["name"])

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (m_label, model_values) in enumerate(metric_values.items()):
        offset = (i - 1) * (bar_width + inner_gap)
        values = [model_values[model] for model in MODELS]
        bar_positions = group_centers + offset
        ax.bar(
            bar_positions,
            values,
            width=bar_width,
            color=colors[i],
            label=m_label,
            zorder=3,
        )

        for x, y in zip(bar_positions, values):
            ax.text(
                x,
                y + config["label_offset"],
                fmt_value(y),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
                clip_on=False,
            )

    ax.grid(True, linestyle="--", alpha=0.5, axis="y", zorder=1)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(DISPLAY_MODELS, fontsize=15)
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel(config["ylabel"], fontsize=14)
    ax.set_ylim(0.0, config["ylim"])
    ax.set_yticks(config["yticks"])
    ax.tick_params(axis="y", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=12)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    output_path = BASE_DIR / config["output"]
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Saved: {output_path}")
    plt.close(fig)


for metric_config in METRICS:
    plot_metric(metric_config)
