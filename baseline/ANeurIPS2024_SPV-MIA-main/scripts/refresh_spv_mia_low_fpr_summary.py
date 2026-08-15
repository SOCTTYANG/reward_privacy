import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fpr_percent(value: float) -> float:
    # Accept both percent-style inputs (1.0, 5.0) and fraction-style inputs (0.01, 0.05).
    return value * 100.0 if 0.0 < value < 1.0 else value


def normalize_fpr(value: float) -> float:
    return fpr_percent(value) / 100.0


def fpr_key(value: float) -> str:
    percent = fpr_percent(value)
    if abs(percent - round(percent)) < 1e-12:
        label = str(int(round(percent)))
    else:
        label = str(percent).rstrip("0").rstrip(".").replace(".", "p")
    return f"{label}pct"


def parse_fprs(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def metrics_at(labels, scores, fpr, tpr, thresholds, target_fpr: float) -> Dict[str, Any]:
    target_fraction = normalize_fpr(target_fpr)
    valid = np.where(fpr <= target_fraction + 1e-12)[0]
    if len(valid) == 0:
        idx = 0
    else:
        best_tpr = np.max(tpr[valid])
        idx = valid[np.where(tpr[valid] == best_tpr)[0][-1]]
    preds = (scores >= thresholds[idx]).astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "target_fpr": target_fpr,
        "target_fpr_fraction": float(target_fraction),
        "tpr": float(tpr[idx]),
        "accuracy": float(accuracy),
        "actual_fpr": float(fpr[idx]),
        "threshold": float(thresholds[idx]),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute(rows: List[Dict[str, Any]], target_fprs: List[float]) -> Dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    raw_scores = np.asarray([float(row["final_score"]) for row in rows], dtype=float)
    raw_auc = float(roc_auc_score(labels, raw_scores))
    score_flipped = raw_auc < 0.5
    scores = -raw_scores if score_flipped else raw_scores
    auc = float(1.0 - raw_auc if score_flipped else raw_auc)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    metrics_by_target_fpr = {}
    for target_fpr in target_fprs:
        metrics_by_target_fpr[fpr_key(target_fpr)] = metrics_at(
            labels,
            scores,
            fpr,
            tpr,
            thresholds,
            target_fpr,
        )

    return {
        "auc": auc,
        "raw_auc": raw_auc,
        "score_flipped": int(score_flipped),
        "target_fprs": target_fprs,
        "metrics_by_target_fpr": metrics_by_target_fpr,
        "tpr_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {}).get("tpr"),
        "acc_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {}).get("accuracy"),
        "actual_fpr_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {}).get("actual_fpr"),
        "threshold_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {}).get("threshold"),
        "tpr_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {}).get("tpr"),
        "acc_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {}).get("accuracy"),
        "actual_fpr_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {}).get("actual_fpr"),
        "threshold_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {}).get("threshold"),
    }


def model_tag_from_path(result_path: str) -> str:
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(result_path)))
    return run_dir.split("-stage2llm-")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh SPV-MIA T@1%FPR and T@5%FPR summaries.")
    parser.add_argument(
        "--output_root",
        default="/mnt/model_data/yang_safe/baseline/baseline_spv_mia_membership_steps_5models",
    )
    parser.add_argument("--target_fprs", default="1.0,5.0")
    args = parser.parse_args()

    target_fprs = parse_fprs(args.target_fprs)
    pattern = os.path.join(args.output_root, "*", "3.4-SPV-MIA-Baseline", "spv_mia_results.jsonl")
    result_paths = sorted(glob.glob(pattern))

    summary_rows = []
    for result_path in result_paths:
        rows = read_jsonl(result_path)
        roc = compute(rows, target_fprs)
        summary_path = os.path.join(os.path.dirname(result_path), "spv_mia_summary.json")
        summary_txt = os.path.join(os.path.dirname(result_path), "spv_mia_summary.txt")

        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            summary = {"config": {"model_tag": model_tag_from_path(result_path)}}

        summary.setdefault("roc_based_metrics", {}).update(roc)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        model_tag = summary.get("config", {}).get("model_tag") or model_tag_from_path(result_path)
        with open(summary_txt, "a", encoding="utf-8") as f:
            f.write("\nRefreshed low-FPR metrics:\n")
            f.write(f"T@1%FPR: {roc.get('tpr_at_1pct_fpr')}\n")
            f.write(f"ACC@1%FPR: {roc.get('acc_at_1pct_fpr')}\n")
            f.write(f"T@5%FPR: {roc.get('tpr_at_5pct_fpr')}\n")
            f.write(f"ACC@5%FPR: {roc.get('acc_at_5pct_fpr')}\n")

        summary_rows.append(
            {
                "model_tag": model_tag,
                "auc": roc["auc"],
                "raw_auc": roc["raw_auc"],
                "score_flipped": roc["score_flipped"],
                "tpr_at_1pct_fpr": roc.get("tpr_at_1pct_fpr"),
                "acc_at_1pct_fpr": roc.get("acc_at_1pct_fpr"),
                "actual_fpr_at_1pct_fpr": roc.get("actual_fpr_at_1pct_fpr"),
                "tpr_at_5pct_fpr": roc.get("tpr_at_5pct_fpr"),
                "acc_at_5pct_fpr": roc.get("acc_at_5pct_fpr"),
                "actual_fpr_at_5pct_fpr": roc.get("actual_fpr_at_5pct_fpr"),
                "summary_json": summary_path,
            }
        )

    summary_csv = os.path.join(args.output_root, "spv_mia_low_fpr_summary.csv")
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[DONE] refreshed models: {len(summary_rows)}")
        print(f"[DONE] low-FPR summary CSV: {summary_csv}")
    else:
        print(f"[WARN] no result files found: {pattern}")


if __name__ == "__main__":
    main()
