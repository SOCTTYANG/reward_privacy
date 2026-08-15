import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved: {path}")


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def format_reward_text(prompt: str, response: str) -> str:
    return "### Human:\n" + str(prompt).strip() + "\n\n### Assistant:\n" + str(response).strip()


@torch.no_grad()
def score_reward_pairs(
    model,
    tokenizer,
    pairs: List[Tuple[str, str]],
    batch_size: int,
    max_length: int,
    device,
) -> List[float]:
    scores = []
    model.eval()

    for start in tqdm(range(0, len(pairs), batch_size), desc="Scoring reward pairs"):
        batch_pairs = pairs[start:start + batch_size]
        texts = [format_reward_text(x, y) for x, y in batch_pairs]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits

        if logits.shape[-1] == 1:
            batch_scores = logits.squeeze(-1)
        elif logits.shape[-1] == 2:
            batch_scores = logits[:, 1]
        else:
            batch_scores = logits.max(dim=-1).values

        scores.extend(batch_scores.detach().float().cpu().tolist())

    return scores


def compute_classification_metrics(y_true: List[int], y_score: List[float], threshold: float):
    y_pred = [1 if s >= threshold else 0 for s in y_score]

    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    tnr = safe_div(tn, tn + fp)
    balanced_accuracy = 0.5 * (recall + tnr)

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": recall,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_auc(y_true: List[int], y_score: List[float]) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    rank_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += rank

    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def percentile(values: List[float], q: float) -> float:
    if len(values) == 0:
        return 0.0

    values = sorted(values)
    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * q / 100.0
    low = int(math.floor(pos))
    high = int(math.ceil(pos))

    if low == high:
        return values[low]

    weight = pos - low
    return values[low] * (1 - weight) + values[high] * weight


def fpr_percent(value: float) -> float:
    return value * 100.0 if 0.0 < value < 1.0 else value


def normalize_fpr(value: float) -> float:
    return fpr_percent(value) / 100.0


def fpr_metric_key(value: float) -> str:
    percent = fpr_percent(value)
    if abs(percent - round(percent)) < 1e-12:
        label = str(int(round(percent)))
    else:
        label = str(percent).rstrip("0").rstrip(".").replace(".", "p")
    return f"{label}pct"


def dedupe_fprs(values: List[float]) -> List[float]:
    output = []
    for value in values:
        percent = fpr_percent(value)
        if all(abs(percent - fpr_percent(existing)) > 1e-12 for existing in output):
            output.append(value)
    return output


def metrics_at_fpr(y_true: List[int], y_score: List[float], target_fpr: float):
    unique_scores = sorted(set(y_score))
    if len(unique_scores) == 0:
        metrics = compute_classification_metrics(y_true, y_score, 0.0)
        metrics["target_fpr"] = target_fpr
        metrics["target_fpr_fraction"] = normalize_fpr(target_fpr)
        return metrics

    eps = 1e-12
    thresholds = [min(unique_scores) - eps] + unique_scores + [max(unique_scores) + eps]
    target_fraction = normalize_fpr(target_fpr)

    best_metrics = None
    for threshold in thresholds:
        metrics = compute_classification_metrics(y_true, y_score, threshold)
        if metrics["fpr"] > target_fraction + 1e-12:
            continue

        if (
            best_metrics is None
            or metrics["tpr"] > best_metrics["tpr"]
            or (
                abs(metrics["tpr"] - best_metrics["tpr"]) <= 1e-12
                and metrics["accuracy"] > best_metrics["accuracy"]
            )
        ):
            best_metrics = metrics

    if best_metrics is None:
        best_metrics = compute_classification_metrics(y_true, y_score, max(unique_scores) + eps)

    best_metrics = dict(best_metrics)
    best_metrics["target_fpr"] = target_fpr
    best_metrics["target_fpr_fraction"] = target_fraction
    best_metrics["actual_fpr"] = best_metrics["fpr"]
    return best_metrics


def scan_best_thresholds(y_true: List[int], y_score: List[float]):
    unique_scores = sorted(set(y_score))
    if len(unique_scores) == 0:
        empty = compute_classification_metrics(y_true, y_score, 0.0)
        return empty, empty, empty

    thresholds = [min(unique_scores) - 1e-6] + unique_scores + [max(unique_scores) + 1e-6]

    best_f1 = None
    best_acc = None
    best_bal_acc = None

    for threshold in thresholds:
        metrics = compute_classification_metrics(y_true, y_score, threshold)

        if best_f1 is None or metrics["f1"] > best_f1["f1"]:
            best_f1 = metrics
        if best_acc is None or metrics["accuracy"] > best_acc["accuracy"]:
            best_acc = metrics
        if best_bal_acc is None or metrics["balanced_accuracy"] > best_bal_acc["balanced_accuracy"]:
            best_bal_acc = metrics

    return best_f1, best_acc, best_bal_acc


def stats(values: List[float]):
    if len(values) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }

    t = torch.tensor(values, dtype=torch.float32)
    return {
        "count": len(values),
        "mean": float(t.mean().item()),
        "std": float(t.std().item()) if len(values) > 1 else 0.0,
        "min": float(t.min().item()),
        "p25": float(percentile(values, 25)),
        "median": float(percentile(values, 50)),
        "p75": float(percentile(values, 75)),
        "max": float(t.max().item()),
    }


def summarize_score_distribution(results: List[Dict[str, Any]]):
    member = [x for x in results if x["label"] == 1]
    nonmember = [x for x in results if x["label"] == 0]

    return {
        "member": {
            "final_score": stats([x["final_score"] for x in member]),
            "r_gap": stats([x["r_gap"] for x in member]),
        },
        "nonmember": {
            "final_score": stats([x["final_score"] for x in nonmember]),
            "r_gap": stats([x["r_gap"] for x in nonmember]),
        },
    }


def candidate_reward_stats(item: Dict[str, Any]) -> Dict[str, float]:
    scores = []
    for candidate in item.get("candidate_responses", []):
        if "r_i" in candidate:
            scores.append(float(candidate["r_i"]))

    if len(scores) == 0:
        raise ValueError(f"Missing candidate reward scores for item id={item.get('id')}")

    return {
        "candidate_reward_count": len(scores),
        "candidate_reward_mean": sum(scores) / len(scores),
        "candidate_reward_max": max(scores),
        "candidate_reward_min": min(scores),
    }


def compute_final_score(
    score_mode: str,
    r_plus: float,
    r_minus: float,
    candidate_stats: Dict[str, float],
) -> Tuple[float, str]:
    original_pair_gap = r_plus - r_minus

    if score_mode == "original_pair_gap":
        return original_pair_gap, "R(x,y_plus)-R(x,y_minus)"
    if score_mode == "candidate_mean_gap":
        return r_plus - candidate_stats["candidate_reward_mean"], "R(x,y_plus)-mean_i R(x,y_i)"
    if score_mode == "candidate_max_gap":
        return r_plus - candidate_stats["candidate_reward_max"], "R(x,y_plus)-max_i R(x,y_i)"
    if score_mode == "candidate_min_gap":
        return r_plus - candidate_stats["candidate_reward_min"], "R(x,y_plus)-min_i R(x,y_i)"

    raise ValueError(f"Unknown score_mode: {score_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--reward_base_model", type=str, default="/home/vipuser/Desktop/model/llama2-7b")
    parser.add_argument(
        "--reward_adapter_path",
        type=str,
        default="/mnt/bai_data/yang-safe/output/rm_lora_overfit_member512_e80_lr1e4_ga1_r64_margin8",
    )
    parser.add_argument("--pretrained_llm_path", type=str, default=None)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="candidate_mean_gap",
        choices=[
            "candidate_mean_gap",
            "candidate_max_gap",
            "candidate_min_gap",
            "original_pair_gap",
        ],
    )
    parser.add_argument("--reward_batch_size", type=int, default=1)
    parser.add_argument("--reward_max_length", type=int, default=768)
    parser.add_argument("--target_fpr", type=float, default=5.0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--delta", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_json = os.path.join(args.output_dir, "3.4_reward_gap_ablation_results.json")
    output_summary = os.path.join(args.output_dir, "3.4_reward_gap_ablation_summary.json")
    output_txt = os.path.join(args.output_dir, "3.4_reward_gap_ablation_summary.txt")
    output_table_csv = os.path.join(args.output_dir, "3.4_reward_gap_ablation_table_metrics.csv")

    print("[INFO] Step 4 ablation: reward-gap-only MIA")
    print(f"[INFO] input_path: {args.input_path}")
    print(f"[INFO] model_tag: {args.model_tag}")
    print(f"[INFO] pretrained_llm_path: {args.pretrained_llm_path}")
    print(f"[INFO] reward_base_model: {args.reward_base_model}")
    print(f"[INFO] reward_adapter_path: {args.reward_adapter_path}")
    print(f"[INFO] score_mode: {args.score_mode}")
    print("[INFO] default final_score uses fixed RM and current model's generated candidates.")
    print("[INFO] policy model, policy LoRA, and PPO update are not used.")
    print(f"[INFO] output_dir: {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    data = load_json(args.input_path)
    if args.max_rows is not None:
        data = data[: args.max_rows]
        print(f"[INFO] using max_rows: {len(data)}")
    print(f"[INFO] loaded rows: {len(data)}")

    print("\n[INFO] Loading fixed reward model...")
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_base_model,
        use_fast=False,
        trust_remote_code=True,
    )
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_base_model,
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    reward_model = PeftModel.from_pretrained(reward_model, args.reward_adapter_path)
    reward_model.to(device)
    reward_model.eval()

    plus_pairs = []
    minus_pairs = []
    for item in data:
        plus_pairs.append((item["x"], item.get("y_plus", "")))
        minus_pairs.append((item["x"], item.get("y_minus", "")))

    print("\n[INFO] Scoring y_plus...")
    r_plus_list = score_reward_pairs(
        model=reward_model,
        tokenizer=reward_tokenizer,
        pairs=plus_pairs,
        batch_size=args.reward_batch_size,
        max_length=args.reward_max_length,
        device=device,
    )

    print("\n[INFO] Scoring y_minus...")
    r_minus_list = score_reward_pairs(
        model=reward_model,
        tokenizer=reward_tokenizer,
        pairs=minus_pairs,
        batch_size=args.reward_batch_size,
        max_length=args.reward_max_length,
        device=device,
    )

    results = []
    y_true = []
    final_scores = []

    for idx, item in enumerate(data):
        membership = item.get("membership", "")
        label = 1 if membership == "member" else 0

        r_plus = float(r_plus_list[idx])
        r_minus = float(r_minus_list[idx])
        original_r_gap = r_plus - r_minus
        cand_stats = candidate_reward_stats(item)
        final_score, score_formula = compute_final_score(
            score_mode=args.score_mode,
            r_plus=r_plus,
            r_minus=r_minus,
            candidate_stats=cand_stats,
        )

        y_true.append(label)
        final_scores.append(final_score)

        results.append(
            {
                "id": item.get("id", idx),
                "membership": membership,
                "label": label,
                "source_dataset": item.get("source_dataset"),
                "source_index": item.get("source_index"),
                "model_tag": args.model_tag,
                "pretrained_llm_path": args.pretrained_llm_path,
                "x": item.get("x"),
                "y_plus": item.get("y_plus"),
                "y_minus": item.get("y_minus"),
                "r_plus": r_plus,
                "r_minus": r_minus,
                "original_r_gap": original_r_gap,
                "r_gap": final_score,
                "final_score": final_score,
                "score_mode": args.score_mode,
                "score_formula": score_formula,
                **cand_stats,
            }
        )

    raw_auc = compute_auc(y_true, final_scores)
    score_flipped = raw_auc < 0.5
    eval_scores = [-s for s in final_scores] if score_flipped else list(final_scores)
    auc = 1.0 - raw_auc if score_flipped else raw_auc

    report_fprs = dedupe_fprs([1.0, 5.0, args.target_fpr])
    metrics_by_target_fpr = {
        fpr_metric_key(target_fpr): metrics_at_fpr(
            y_true=y_true,
            y_score=eval_scores,
            target_fpr=target_fpr,
        )
        for target_fpr in report_fprs
    }
    target_key = fpr_metric_key(args.target_fpr)
    metrics_target_fpr = metrics_by_target_fpr[target_key]
    threshold_target_fpr = metrics_target_fpr["threshold"]
    threshold = threshold_target_fpr if args.delta is None else args.delta

    final_metrics = compute_classification_metrics(
        y_true=y_true,
        y_score=eval_scores,
        threshold=threshold,
    )

    best_f1_metrics, best_accuracy_metrics, best_balanced_accuracy_metrics = scan_best_thresholds(
        y_true=y_true,
        y_score=eval_scores,
    )

    table_metrics = {
        "ASR": best_accuracy_metrics["accuracy"],
        "AUC": auc,
        "T@1%F": metrics_by_target_fpr.get("1pct", {}).get("tpr"),
        "T@5%F": metrics_by_target_fpr.get("5pct", {}).get("tpr"),
    }

    summary = {
        "config": {
            "input_path": args.input_path,
            "model_tag": args.model_tag,
            "pretrained_llm_path": args.pretrained_llm_path,
            "reward_base_model": args.reward_base_model,
            "reward_adapter_path": args.reward_adapter_path,
            "score_mode": args.score_mode,
            "target_fpr": args.target_fpr,
            "delta_used_for_final_metrics": threshold,
            "max_rows": args.max_rows,
            "ablation": "reward_gap_only_no_policy_no_ppo",
            "score": results[0]["score_formula"] if results else args.score_mode,
            "note": "original_pair_gap does not depend on pretrained_llm_path; candidate_* modes do.",
        },
        "final_metrics": final_metrics,
        "roc_based_metrics": {
            "auc": auc,
            "raw_auc": raw_auc,
            "score_flipped": int(score_flipped),
            "target_fpr_percent": args.target_fpr,
            "target_fprs": report_fprs,
            "tpr_at_target_fpr": metrics_target_fpr["tpr"],
            "actual_fpr_at_target_fpr": metrics_target_fpr["actual_fpr"],
            "threshold_at_target_fpr": threshold_target_fpr,
            "metrics_at_target_fpr": metrics_target_fpr,
            "metrics_by_target_fpr": metrics_by_target_fpr,
            "tpr_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {}).get("tpr"),
            "acc_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {}).get("accuracy"),
            "actual_fpr_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {}).get("actual_fpr"),
            "threshold_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {}).get("threshold"),
            "tpr_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {}).get("tpr"),
            "acc_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {}).get("accuracy"),
            "actual_fpr_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {}).get("actual_fpr"),
            "threshold_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {}).get("threshold"),
        },
        "table_metrics": table_metrics,
        "best_threshold_metrics": {
            "best_f1": best_f1_metrics,
            "best_accuracy": best_accuracy_metrics,
            "best_balanced_accuracy": best_balanced_accuracy_metrics,
        },
        "score_distribution": summarize_score_distribution(results),
    }

    save_json(results, output_json)
    save_json(summary, output_summary)

    with open(output_table_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ASR", "AUC", "T@1%F", "T@5%F"])
        writer.writeheader()
        writer.writerow(table_metrics)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Reward-gap-only MIA ablation\n")
        f.write(f"score_mode = {args.score_mode}\n")
        f.write(f"final_score = {results[0]['score_formula'] if results else args.score_mode}\n")
        f.write("No policy model, no policy LoRA, no PPO update.\n\n")
        f.write("Table metrics:\n")
        f.write(f"ASR: {table_metrics['ASR']}\n")
        f.write(f"AUC: {table_metrics['AUC']}\n")
        f.write(f"T@1%F: {table_metrics['T@1%F']}\n")
        f.write(f"T@5%F: {table_metrics['T@5%F']}\n\n")
        f.write("Final metrics:\n")
        for k, v in final_metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nROC-based metrics:\n")
        f.write(f"auc: {auc}\n")
        f.write(f"raw_auc: {raw_auc}\n")
        f.write(f"score_flipped: {int(score_flipped)}\n")
        f.write(f"T@1%FPR: {metrics_by_target_fpr.get('1pct', {}).get('tpr')}\n")
        f.write(f"T@5%FPR: {metrics_by_target_fpr.get('5pct', {}).get('tpr')}\n")
        f.write("\nScore distribution:\n")
        f.write(json.dumps(summary["score_distribution"], ensure_ascii=False, indent=2))
        f.write("\n")

    print("\nTable metrics:")
    print(f"ASR: {table_metrics['ASR']}")
    print(f"AUC: {table_metrics['AUC']}")
    print(f"T@1%F: {table_metrics['T@1%F']}")
    print(f"T@5%F: {table_metrics['T@5%F']}")
    print("\n[DONE] Reward-gap-only MIA ablation finished.")
    print(f"[DONE] Results JSON: {output_json}")
    print(f"[DONE] Summary JSON: {output_summary}")
    print(f"[DONE] Summary TXT: {output_txt}")
    print(f"[DONE] Table metrics CSV: {output_table_csv}")


if __name__ == "__main__":
    main()
