import argparse
import csv
import json
import math
import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()

    if not text:
        return []

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data

    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}")
        rows.append(obj)
    return rows


def get_field(obj: Dict[str, Any], names: Iterable[str], default: Optional[Any] = None) -> Optional[Any]:
    for name in names:
        if name in obj and obj[name] is not None:
            return obj[name]
    return default


def normalize_item(obj: Dict[str, Any], response_field: str) -> Dict[str, Any]:
    x = get_field(obj, ["x", "X", "prompt", "instruction", "question", "query"])
    y_plus = get_field(obj, ["y_plus", "Y+", "chosen", "better", "preferred", "output", "response"])
    y_minus = get_field(obj, ["y_minus", "Y-", "rejected", "worse", "dispreferred"])

    if x is None:
        raise ValueError(f"Missing prompt field in item keys={list(obj.keys())}")

    if response_field == "y_plus":
        response = y_plus
    elif response_field == "y_minus":
        response = y_minus
    elif response_field == "output":
        response = get_field(obj, ["output", "response", "chosen", "y_plus", "Y+"])
    else:
        raise ValueError(f"Unsupported response_field: {response_field}")

    if response is None:
        raise ValueError(f"Missing response field `{response_field}` in item keys={list(obj.keys())}")

    return {
        "x": str(x),
        "response": str(response),
        "y_plus": None if y_plus is None else str(y_plus),
        "y_minus": None if y_minus is None else str(y_minus),
    }


def split_labeled_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    members = []
    nonmembers = []

    for row in rows:
        membership = str(row.get("membership", "")).lower()
        label = row.get("label")
        if membership == "member" or label == 1:
            members.append(row)
        elif membership in {"nonmember", "non-member"} or label == 0:
            nonmembers.append(row)
        else:
            raise ValueError(f"Cannot infer membership for row keys={list(row.keys())}")

    return members, nonmembers


def select_rows(rows: List[Dict[str, Any]], size: int, mode: str, seed: int, name: str) -> List[Dict[str, Any]]:
    if len(rows) < size:
        raise ValueError(f"{name} has only {len(rows)} rows, but {size} were requested")

    if mode == "first":
        return rows[:size]
    if mode == "random":
        rng = random.Random(seed)
        return rng.sample(rows, size)
    raise ValueError(f"Unsupported sample mode: {mode}")


def prepare_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.step2_path:
        rows = read_json_or_jsonl(args.step2_path)
        member_rows, nonmember_rows = split_labeled_rows(rows)
    else:
        if not args.member_path or not args.nonmember_path:
            raise ValueError("Either --step2_path or both --member_path/--nonmember_path are required")
        member_rows = read_json_or_jsonl(args.member_path)
        nonmember_rows = read_json_or_jsonl(args.nonmember_path)

    members = select_rows(member_rows, args.member_size, args.member_sample_mode, args.seed, "member")
    nonmembers = select_rows(
        nonmember_rows,
        args.nonmember_size,
        args.nonmember_sample_mode,
        args.seed,
        "nonmember",
    )

    records = []
    for label, membership, rows in ((1, "member", members), (0, "nonmember", nonmembers)):
        for fallback_index, row in enumerate(rows):
            item = normalize_item(row, args.response_field)
            records.append(
                {
                    "id": row.get("id", f"{membership}_{fallback_index}"),
                    "label": label,
                    "membership": membership,
                    "source_dataset": row.get("source_dataset"),
                    "source_index": row.get("source_index", fallback_index),
                    "x": item["x"],
                    "response": item["response"],
                    "y_plus": item["y_plus"],
                    "y_minus": item["y_minus"],
                    "candidate_responses": row.get("candidate_responses", []),
                }
            )

    return records


def parse_dtype(name: str):
    normalized = str(name).lower()
    if normalized in {"auto", "bf16", "bfloat16"}:
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def load_causal_lm(
    base_model_path: str,
    adapter_path: Optional[str],
    device: torch.device,
    torch_dtype,
    local_files_only: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
            local_files_only=local_files_only,
        )

    model.to(device)
    model.eval()
    return model, tokenizer


def build_prompt(prompt: str, template: str) -> str:
    text = str(prompt).strip()
    normalized = template.lower()

    if normalized == "alpaca":
        return f"### Human:\n{text}\n\n### Assistant:\n"
    if normalized == "chatml":
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    if normalized == "llama3":
        return f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    if normalized == "plain":
        return f"{text}\n"

    raise ValueError(f"Unsupported prompt_template: {template}")


@torch.no_grad()
def conditional_loss(
    model,
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    device: torch.device,
) -> float:
    full_text = prompt + str(response).strip()

    prompt_enc = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    enc = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()

    prompt_len = min(len(prompt_enc["input_ids"]), labels.shape[1])
    labels[:, :prompt_len] = -100
    if labels.ne(-100).sum().item() == 0:
        labels[:, -1] = input_ids[:, -1]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    return float(outputs.loss.detach().float().cpu().item())


class MaskPerturber:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        torch_dtype,
        local_files_only: bool,
        span_length: int,
        pct: float,
        buffer_size: int,
        mask_top_p: float,
        ceil_pct: bool,
        max_fill_length: int,
    ):
        kwargs = {"local_files_only": local_files_only}
        if device.type == "cuda":
            kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.device = device
        self.span_length = span_length
        self.pct = pct
        self.buffer_size = buffer_size
        self.mask_top_p = mask_top_p
        self.ceil_pct = ceil_pct
        self.max_fill_length = max_fill_length
        self.pattern = re.compile(r"<extra_id_\d+>")

    def tokenize_and_mask(self, text: str) -> str:
        tokens = str(text).split()
        if len(tokens) <= self.span_length:
            return str(text)

        mask_string = "<<<mask>>>"
        n_spans = self.pct * len(tokens) / (self.span_length + self.buffer_size * 2)
        n_spans = int(np.ceil(n_spans)) if self.ceil_pct else int(n_spans)
        if n_spans <= 0:
            return str(text)

        n_masks = 0
        max_attempts = max(20, len(tokens) * 2)
        attempts = 0
        while n_masks < n_spans and attempts < max_attempts:
            attempts += 1
            start = random.randint(0, len(tokens) - self.span_length)
            end = start + self.span_length
            search_start = max(0, start - self.buffer_size)
            search_end = min(len(tokens), end + self.buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f"<extra_id_{num_filled}>"
                num_filled += 1

        return " ".join(tokens)

    @staticmethod
    def count_masks(texts: List[str]) -> List[int]:
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

    def replace_masks(self, texts: List[str]) -> List[str]:
        n_expected = self.count_masks(texts)
        if max(n_expected, default=0) == 0:
            return texts

        stop_id = self.tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(
            **tokens,
            max_length=self.max_fill_length,
            do_sample=True,
            top_p=self.mask_top_p,
            num_return_sequences=1,
            eos_token_id=stop_id,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def extract_fills(self, texts: List[str]) -> List[List[str]]:
        cleaned = [text.replace("<pad>", "").replace("</s>", "").strip() for text in texts]
        extracted = [self.pattern.split(text)[1:-1] for text in cleaned]
        return [[fill.strip() for fill in fills] for fills in extracted]

    def apply_extracted_fills(self, masked_texts: List[str], extracted_fills: List[List[str]]) -> List[str]:
        tokens_list = [text.split() for text in masked_texts]
        n_expected = self.count_masks(masked_texts)

        for idx, (tokens, fills, n) in enumerate(zip(tokens_list, extracted_fills, n_expected)):
            if n == 0:
                continue
            if len(fills) < n:
                tokens_list[idx] = []
                continue
            for fill_idx in range(n):
                marker = f"<extra_id_{fill_idx}>"
                if marker in tokens:
                    tokens[tokens.index(marker)] = fills[fill_idx]

        return [" ".join(tokens) for tokens in tokens_list]

    def perturb_one(self, text: str, max_attempts: int = 5) -> str:
        if not str(text).strip():
            return str(text)

        for _ in range(max_attempts):
            masked = self.tokenize_and_mask(text)
            if masked == text:
                return str(text)
            raw_fills = self.replace_masks([masked])
            fills = self.extract_fills(raw_fills)
            perturbed = self.apply_extracted_fills([masked], fills)[0]
            if perturbed.strip():
                return perturbed

        return str(text)


def candidate_perturbations(item: Dict[str, Any], sample_number: int) -> List[str]:
    candidates = []
    for candidate in item.get("candidate_responses", []):
        text = str(candidate.get("y_i", "")).strip()
        if text:
            candidates.append(text)

    if not candidates:
        return [item["response"]] * sample_number

    output = []
    while len(output) < sample_number:
        output.extend(candidates)
    return output[:sample_number]


def compute_spv_scores(
    records: List[Dict[str, Any]],
    target_model,
    target_tokenizer,
    reference_model,
    reference_tokenizer,
    perturber: Optional[MaskPerturber],
    args: argparse.Namespace,
    device: torch.device,
) -> List[Dict[str, Any]]:
    results = []
    use_calibration = reference_model is not None

    for item in tqdm(records, desc="Computing SPV scores"):
        prompt = build_prompt(item["x"], args.prompt_template)
        response = item["response"]

        if args.perturbation_source == "candidates":
            perturbations = candidate_perturbations(item, args.sample_number)
        else:
            assert perturber is not None
            perturbations = [perturber.perturb_one(response) for _ in range(args.sample_number)]

        target_original_loss = conditional_loss(
            target_model,
            target_tokenizer,
            prompt,
            response,
            args.max_length,
            device,
        )
        target_perturbed_losses = [
            conditional_loss(target_model, target_tokenizer, prompt, y, args.max_length, device)
            for y in perturbations
        ]

        target_loss_variation = float(np.mean(target_perturbed_losses) - target_original_loss)
        target_prob_variation = float(np.mean(np.exp(-np.asarray(target_perturbed_losses))) - math.exp(-target_original_loss))

        reference_original_loss = None
        reference_perturbed_losses = None
        reference_loss_variation = 0.0
        reference_prob_variation = 0.0

        if use_calibration:
            reference_original_loss = conditional_loss(
                reference_model,
                reference_tokenizer,
                prompt,
                response,
                args.max_length,
                device,
            )
            reference_perturbed_losses = [
                conditional_loss(reference_model, reference_tokenizer, prompt, y, args.max_length, device)
                for y in perturbations
            ]
            reference_loss_variation = float(np.mean(reference_perturbed_losses) - reference_original_loss)
            reference_prob_variation = float(
                np.mean(np.exp(-np.asarray(reference_perturbed_losses))) - math.exp(-reference_original_loss)
            )

        if args.score_mode == "prob":
            final_score = target_prob_variation - reference_prob_variation
        elif args.score_mode == "loss":
            final_score = target_loss_variation - reference_loss_variation
        else:
            raise ValueError(f"Unsupported score_mode: {args.score_mode}")

        results.append(
            {
                "id": item["id"],
                "membership": item["membership"],
                "label": item["label"],
                "source_dataset": item.get("source_dataset"),
                "source_index": item.get("source_index"),
                "x": item["x"],
                "response": response,
                "y_plus": item.get("y_plus"),
                "y_minus": item.get("y_minus"),
                "perturbation_source": args.perturbation_source,
                "sample_number": args.sample_number,
                "target_original_loss": target_original_loss,
                "target_perturbed_loss_mean": float(np.mean(target_perturbed_losses)),
                "target_loss_variation": target_loss_variation,
                "target_prob_variation": target_prob_variation,
                "reference_original_loss": reference_original_loss,
                "reference_perturbed_loss_mean": None
                if reference_perturbed_losses is None
                else float(np.mean(reference_perturbed_losses)),
                "reference_loss_variation": reference_loss_variation,
                "reference_prob_variation": reference_prob_variation,
                "score_mode": args.score_mode,
                "final_score": float(final_score),
            }
        )

    return results


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    preds = (scores >= threshold).astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())

    tpr = safe_div(tp, tp + fn)
    fpr = safe_div(fp, fp + tn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = tpr
    f1 = safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (tpr + safe_div(tn, tn + fp))

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


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


def parse_target_fprs(value: str, primary_target_fpr: float) -> List[float]:
    parsed = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(float(item))

    if not parsed:
        parsed = [primary_target_fpr]

    if all(abs(x - primary_target_fpr) > 1e-12 for x in parsed):
        parsed.append(primary_target_fpr)

    deduped = []
    for item in parsed:
        if all(abs(item - existing) > 1e-12 for existing in deduped):
            deduped.append(item)
    return deduped


def metrics_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    target_fpr: float,
) -> Dict[str, Any]:
    target_fpr_fraction = normalize_fpr(target_fpr)
    valid = np.where(fpr <= target_fpr_fraction + 1e-12)[0]
    if len(valid) == 0:
        idx = 0
    else:
        best_tpr = np.max(tpr[valid])
        idx = valid[np.where(tpr[valid] == best_tpr)[0][-1]]

    metrics = classification_metrics(labels, scores, float(thresholds[idx]))
    return {
        "target_fpr": target_fpr,
        "target_fpr_fraction": target_fpr_fraction,
        "tpr": float(tpr[idx]),
        "actual_fpr": float(fpr[idx]),
        "threshold": float(thresholds[idx]),
        "classification_metrics": metrics,
    }


def scan_best_thresholds(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    unique_scores = sorted(set(scores.tolist()))
    thresholds = [min(unique_scores) - 1e-12] + unique_scores + [max(unique_scores) + 1e-12]

    best_acc = None
    best_balanced_acc = None
    best_f1 = None

    for threshold in thresholds:
        metrics = classification_metrics(labels, scores, threshold)
        if best_acc is None or metrics["accuracy"] > best_acc["accuracy"]:
            best_acc = metrics
        if best_balanced_acc is None or metrics["balanced_accuracy"] > best_balanced_acc["balanced_accuracy"]:
            best_balanced_acc = metrics
        if best_f1 is None or metrics["f1"] > best_f1["f1"]:
            best_f1 = metrics

    return {
        "best_accuracy": best_acc,
        "best_balanced_accuracy": best_balanced_acc,
        "best_f1": best_f1,
    }


def score_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def compute_summary(
    results: List[Dict[str, Any]],
    target_fprs: List[float],
    primary_target_fpr: float,
) -> Dict[str, Any]:
    labels = np.asarray([row["label"] for row in results], dtype=int)
    raw_scores = np.asarray([row["final_score"] for row in results], dtype=float)

    raw_auc = float(roc_auc_score(labels, raw_scores))
    score_flipped = raw_auc < 0.5
    eval_scores = -raw_scores if score_flipped else raw_scores
    auc = float(1.0 - raw_auc if score_flipped else raw_auc)

    fpr, tpr, thresholds = roc_curve(labels, eval_scores)
    metrics_by_target_fpr = {}
    for target_fpr in target_fprs:
        metrics_by_target_fpr[fpr_metric_key(target_fpr)] = metrics_at_fpr(
            labels=labels,
            scores=eval_scores,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            target_fpr=target_fpr,
        )

    primary_metrics_key = fpr_metric_key(primary_target_fpr)
    primary_metrics = metrics_by_target_fpr[primary_metrics_key]
    best_threshold_metrics = scan_best_thresholds(labels, eval_scores)

    member_scores = [row["final_score"] for row in results if row["label"] == 1]
    nonmember_scores = [row["final_score"] for row in results if row["label"] == 0]

    return {
        "roc_based_metrics": {
            "auc": auc,
            "raw_auc": raw_auc,
            "score_flipped": int(score_flipped),
            "target_fprs": target_fprs,
            "target_fpr": primary_target_fpr,
            "target_fpr_fraction": primary_metrics["target_fpr_fraction"],
            "tpr_at_target_fpr": primary_metrics["tpr"],
            "actual_fpr_at_target_fpr": primary_metrics["actual_fpr"],
            "threshold_at_target_fpr": primary_metrics["threshold"],
            "metrics_at_target_fpr": primary_metrics["classification_metrics"],
            "metrics_by_target_fpr": metrics_by_target_fpr,
            "tpr_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {}).get("tpr"),
            "acc_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {})
            .get("classification_metrics", {})
            .get("accuracy"),
            "actual_fpr_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {}).get("actual_fpr"),
            "threshold_at_1pct_fpr": metrics_by_target_fpr.get("1pct", {}).get("threshold"),
            "tpr_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {}).get("tpr"),
            "acc_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {})
            .get("classification_metrics", {})
            .get("accuracy"),
            "actual_fpr_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {}).get("actual_fpr"),
            "threshold_at_5pct_fpr": metrics_by_target_fpr.get("5pct", {}).get("threshold"),
        },
        "best_threshold_metrics": best_threshold_metrics,
        "score_distribution": {
            "member_final_score": score_stats(member_scores),
            "nonmember_final_score": score_stats(nonmember_scores),
        },
    }


def write_outputs(args: argparse.Namespace, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    results_jsonl = os.path.join(args.output_dir, "spv_mia_results.jsonl")
    summary_json = os.path.join(args.output_dir, "spv_mia_summary.json")
    summary_txt = os.path.join(args.output_dir, "spv_mia_summary.txt")
    summary_csv = os.path.join(args.output_dir, "spv_mia_summary_row.csv")

    with open(results_jsonl, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    full_summary = {
        "config": vars(args),
        **summary,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)

    roc = summary["roc_based_metrics"]
    best = summary["best_threshold_metrics"]
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("SPV-MIA baseline on Safe-RLHF membership steps\n")
        f.write(f"model_tag: {args.model_tag}\n")
        f.write(f"auc: {roc['auc']}\n")
        f.write(f"raw_auc: {roc['raw_auc']}\n")
        f.write(f"score_flipped: {roc['score_flipped']}\n")
        f.write("\nLow-FPR metrics:\n")
        for key, metrics in roc["metrics_by_target_fpr"].items():
            f.write(f"T@{metrics['target_fpr']}%FPR: {metrics['tpr']}\n")
            f.write(
                f"ACC@{metrics['target_fpr']}%FPR: "
                f"{metrics['classification_metrics']['accuracy']}\n"
            )
            f.write(f"actual_fpr@{metrics['target_fpr']}%FPR: {metrics['actual_fpr']}\n")
            f.write(f"threshold@{metrics['target_fpr']}%FPR: {metrics['threshold']}\n")
        f.write("\nPrimary target FPR:\n")
        f.write(f"target_fpr: {roc['target_fpr']}\n")
        f.write(f"tpr_at_target_fpr: {roc['tpr_at_target_fpr']}\n")
        f.write(f"actual_fpr_at_target_fpr: {roc['actual_fpr_at_target_fpr']}\n")
        f.write(f"threshold_at_target_fpr: {roc['threshold_at_target_fpr']}\n")
        f.write("\nMetrics at target FPR:\n")
        for key, value in roc["metrics_at_target_fpr"].items():
            f.write(f"{key}: {value}\n")
        f.write("\nBest accuracy:\n")
        for key, value in best["best_accuracy"].items():
            f.write(f"{key}: {value}\n")
        f.write("\nBest balanced accuracy:\n")
        for key, value in best["best_balanced_accuracy"].items():
            f.write(f"{key}: {value}\n")
        f.write("\nBest F1:\n")
        for key, value in best["best_f1"].items():
            f.write(f"{key}: {value}\n")

    csv_row = {
        "model_tag": args.model_tag,
        "auc": roc["auc"],
        "raw_auc": roc["raw_auc"],
        "score_flipped": roc["score_flipped"],
        "target_fpr": roc["target_fpr"],
        "tpr_at_target_fpr": roc["tpr_at_target_fpr"],
        "actual_fpr_at_target_fpr": roc["actual_fpr_at_target_fpr"],
        "tpr_at_1pct_fpr": roc.get("tpr_at_1pct_fpr"),
        "acc_at_1pct_fpr": roc.get("acc_at_1pct_fpr"),
        "actual_fpr_at_1pct_fpr": roc.get("actual_fpr_at_1pct_fpr"),
        "threshold_at_1pct_fpr": roc.get("threshold_at_1pct_fpr"),
        "tpr_at_5pct_fpr": roc.get("tpr_at_5pct_fpr"),
        "acc_at_5pct_fpr": roc.get("acc_at_5pct_fpr"),
        "actual_fpr_at_5pct_fpr": roc.get("actual_fpr_at_5pct_fpr"),
        "threshold_at_5pct_fpr": roc.get("threshold_at_5pct_fpr"),
        "best_accuracy": best["best_accuracy"]["accuracy"],
        "best_balanced_accuracy": best["best_balanced_accuracy"]["balanced_accuracy"],
        "best_f1": best["best_f1"]["f1"],
        "output_dir": args.output_dir,
    }
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        writer.writeheader()
        writer.writerow(csv_row)

    print(f"[DONE] Results JSONL: {results_jsonl}")
    print(f"[DONE] Summary JSON : {summary_json}")
    print(f"[DONE] Summary TXT  : {summary_txt}")
    print(f"[DONE] Summary CSV  : {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPV-MIA on Safe-RLHF membership-step outputs.")
    parser.add_argument("--step2_path", default=None, help="3.2 candidate response JSON from membership inference.")
    parser.add_argument("--member_path", default=None)
    parser.add_argument("--nonmember_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_tag", default="model")

    parser.add_argument("--target_base_model", required=True)
    parser.add_argument("--target_adapter_path", default=None)
    parser.add_argument("--reference_base_model", default=None)
    parser.add_argument("--reference_adapter_path", default=None)
    parser.add_argument("--disable_calibration", action="store_true")

    parser.add_argument("--response_field", choices=["y_plus", "y_minus", "output"], default="y_plus")
    parser.add_argument("--prompt_template", choices=["alpaca", "chatml", "llama3", "plain"], default="alpaca")
    parser.add_argument("--member_size", type=int, default=512)
    parser.add_argument("--nonmember_size", type=int, default=512)
    parser.add_argument("--member_sample_mode", choices=["first", "random"], default="first")
    parser.add_argument("--nonmember_sample_mode", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--perturbation_source", choices=["mask", "candidates"], default="mask")
    parser.add_argument("--mask_filling_model_name", default="t5-base")
    parser.add_argument("--sample_number", type=int, default=10)
    parser.add_argument("--span_length", type=int, default=2)
    parser.add_argument("--pct", type=float, default=0.3)
    parser.add_argument("--buffer_size", type=int, default=1)
    parser.add_argument("--mask_top_p", type=float, default=1.0)
    parser.add_argument("--ceil_pct", action="store_true")
    parser.add_argument("--max_fill_length", type=int, default=150)

    parser.add_argument("--score_mode", choices=["prob", "loss"], default="prob")
    parser.add_argument("--target_fpr", type=float, default=5.0)
    parser.add_argument("--target_fprs", default="1.0,5.0")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch_dtype = parse_dtype(args.torch_dtype)

    print("[INFO] SPV-MIA baseline on Safe-RLHF membership steps")
    print(f"[INFO] model_tag           : {args.model_tag}")
    print(f"[INFO] step2_path          : {args.step2_path}")
    print(f"[INFO] target_base_model   : {args.target_base_model}")
    print(f"[INFO] target_adapter_path : {args.target_adapter_path}")
    print(f"[INFO] reference_base_model: {args.reference_base_model}")
    print(f"[INFO] reference_adapter   : {args.reference_adapter_path}")
    print(f"[INFO] output_dir          : {args.output_dir}")
    print(f"[INFO] device              : {device}")

    records = prepare_records(args)
    print(f"[INFO] records             : {len(records)}")
    print(f"[INFO] members/nonmembers  : {sum(r['label'] == 1 for r in records)}/{sum(r['label'] == 0 for r in records)}")

    target_model, target_tokenizer = load_causal_lm(
        args.target_base_model,
        args.target_adapter_path,
        device,
        torch_dtype,
        args.local_files_only,
    )

    reference_model = None
    reference_tokenizer = None
    if not args.disable_calibration:
        reference_base = args.reference_base_model or args.target_base_model
        reference_model, reference_tokenizer = load_causal_lm(
            reference_base,
            args.reference_adapter_path,
            device,
            torch_dtype,
            args.local_files_only,
        )

    perturber = None
    if args.perturbation_source == "mask":
        perturber = MaskPerturber(
            model_name=args.mask_filling_model_name,
            device=device,
            torch_dtype=torch_dtype,
            local_files_only=args.local_files_only,
            span_length=args.span_length,
            pct=args.pct,
            buffer_size=args.buffer_size,
            mask_top_p=args.mask_top_p,
            ceil_pct=args.ceil_pct,
            max_fill_length=args.max_fill_length,
        )

    results = compute_spv_scores(
        records=records,
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        reference_model=reference_model,
        reference_tokenizer=reference_tokenizer,
        perturber=perturber,
        args=args,
        device=device,
    )
    target_fprs = parse_target_fprs(args.target_fprs, args.target_fpr)
    summary = compute_summary(results, target_fprs, args.target_fpr)
    write_outputs(args, results, summary)

    roc = summary["roc_based_metrics"]
    print("\n========== SPV-MIA Summary ==========")
    print(f"auc: {roc['auc']}")
    print(f"raw_auc: {roc['raw_auc']}")
    print(f"score_flipped: {roc['score_flipped']}")
    print(f"T@1%FPR: {roc.get('tpr_at_1pct_fpr')}")
    print(f"ACC@1%FPR: {roc.get('acc_at_1pct_fpr')}")
    print(f"T@5%FPR: {roc.get('tpr_at_5pct_fpr')}")
    print(f"ACC@5%FPR: {roc.get('acc_at_5pct_fpr')}")
    print(f"target_fpr: {roc['target_fpr']}")
    print(f"tpr_at_target_fpr: {roc['tpr_at_target_fpr']}")
    print("[DONE] SPV-MIA baseline finished.")


if __name__ == "__main__":
    main()
