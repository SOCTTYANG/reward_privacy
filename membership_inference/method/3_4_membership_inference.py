import json
import csv
import math
import os
import argparse
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

from peft import PeftModel, LoraConfig, get_peft_model, TaskType


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved: {path}")


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def format_reward_text(prompt: str, response: str) -> str:
    return (
        "### Human:\n"
        + str(prompt).strip()
        + "\n\n### Assistant:\n"
        + str(response).strip()
    )


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

    for start in tqdm(range(0, len(pairs), batch_size), desc="Scoring y_plus/y_minus"):
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


def encode_policy_batch(tokenizer, prompt: str, responses: List[str], max_length: int, device):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for response in responses:
        prompt_text = str(prompt).strip()
        response_text = str(response).strip()

        full_text = prompt_text + response_text

        prompt_enc = tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        full_enc = tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        prompt_len = len(prompt_enc["input_ids"])
        prompt_len = min(prompt_len, len(input_ids))

        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len

        if all(x == -100 for x in labels):
            labels[-1] = input_ids[-1]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    features = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(input_ids_list, attention_mask_list)
    ]

    batch = tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt",
    )

    max_len = batch["input_ids"].shape[1]

    padded_labels = []
    for labels in labels_list:
        pad_len = max_len - len(labels)
        padded_labels.append(labels + [-100] * pad_len)

    batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    batch = {k: v.to(device) for k, v in batch.items()}

    return batch


def sequence_logprob(model, input_ids, attention_mask, labels):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    mask = shift_labels.ne(-100)

    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    log_probs = F.log_softmax(shift_logits, dim=-1)

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    token_log_probs = token_log_probs * mask.float()

    lengths = mask.float().sum(dim=-1).clamp(min=1.0)
    seq_log_probs = token_log_probs.sum(dim=-1) / lengths

    return seq_log_probs


def ppo_clipped_loss(new_logp, old_logp, advantages, clip_eps: float):
    log_ratio = new_logp - old_logp
    log_ratio = torch.clamp(log_ratio, min=-20, max=20)

    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    obj1 = ratio * advantages
    obj2 = clipped_ratio * advantages

    loss = -torch.min(obj1, obj2).mean()
    return loss


def compute_grad_norm_for_one_sample(
    policy_model,
    tokenizer,
    item: Dict[str, Any],
    max_length: int,
    clip_eps: float,
    device,
):
    x = item["x"]
    candidates = item.get("candidate_responses", [])

    responses = []
    rewards = []

    for c in candidates:
        y_i = str(c.get("y_i", "")).strip()
        if y_i == "":
            continue
        responses.append(y_i)
        rewards.append(float(c.get("r_i", 0.0)))

    if len(responses) == 0:
        return 0.0, 0.0, 0.0

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = rewards_tensor - rewards_tensor.mean()

    batch = encode_policy_batch(
        tokenizer=tokenizer,
        prompt=x,
        responses=responses,
        max_length=max_length,
        device=device,
    )

    policy_model.zero_grad(set_to_none=True)

    with torch.no_grad():
        old_logp = sequence_logprob(
            policy_model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).detach()

    new_logp = sequence_logprob(
        policy_model,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )

    ppo_loss = ppo_clipped_loss(
        new_logp=new_logp,
        old_logp=old_logp,
        advantages=advantages,
        clip_eps=clip_eps,
    )

    ppo_loss.backward()

    total_norm_sq = 0.0

    for p in policy_model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.detach().float().norm(2).item()
            total_norm_sq += param_norm ** 2

    grad_norm = math.sqrt(total_norm_sq)

    policy_model.zero_grad(set_to_none=True)

    mean_abs_adv = advantages.abs().mean().detach().float().cpu().item()

    return (
        float(grad_norm),
        float(ppo_loss.detach().float().cpu().item()),
        float(mean_abs_adv),
    )


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


def compute_auc(y_true: List[int], y_score: List[float]):
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    rank_sum = 0.0

    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += rank

    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def percentile(values: List[float], q: float):
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


    thresholds = []
    thresholds.append(min(unique_scores) - 1e-6)
    thresholds.extend(unique_scores)
    thresholds.append(max(unique_scores) + 1e-6)

    best_f1 = None
    best_acc = None
    best_bal_acc = None

    for th in thresholds:
        metrics = compute_classification_metrics(y_true, y_score, th)

        tp = metrics["tp"]
        tn = metrics["tn"]
        fp = metrics["fp"]
        fn = metrics["fn"]

        tpr = safe_div(tp, tp + fn)
        tnr = safe_div(tn, tn + fp)
        metrics["balanced_accuracy"] = 0.5 * (tpr + tnr)
        metrics["tpr"] = tpr
        metrics["fpr"] = safe_div(fp, fp + tn)

        if best_f1 is None or metrics["f1"] > best_f1["f1"]:
            best_f1 = metrics

        if best_acc is None or metrics["accuracy"] > best_acc["accuracy"]:
            best_acc = metrics

        if best_bal_acc is None or metrics["balanced_accuracy"] > best_bal_acc["balanced_accuracy"]:
            best_bal_acc = metrics

    return best_f1, best_acc, best_bal_acc


def summarize_score_distribution(results: List[Dict[str, Any]]):
    def stats(values):
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

    member = [x for x in results if x["label"] == 1]
    nonmember = [x for x in results if x["label"] == 0]

    return {
        "member": {
            "final_score": stats([x["final_score"] for x in member]),
            "r_gap": stats([x["r_gap"] for x in member]),
            "grad_norm": stats([x["grad_norm"] for x in member]),
        },
        "nonmember": {
            "final_score": stats([x["final_score"] for x in nonmember]),
            "r_gap": stats([x["r_gap"] for x in nonmember]),
            "grad_norm": stats([x["grad_norm"] for x in nonmember]),
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="/run/media/vipuser/data/yang-safe/membership inference/3.2-Candidate Response Generation/3.2-Candidate Response Generation.json",
    )

    parser.add_argument(
        "--reward_base_model",
        type=str,
        default="/home/vipuser/Desktop/model/llama2-7b",
    )

    parser.add_argument(
        "--reward_adapter_path",
        type=str,
        default="/run/media/vipuser/data/yang-safe/output/rm_lora_overfit_member512_e30/checkpoint-5000",
    )

    parser.add_argument(
        "--policy_base_model",
        type=str,
        default="/home/vipuser/Desktop/model/llama2-7b",
    )

    parser.add_argument(
        "--policy_adapter_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/run/media/vipuser/data/yang-safe/membership inference/3.4-Membership Inference",
    )

    parser.add_argument("--reward_batch_size", type=int, default=1)
    parser.add_argument("--reward_max_length", type=int, default=512)
    parser.add_argument("--policy_max_length", type=int, default=640)

    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument(
        "--score_signal",
        type=str,
        default="combined",
        choices=["combined", "ppo_gradient_only"],
        help=(
            "Membership score signal. combined uses lambda1*r_gap-lambda2*grad_norm; "
            "ppo_gradient_only removes reward gap and uses -lambda2*grad_norm."
        ),
    )
    parser.add_argument(
        "--skip_reward_gap_scoring",
        action="store_true",
        help="Skip y_plus/y_minus reward-gap scoring. Enabled automatically for ppo_gradient_only.",
    )
    parser.add_argument("--target_fpr", type=float, default=1.0)
    # 如果不传 delta，默认使用 1% FPR 阈值。
    parser.add_argument("--delta", type=float, default=None)

    parser.add_argument("--clip_eps", type=float, default=0.2)

    parser.add_argument("--max_rows", type=int, default=None)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    skip_reward_gap_scoring = args.skip_reward_gap_scoring or args.score_signal == "ppo_gradient_only"
    if args.score_signal == "combined" and skip_reward_gap_scoring:
        raise ValueError("--skip_reward_gap_scoring is only valid with --score_signal ppo_gradient_only")

    os.makedirs(args.output_dir, exist_ok=True)

    output_json = os.path.join(args.output_dir, "3.4_membership_inference_results.json")
    output_summary = os.path.join(args.output_dir, "3.4_membership_inference_summary.json")
    output_txt = os.path.join(args.output_dir, "3.4_membership_inference_summary.txt")
    output_table_csv = os.path.join(args.output_dir, "3.4_mia_table_metrics.csv")

    print("[INFO] Step 4: Membership Inference")
    print(f"[INFO] input_path: {args.input_path}")
    print(f"[INFO] reward_base_model: {args.reward_base_model}")
    print(f"[INFO] reward_adapter_path: {args.reward_adapter_path}")
    print(f"[INFO] policy_base_model: {args.policy_base_model}")
    print(f"[INFO] policy_adapter_path: {args.policy_adapter_path}")
    print(f"[INFO] output_dir: {args.output_dir}")
    print(f"[INFO] score_signal: {args.score_signal}")
    print(f"[INFO] skip_reward_gap_scoring: {skip_reward_gap_scoring}")
    print(f"[INFO] lambda1: {args.lambda1}")
    print(f"[INFO] lambda2: {args.lambda2}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    data = load_json(args.input_path)

    if args.max_rows is not None:
        data = data[: args.max_rows]
        print(f"[INFO] using max_rows: {len(data)}")

    print(f"[INFO] loaded rows: {len(data)}")


    if skip_reward_gap_scoring:
        print("\n[INFO] Skipping reward-gap scoring; PPO-gradient is the only membership signal.")
        r_plus_list = [0.0 for _ in data]
        r_minus_list = [0.0 for _ in data]
    else:
        print("\n[INFO] Loading reward model...")

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

        reward_model = PeftModel.from_pretrained(
            reward_model,
            args.reward_adapter_path,
        )

        reward_model.to(device)
        reward_model.eval()

        plus_pairs = []
        minus_pairs = []

        for item in data:
            x = item["x"]
            y_plus = item.get("y_plus", "")
            y_minus = item.get("y_minus", "")

            plus_pairs.append((x, y_plus))
            minus_pairs.append((x, y_minus))

        r_plus_list = score_reward_pairs(
            model=reward_model,
            tokenizer=reward_tokenizer,
            pairs=plus_pairs,
            batch_size=args.reward_batch_size,
            max_length=args.reward_max_length,
            device=device,
        )

        r_minus_list = score_reward_pairs(
            model=reward_model,
            tokenizer=reward_tokenizer,
            pairs=minus_pairs,
            batch_size=args.reward_batch_size,
            max_length=args.reward_max_length,
            device=device,
        )

        del reward_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    print("\n[INFO] Loading policy model for gradient norm...")

    policy_tokenizer = AutoTokenizer.from_pretrained(
        args.policy_base_model,
        use_fast=False,
        trust_remote_code=True,
    )

    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    policy_tokenizer.padding_side = "right"

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy_base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    policy_model.config.pad_token_id = policy_tokenizer.pad_token_id
    policy_model.config.use_cache = False
    policy_model.gradient_checkpointing_enable()

    if args.policy_adapter_path is not None:
        print(f"[INFO] Loading existing policy LoRA: {args.policy_adapter_path}")
        policy_model = PeftModel.from_pretrained(
            policy_model,
            args.policy_adapter_path,
            is_trainable=True,
        )
    else:
        print("[INFO] Creating fresh policy LoRA for gradient calculation...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        )

        policy_model = get_peft_model(policy_model, lora_config)

    policy_model.to(device)
    policy_model.train()

    grad_norms = []
    ppo_losses = []
    mean_abs_advs = []

    for item in tqdm(data, desc="Computing gradient norms"):
        grad_norm, ppo_loss, mean_abs_adv = compute_grad_norm_for_one_sample(
            policy_model=policy_model,
            tokenizer=policy_tokenizer,
            item=item,
            max_length=args.policy_max_length,
            clip_eps=args.clip_eps,
            device=device,
        )

        grad_norms.append(grad_norm)
        ppo_losses.append(ppo_loss)
        mean_abs_advs.append(mean_abs_adv)


    results = []
    y_true = []
    final_scores = []

    for idx, item in enumerate(data):
        membership = item.get("membership", "")
        label = 1 if membership == "member" else 0

        r_plus = float(r_plus_list[idx])
        r_minus = float(r_minus_list[idx])
        r_gap = r_plus - r_minus

        grad_norm = float(grad_norms[idx])
        ppo_loss = float(ppo_losses[idx])
        mean_abs_adv = float(mean_abs_advs[idx])

        if args.score_signal == "ppo_gradient_only":
            final_score = -args.lambda2 * grad_norm
        else:
            final_score = args.lambda1 * r_gap - args.lambda2 * grad_norm

        y_true.append(label)
        final_scores.append(final_score)

        results.append(
            {
                "id": item.get("id", idx),
                "membership": membership,
                "label": label,
                "source_dataset": item.get("source_dataset"),
                "source_index": item.get("source_index"),

                "x": item.get("x"),
                "y_plus": item.get("y_plus"),
                "y_minus": item.get("y_minus"),

                "r_plus": r_plus,
                "r_minus": r_minus,
                "r_gap": r_gap,

                "grad_norm": grad_norm,
                "ppo_loss": ppo_loss,
                "mean_abs_advantage": mean_abs_adv,

                "lambda1": args.lambda1,
                "lambda2": args.lambda2,
                "score_signal": args.score_signal,
                "reward_gap_signal_used": args.score_signal != "ppo_gradient_only",
                "final_score": final_score,
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
    tpr_at_target_fpr = metrics_target_fpr["tpr"]


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

    eps = 1e-12
    total_rows = len(results)

    final_score_zero_count = sum(1 for x in results if abs(x["final_score"]) <= eps)
    grad_norm_zero_count = sum(1 for x in results if abs(x["grad_norm"]) <= eps)
    gap_help_zero_count = sum(1 for x in results if abs(x["r_gap"]) <= eps)


    gap_safe_zero_count = total_rows

    ppo_loss_zero_count = sum(1 for x in results if abs(x["ppo_loss"]) <= eps)

    degenerate_stats = {
        "total_rows": total_rows,
        "final_score_zero_count": final_score_zero_count,
        "final_score_zero_ratio": safe_div(final_score_zero_count, total_rows),
        "grad_norm_zero_count": grad_norm_zero_count,
        "grad_norm_zero_ratio": safe_div(grad_norm_zero_count, total_rows),
        "gap_help_zero_count": gap_help_zero_count,
        "gap_help_zero_ratio": safe_div(gap_help_zero_count, total_rows),
        "gap_safe_zero_count": gap_safe_zero_count,
        "gap_safe_zero_ratio": safe_div(gap_safe_zero_count, total_rows),
        "ppo_loss_zero_count": ppo_loss_zero_count,
        "ppo_loss_zero_ratio": safe_div(ppo_loss_zero_count, total_rows),
    }

    score_distribution = summarize_score_distribution(results)

    summary = {
        "config": {
            "input_path": args.input_path,
            "reward_base_model": args.reward_base_model,
            "reward_adapter_path": args.reward_adapter_path,
            "policy_base_model": args.policy_base_model,
            "policy_adapter_path": args.policy_adapter_path,
            "score_signal": args.score_signal,
            "skip_reward_gap_scoring": skip_reward_gap_scoring,
            "lambda1": args.lambda1,
            "lambda2": args.lambda2,
            "delta_used_for_final_metrics": threshold,
            "clip_eps": args.clip_eps,
            "max_rows": args.max_rows,
        },
        "final_metrics": final_metrics,
        "roc_based_metrics": {
            "auc": auc,
            "raw_auc": raw_auc,
            "score_flipped": int(score_flipped),
            "target_fpr_percent": args.target_fpr,
            "target_fprs": report_fprs,
            "tpr_at_target_fpr": tpr_at_target_fpr,
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
        "score_distribution": score_distribution,
        "degenerate_stats": degenerate_stats,
    }

    save_json(results, output_json)
    save_json(summary, output_summary)

    with open(output_table_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ASR", "AUC", "T@1%F", "T@5%F"])
        writer.writeheader()
        writer.writerow(table_metrics)

    with open(output_txt, "w", encoding="utf-8") as f:
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
        f.write(f"target_fpr_percent: {args.target_fpr}\n")
        f.write(f"tpr_at_target_fpr: {tpr_at_target_fpr}\n")
        f.write(f"actual_fpr_at_target_fpr: {metrics_target_fpr['actual_fpr']}\n")
        f.write(f"threshold_at_target_fpr: {threshold_target_fpr}\n")
        f.write(f"T@1%FPR: {metrics_by_target_fpr.get('1pct', {}).get('tpr')}\n")
        f.write(f"ACC@1%FPR: {metrics_by_target_fpr.get('1pct', {}).get('accuracy')}\n")
        f.write(f"T@5%FPR: {metrics_by_target_fpr.get('5pct', {}).get('tpr')}\n")
        f.write(f"ACC@5%FPR: {metrics_by_target_fpr.get('5pct', {}).get('accuracy')}\n")

        f.write("\nBest-threshold metrics:\n")

        f.write("\nBest F1 metrics:\n")
        for k, v in best_f1_metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\nBest Accuracy metrics:\n")
        for k, v in best_accuracy_metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\nBest Balanced Accuracy metrics:\n")
        for k, v in best_balanced_accuracy_metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\nScore distribution:\n")
        f.write(json.dumps(score_distribution, ensure_ascii=False, indent=2))
        f.write("\n")

        f.write("\nDegenerate stats:\n")
        for k, v in degenerate_stats.items():
            f.write(f"{k}: {v}\n")

    print("\nTable metrics:")
    print(f"ASR: {table_metrics['ASR']}")
    print(f"AUC: {table_metrics['AUC']}")
    print(f"T@1%F: {table_metrics['T@1%F']}")
    print(f"T@5%F: {table_metrics['T@5%F']}")

    print("\nFinal metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")

    print("\nROC-based metrics:")
    print(f"auc: {auc}")
    print(f"raw_auc: {raw_auc}")
    print(f"score_flipped: {int(score_flipped)}")
    print(f"target_fpr_percent: {args.target_fpr}")
    print(f"tpr_at_target_fpr: {tpr_at_target_fpr}")
    print(f"actual_fpr_at_target_fpr: {metrics_target_fpr['actual_fpr']}")
    print(f"threshold_at_target_fpr: {threshold_target_fpr}")
    print(f"T@1%FPR: {metrics_by_target_fpr.get('1pct', {}).get('tpr')}")
    print(f"T@5%FPR: {metrics_by_target_fpr.get('5pct', {}).get('tpr')}")

    print("\nBest-threshold metrics:")

    print("\nBest F1 metrics:")
    for k, v in best_f1_metrics.items():
        print(f"{k}: {v}")

    print("\nBest Accuracy metrics:")
    for k, v in best_accuracy_metrics.items():
        print(f"{k}: {v}")

    print("\nBest Balanced Accuracy metrics:")
    for k, v in best_balanced_accuracy_metrics.items():
        print(f"{k}: {v}")

    print("\nDegenerate stats:")
    for k, v in degenerate_stats.items():
        print(f"{k}: {v}")

    print("\n[DONE] Step 4 Membership Inference finished.")
    print(f"[DONE] Results JSON: {output_json}")
    print(f"[DONE] Summary JSON: {output_summary}")
    print(f"[DONE] Summary TXT: {output_txt}")
    print(f"[DONE] Table metrics CSV: {output_table_csv}")


if __name__ == "__main__":
    main()
