#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MiniLLM-style Reward Model Extraction Baseline

Core idea:
For each prompt with K=2 candidate responses, convert target RM scores and
substitute RM scores into soft preference distributions, then minimize:

    KL[p_sub || p_target]

This adapts MiniLLM reverse-KL distillation to reward model extraction.
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def get_prompt(row: Dict) -> str:
    for k in ["prompt", "instruction", "query", "question"]:
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def parse_pair(row: Dict) -> Tuple[str, str, str, Optional[int]]:
    """
    Return:
        prompt, response_0, response_1, gold_idx

    gold_idx:
        0 means response_0 is preferred
        1 means response_1 is preferred
        None means no human label found
    """

    prompt = get_prompt(row)

    # Format 1: chosen / rejected
    if "chosen" in row and "rejected" in row:
        return prompt, str(row["chosen"]), str(row["rejected"]), 0

    # Format 2: response_0 / response_1 + better_response_id
    if "response_0" in row and "response_1" in row:
        r0 = str(row["response_0"])
        r1 = str(row["response_1"])

        gold_idx = None
        for key in ["better_response_id", "safer_response_id", "preferred_response_id", "label"]:
            if key in row:
                try:
                    gold_idx = int(row[key])
                    if gold_idx not in [0, 1]:
                        gold_idx = None
                except Exception:
                    gold_idx = None
                break

        return prompt, r0, r1, gold_idx

    raise ValueError(f"Cannot parse pair fields from row keys: {list(row.keys())}")


def build_text(prompt: str, response: str) -> str:
    return f"Prompt:\n{prompt}\n\nResponse:\n{response}"


class PairPreferenceDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        rows = read_jsonl(path, max_samples=max_samples)
        self.examples = []

        for row in rows:
            prompt, r0, r1, gold_idx = parse_pair(row)
            self.examples.append({
                "prompt": prompt,
                "response_0": r0,
                "response_1": r1,
                "gold_idx": gold_idx,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "prompts": [x["prompt"] for x in batch],
        "response_0": [x["response_0"] for x in batch],
        "response_1": [x["response_1"] for x in batch],
        "gold_idx": [x["gold_idx"] for x in batch],
    }


def load_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(
        path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_target_rm(base_model_path: str, adapter_path: str, dtype):
    tokenizer = load_tokenizer(base_model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return tokenizer, model


def load_student_rm(student_model_path: str, dtype, device: str):
    tokenizer = load_tokenizer(student_model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        student_model_path,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.train()

    return tokenizer, model


def encode_pair_texts(
    tokenizer,
    prompts: List[str],
    response_0: List[str],
    response_1: List[str],
    max_length: int,
    device: str,
):
    texts = []
    for p, r0, r1 in zip(prompts, response_0, response_1):
        texts.append(build_text(p, r0))
        texts.append(build_text(p, r1))

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def score_pairs(
    model,
    tokenizer,
    prompts: List[str],
    response_0: List[str],
    response_1: List[str],
    max_length: int,
    device: str,
):
    """
    Return scores shaped [batch_size, 2]
    """
    inputs = encode_pair_texts(
        tokenizer=tokenizer,
        prompts=prompts,
        response_0=response_0,
        response_1=response_1,
        max_length=max_length,
        device=device,
    )

    outputs = model(**inputs)
    scores = outputs.logits.view(-1).float()
    scores = scores.view(len(prompts), 2)
    return scores


def reverse_kl_loss(sub_scores: torch.Tensor, target_scores: torch.Tensor, tau: float):
    """
    KL[p_sub || p_target]

    sub_scores:    [B, 2]
    target_scores: [B, 2]
    """
    log_p_sub = F.log_softmax(sub_scores / tau, dim=-1)
    p_sub = torch.softmax(sub_scores / tau, dim=-1)

    log_p_target = F.log_softmax(target_scores / tau, dim=-1)

    loss = torch.sum(p_sub * (log_p_sub - log_p_target), dim=-1).mean()
    return loss


@torch.no_grad()
def evaluate(
    target_model,
    target_tokenizer,
    student_model,
    student_tokenizer,
    dataloader,
    max_length: int,
    tau: float,
    device: str,
    split_name: str,
):
    student_model.eval()

    losses = []
    mse_list = []
    abs_diff_list = []

    target_pref_correct = 0
    student_pref_correct = 0
    agreement_correct = 0
    gold_count = 0
    total = 0

    for batch in tqdm(dataloader, desc=f"Eval {split_name}"):
        prompts = batch["prompts"]
        r0 = batch["response_0"]
        r1 = batch["response_1"]
        gold_idx = batch["gold_idx"]

        target_scores = score_pairs(
            target_model,
            target_tokenizer,
            prompts,
            r0,
            r1,
            max_length=max_length,
            device=device,
        ).detach()

        student_scores = score_pairs(
            student_model,
            student_tokenizer,
            prompts,
            r0,
            r1,
            max_length=max_length,
            device=device,
        )

        loss = reverse_kl_loss(student_scores, target_scores, tau=tau)
        losses.append(float(loss.item()))

        mse = F.mse_loss(student_scores, target_scores).item()
        mse_list.append(float(mse))

        abs_diff = torch.abs(student_scores - target_scores).mean(dim=-1)
        abs_diff_list.extend(abs_diff.detach().cpu().numpy().tolist())

        target_pref = torch.argmax(target_scores, dim=-1).detach().cpu().numpy().tolist()
        student_pref = torch.argmax(student_scores, dim=-1).detach().cpu().numpy().tolist()

        for tp, sp, g in zip(target_pref, student_pref, gold_idx):
            total += 1
            if tp == sp:
                agreement_correct += 1

            if g is not None:
                gold_count += 1
                if tp == g:
                    target_pref_correct += 1
                if sp == g:
                    student_pref_correct += 1

    result = {
        "split": split_name,
        "num_items": total,
        "reverse_kl": float(np.mean(losses)) if losses else None,
        "score_mse": float(np.mean(mse_list)) if mse_list else None,
        "Diff_avg": float(np.mean(abs_diff_list)) if abs_diff_list else None,
        "Diff_var": float(np.var(abs_diff_list)) if abs_diff_list else None,
        "target_student_agreement_acc": agreement_correct / total if total else None,
        "gold_count": gold_count,
        "target_gold_acc": target_pref_correct / gold_count if gold_count else None,
        "student_gold_acc": student_pref_correct / gold_count if gold_count else None,
    }

    student_model.train()
    return result


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def train(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================")
    print("MiniLLM-RM Reverse-KL Distillation")
    print("Target base model:    ", args.target_base_model)
    print("Target adapter:       ", args.target_adapter_path)
    print("Student model:        ", args.student_model_path)
    print("Train path:           ", args.train_path)
    print("Eval path:            ", args.eval_path)
    print("Output dir:           ", args.output_dir)
    print("Device:               ", device)
    print("dtype:                ", dtype)
    print("tau:                  ", args.tau)
    print("======================================================")

    print("[INFO] Loading frozen target RM...")
    target_tokenizer, target_model = load_target_rm(
        args.target_base_model,
        args.target_adapter_path,
        dtype=dtype,
    )

    print("[INFO] Loading trainable substitute RM...")
    student_tokenizer, student_model = load_student_rm(
        args.student_model_path,
        dtype=dtype,
        device=device,
    )

    train_dataset = PairPreferenceDataset(
        args.train_path,
        max_samples=args.max_train_samples,
    )
    eval_dataset = PairPreferenceDataset(
        args.eval_path,
        max_samples=args.max_eval_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print("[INFO] train samples =", len(train_dataset))
    print("[INFO] eval samples  =", len(eval_dataset))

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = num_update_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    metrics = {
        "args": vars(args),
        "eval_before": None,
        "epochs": [],
        "eval_after": None,
    }

    print("========== Before Training Eval ==========")
    before = evaluate(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        dataloader=eval_loader,
        max_length=args.max_length,
        tau=args.tau,
        device=device,
        split_name="eval_before",
    )
    print(json.dumps(before, ensure_ascii=False, indent=2))
    metrics["eval_before"] = before
    save_json(metrics, output_dir / "minillm_rm_metrics.json")

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        student_model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar, start=1):
            prompts = batch["prompts"]
            r0 = batch["response_0"]
            r1 = batch["response_1"]

            with torch.no_grad():
                target_scores = score_pairs(
                    target_model,
                    target_tokenizer,
                    prompts,
                    r0,
                    r1,
                    max_length=args.max_length,
                    device=device,
                ).detach()

            student_scores = score_pairs(
                student_model,
                student_tokenizer,
                prompts,
                r0,
                r1,
                max_length=args.max_length,
                device=device,
            )

            loss = reverse_kl_loss(student_scores, target_scores, tau=args.tau)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            epoch_losses.append(float(loss.item() * args.gradient_accumulation_steps))

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                if args.grad_clip is not None and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_postfix({
                "loss": f"{np.mean(epoch_losses[-20:]):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        print(f"[Epoch {epoch}] train_reverse_kl={np.mean(epoch_losses):.6f}")

        eval_result = evaluate(
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            student_model=student_model,
            student_tokenizer=student_tokenizer,
            dataloader=eval_loader,
            max_length=args.max_length,
            tau=args.tau,
            device=device,
            split_name=f"eval_epoch_{epoch}",
        )
        print(json.dumps(eval_result, ensure_ascii=False, indent=2))

        epoch_record = {
            "epoch": epoch,
            "train_reverse_kl": float(np.mean(epoch_losses)),
            "eval": eval_result,
        }
        metrics["epochs"].append(epoch_record)
        save_json(metrics, output_dir / "minillm_rm_metrics.json")

        if args.save_each_epoch:
            epoch_dir = output_dir / f"checkpoint_epoch_{epoch}"
            student_model.save_pretrained(epoch_dir)
            student_tokenizer.save_pretrained(epoch_dir)
            print("[OK] checkpoint saved to:", epoch_dir)

    print("========== Final Eval ==========")
    final_eval = evaluate(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        dataloader=eval_loader,
        max_length=args.max_length,
        tau=args.tau,
        device=device,
        split_name="eval_after",
    )
    print(json.dumps(final_eval, ensure_ascii=False, indent=2))
    metrics["eval_after"] = final_eval

    print("[INFO] Saving final substitute RM...")
    student_model.save_pretrained(output_dir)
    student_tokenizer.save_pretrained(output_dir)

    save_json(metrics, output_dir / "minillm_rm_metrics.json")

    print("======================================================")
    print("[OK] final MiniLLM-RM substitute saved to:", output_dir)
    print("[OK] metrics saved to:", output_dir / "minillm_rm_metrics.json")
    print("======================================================")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_base_model", required=True)
    parser.add_argument("--target_adapter_path", required=True)
    parser.add_argument("--student_model_path", required=True)

    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--max_eval_samples", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save_each_epoch", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
