from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_text(prompt: str, response: str) -> str:
    return f"### Prompt:\n{prompt}\n\n### Response:\n{response}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class PreferenceDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        raw = read_jsonl(path, max_samples=max_samples)
        self.data = []

        for item in raw:
            if "prompt" not in item or "chosen" not in item or "rejected" not in item:
                raise KeyError(f"Need prompt/chosen/rejected, got keys: {list(item.keys())}")

            prompt = str(item["prompt"])
            chosen = str(item["chosen"])
            rejected = str(item["rejected"])

            if not prompt.strip() or not chosen.strip() or not rejected.strip():
                continue

            self.data.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class ScoredAuxDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        raw = read_jsonl(path, max_samples=max_samples)
        self.data = []

        for item in raw:
            if "prompt" not in item or "response" not in item or "target_score" not in item:
                raise KeyError(f"Need prompt/response/target_score, got keys: {list(item.keys())}")

            prompt = str(item["prompt"])
            response = str(item["response"])
            target_score = float(item["target_score"])

            if not prompt.strip() or not response.strip():
                continue

            self.data.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "target_score": target_score,
                    "category": str(item.get("category", "")),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class PreferenceCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_texts = [build_text(x["prompt"], x["chosen"]) for x in batch]
        rejected_texts = [build_text(x["prompt"], x["rejected"]) for x in batch]

        chosen = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        rejected = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }


class ScoredAuxCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [build_text(x["prompt"], x["response"]) for x in batch]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        target_scores = torch.tensor(
            [float(x["target_score"]) for x in batch],
            dtype=torch.float32,
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "target_scores": target_scores,
        }


def compute_pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0

    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)

    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)

    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0

    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


@torch.no_grad()
def evaluate_preference(model, dataloader: DataLoader, device: str, desc: str) -> Dict[str, float]:
    model.eval()

    total = 0
    correct = 0
    gap_sum = 0.0
    loss_sum = 0.0

    for batch in tqdm(dataloader, desc=desc):
        batch = move_to_device(batch, device)

        chosen_scores = model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        ).logits.squeeze(-1)

        rejected_scores = model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        ).logits.squeeze(-1)

        gaps = chosen_scores - rejected_scores
        loss = -F.logsigmoid(gaps.float()).mean()

        correct += (gaps > 0).long().sum().item()
        total += gaps.numel()
        gap_sum += gaps.detach().float().sum().item()
        loss_sum += loss.item() * gaps.numel()

    return {
        "preference_acc": correct / max(total, 1),
        "avg_gap": gap_sum / max(total, 1),
        "ranking_loss": loss_sum / max(total, 1),
    }


@torch.no_grad()
def evaluate_regression(model, dataloader: DataLoader, device: str, desc: str) -> Dict[str, float]:
    model.eval()

    preds = []
    targets = []

    for batch in tqdm(dataloader, desc=desc):
        batch = move_to_device(batch, device)

        pred_scores = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.squeeze(-1)

        preds.extend(pred_scores.detach().float().cpu().tolist())
        targets.extend(batch["target_scores"].detach().float().cpu().tolist())

    n = len(preds)
    if n == 0:
        return {"mae": 0.0, "rmse": 0.0, "pearson": 0.0}

    mae = sum(abs(p - t) for p, t in zip(preds, targets)) / n
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, targets)) / n)
    pearson = compute_pearson(preds, targets)

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
    }


def make_optimizer_and_scheduler(model, lr: float, weight_decay: float, total_steps: int, warmup_ratio: float):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    return optimizer, scheduler


def train_stage_a(model, train_loader, eval_loader, args) -> Dict[str, Any]:
    total_steps = max((len(train_loader) * args.pref_epochs) // args.gradient_accumulation_steps, 1)

    optimizer, scheduler = make_optimizer_and_scheduler(
        model=model,
        lr=args.pref_lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    metrics = {"before": None, "epochs": [], "after": None}

    if eval_loader is not None:
        print("========== Stage A Before Eval: Attacker Preference Preference ==========")
        before = evaluate_preference(model, eval_loader, args.device, "Stage A before eval")
        print(before)
        metrics["before"] = before

    for epoch in range(args.pref_epochs):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Stage A Attacker Preference Preference epoch {epoch + 1}")

        for step, batch in enumerate(pbar):
            batch = move_to_device(batch, args.device)

            chosen_scores = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            ).logits.squeeze(-1)

            rejected_scores = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            ).logits.squeeze(-1)

            gaps = chosen_scores - rejected_scores
            loss = -F.logsigmoid(gaps.float()).mean()
            (loss / args.gradient_accumulation_steps).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            bs = chosen_scores.size(0)
            total_loss += loss.item() * bs
            total_count += bs

            pbar.set_postfix(
                {
                    "rank_loss": f"{loss.item():.4f}",
                    "gap": f"{gaps.detach().float().mean().item():.4f}",
                }
            )

        avg_loss = total_loss / max(total_count, 1)
        record = {"epoch": epoch + 1, "train_ranking_loss": avg_loss}
        print(f"[Stage A] epoch={epoch + 1}, train_ranking_loss={avg_loss:.6f}")

        if eval_loader is not None:
            eval_metrics = evaluate_preference(model, eval_loader, args.device, f"Stage A eval epoch {epoch + 1}")
            print(f"[Stage A Eval] epoch={epoch + 1}, metrics={eval_metrics}")
            record["eval"] = eval_metrics

        metrics["epochs"].append(record)

    if eval_loader is not None:
        print("========== Stage A After Eval: Attacker Preference Preference ==========")
        after = evaluate_preference(model, eval_loader, args.device, "Stage A after eval")
        print(after)
        metrics["after"] = after

    return metrics


def train_stage_b(model, train_loader, eval_loader, args) -> Dict[str, Any]:
    total_steps = max((len(train_loader) * args.distill_epochs) // args.gradient_accumulation_steps, 1)

    optimizer, scheduler = make_optimizer_and_scheduler(
        model=model,
        lr=args.distill_lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    metrics = {"before": None, "epochs": [], "after": None}

    print("========== Stage B Before Eval: LLaMA Scored Auxiliary ==========")
    before = evaluate_regression(model, eval_loader, args.device, "Stage B before eval")
    print(before)
    metrics["before"] = before

    for epoch in range(args.distill_epochs):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Stage B Distillation epoch {epoch + 1}")

        for step, batch in enumerate(pbar):
            batch = move_to_device(batch, args.device)

            pred_scores = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits.squeeze(-1)

            target_scores = batch["target_scores"]
            loss = F.mse_loss(pred_scores.float(), target_scores.float())

            (loss / args.gradient_accumulation_steps).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            bs = pred_scores.size(0)
            total_loss += loss.item() * bs
            total_count += bs

            pbar.set_postfix(
                {
                    "mse": f"{loss.item():.4f}",
                    "pred": f"{pred_scores.detach().float().mean().item():.4f}",
                    "target": f"{target_scores.detach().float().mean().item():.4f}",
                }
            )

        avg_loss = total_loss / max(total_count, 1)
        record = {"epoch": epoch + 1, "train_mse_loss": avg_loss}
        print(f"[Stage B] epoch={epoch + 1}, train_mse_loss={avg_loss:.6f}")

        eval_metrics = evaluate_regression(model, eval_loader, args.device, f"Stage B eval epoch {epoch + 1}")
        print(f"[Stage B Eval] epoch={epoch + 1}, metrics={eval_metrics}")
        record["eval"] = eval_metrics

        metrics["epochs"].append(record)

    print("========== Stage B After Eval: LLaMA Scored Auxiliary ==========")
    after = evaluate_regression(model, eval_loader, args.device, "Stage B after eval")
    print(after)
    metrics["after"] = after

    return metrics


def train(args) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] LLaMA2-7B LoRA two-stage substitute RM training")
    print(f"[INFO] student base model: {args.student_model_path}")
    print(f"[INFO] scored aux path:    {args.scored_aux_path}")
    print(f"[INFO] output dir:         {args.output_dir}")
    print(f"[INFO] device:             {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.student_model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        device_map=None,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=["score"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(args.device)

    pref_collator = PreferenceCollator(tokenizer, args.max_length)
    aux_collator = ScoredAuxCollator(tokenizer, args.max_length)

    attacker_preference_train = PreferenceDataset(args.attacker_preference_dataset_train_path, args.max_attacker_preference_train_samples)
    attacker_preference_eval = PreferenceDataset(args.attacker_preference_dataset_eval_path, args.max_attacker_preference_eval_samples)

    attacker_preference_train_loader = DataLoader(
        attacker_preference_train,
        batch_size=args.pref_batch_size,
        shuffle=True,
        collate_fn=pref_collator,
        num_workers=0,
    )

    attacker_preference_eval_loader = DataLoader(
        attacker_preference_eval,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=pref_collator,
        num_workers=0,
    )

    aux_all = ScoredAuxDataset(args.scored_aux_path, args.max_aux_samples)
    aux_train_size = int(len(aux_all) * args.aux_train_ratio)
    aux_eval_size = len(aux_all) - aux_train_size

    aux_train, aux_eval = random_split(
        aux_all,
        [aux_train_size, aux_eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    aux_train_loader = DataLoader(
        aux_train,
        batch_size=args.distill_batch_size,
        shuffle=True,
        collate_fn=aux_collator,
        num_workers=0,
    )

    aux_eval_loader = DataLoader(
        aux_eval,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=aux_collator,
        num_workers=0,
    )

    defender_evaluation_eval_loader = None
    if args.defender_eval_eval_path is not None:
        defender_evaluation_eval = PreferenceDataset(args.defender_eval_eval_path, args.max_defender_evaluation_eval_samples)
        defender_evaluation_eval_loader = DataLoader(
            defender_evaluation_eval,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=pref_collator,
            num_workers=0,
        )

    print(f"[INFO] Attacker Preference train samples = {len(attacker_preference_train)}")
    print(f"[INFO] Attacker Preference eval samples  = {len(attacker_preference_eval)}")
    print(f"[INFO] aux total samples = {len(aux_all)}")
    print(f"[INFO] aux train samples = {len(aux_train)}")
    print(f"[INFO] aux eval samples  = {len(aux_eval)}")

    final_metrics: Dict[str, Any] = {}

    if defender_evaluation_eval_loader is not None:
        print("========== Initial Eval: Defender Evaluation Preference ==========")
        initial_defender_evaluation = evaluate_preference(model, defender_evaluation_eval_loader, args.device, "Initial Defender Evaluation eval")
        print(initial_defender_evaluation)
        final_metrics["initial_defender_evalerence"] = initial_defender_evaluation

    print("\n================ Stage A: Attacker Preference Preference Pretraining ================")
    stage_a_metrics = train_stage_a(model, attacker_preference_train_loader, attacker_preference_eval_loader, args)
    final_metrics["stage_a_attacker_preference_dataseterence"] = stage_a_metrics

    stage_a_dir = os.path.join(args.output_dir, "after_stage_a_attacker_preference_dataseterence")
    os.makedirs(stage_a_dir, exist_ok=True)
    model.save_pretrained(stage_a_dir)
    tokenizer.save_pretrained(stage_a_dir)
    print(f"[OK] Stage A checkpoint saved to: {stage_a_dir}")

    if defender_evaluation_eval_loader is not None:
        print("========== After Stage A Eval: Defender Evaluation Preference ==========")
        after_a_defender_evaluation = evaluate_preference(model, defender_evaluation_eval_loader, args.device, "After Stage A Defender Evaluation eval")
        print(after_a_defender_evaluation)
        final_metrics["after_stage_a_defender_evalerence"] = after_a_defender_evaluation

    print("\n================ Stage B: LLaMA Scored Auxiliary Distillation ================")
    stage_b_metrics = train_stage_b(model, aux_train_loader, aux_eval_loader, args)
    final_metrics["stage_b_llama_regression"] = stage_b_metrics

    print("========== Final Eval: Attacker Preference Preference ==========")
    final_hh = evaluate_preference(model, attacker_preference_eval_loader, args.device, "Final Attacker Preference eval")
    print(final_hh)
    final_metrics["final_attacker_preference_dataseterence"] = final_hh

    if defender_evaluation_eval_loader is not None:
        print("========== Final Eval: Defender Evaluation Preference ==========")
        final_defender_evaluation = evaluate_preference(model, defender_evaluation_eval_loader, args.device, "Final Defender Evaluation eval")
        print(final_defender_evaluation)
        final_metrics["final_defender_evalerence"] = final_defender_evaluation

    print("========== Final Eval: LLaMA Scored Auxiliary Regression ==========")
    final_aux = evaluate_regression(model, aux_eval_loader, args.device, "Final aux regression eval")
    print(final_aux)
    final_metrics["final_llama_regression"] = final_aux

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = os.path.join(args.output_dir, "two_stage_extracted_rm_metrics.json")
    save_json(metrics_path, final_metrics)

    print(f"[OK] final LLaMA substitute reward model saved to: {args.output_dir}")
    print(f"[OK] metrics saved to: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--student_model_path", type=str, required=True)
    parser.add_argument("--attacker_preference_dataset_train_path", type=str, required=True)
    parser.add_argument("--attacker_preference_dataset_eval_path", type=str, required=True)
    parser.add_argument("--scored_aux_path", type=str, required=True)
    parser.add_argument("--defender_eval_eval_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_attacker_preference_train_samples", type=int, default=5000)
    parser.add_argument("--max_attacker_preference_eval_samples", type=int, default=1000)
    parser.add_argument("--max_aux_samples", type=int, default=5000)
    parser.add_argument("--max_defender_evaluation_eval_samples", type=int, default=1000)

    parser.add_argument("--aux_train_ratio", type=float, default=0.9)

    parser.add_argument("--pref_epochs", type=int, default=1)
    parser.add_argument("--distill_epochs", type=int, default=1)

    parser.add_argument("--pref_batch_size", type=int, default=1)
    parser.add_argument("--distill_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--pref_lr", type=float, default=2e-5)
    parser.add_argument("--distill_lr", type=float, default=1e-5)

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())