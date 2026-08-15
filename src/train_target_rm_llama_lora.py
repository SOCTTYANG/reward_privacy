from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
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


def build_text(prompt: str, response: str) -> str:
    return f"### Prompt:\n{prompt}\n\n### Response:\n{response}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PreferenceDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        raw = read_jsonl(path, max_samples=max_samples)

        self.data = []
        for item in raw:
            if "prompt" not in item or "chosen" not in item or "rejected" not in item:
                raise KeyError(f"Need prompt/chosen/rejected, got keys: {list(item.keys())}")

            self.data.append(
                {
                    "prompt": str(item["prompt"]),
                    "chosen": str(item["chosen"]),
                    "rejected": str(item["rejected"]),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


@dataclass
class PreferenceCollator:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
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


def move_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()

    total = 0
    correct = 0
    gap_sum = 0.0
    loss_sum = 0.0

    for batch in tqdm(dataloader, desc="Evaluating"):
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


def train(args) -> None:
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] base model: {args.model_name_or_path}")
    print(f"[INFO] output dir: {args.output_dir}")
    print(f"[INFO] device: {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
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

    train_dataset = PreferenceDataset(args.train_path, args.max_train_samples)
    eval_dataset = PreferenceDataset(args.eval_path, args.max_eval_samples)

    print(f"[INFO] train samples = {len(train_dataset)}")
    print(f"[INFO] eval samples  = {len(eval_dataset)}")

    collator = PreferenceCollator(tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    total_update_steps = max(
        (len(train_loader) * args.epochs) // args.gradient_accumulation_steps,
        1,
    )
    warmup_steps = int(total_update_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    print("========== Before Training Eval ==========")
    before_metrics = evaluate(model, eval_loader, args.device)
    print(before_metrics)

    global_step = 0

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}")
        optimizer.zero_grad()

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

            '''gaps = chosen_scores - rejected_scores
            loss = -F.logsigmoid(gaps.float()).mean()'''
            gaps = chosen_scores - rejected_scores
            loss = -F.logsigmoid(gaps.float() - args.margin).mean()
            loss_for_backward = loss / args.gradient_accumulation_steps

            loss_for_backward.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            bs = chosen_scores.size(0)
            total_loss += loss.item() * bs
            total_count += bs

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "gap": f"{gaps.detach().float().mean().item():.4f}",
                }
            )

        avg_loss = total_loss / max(total_count, 1)
        print(f"[TRAIN] epoch={epoch + 1}, avg_loss={avg_loss:.6f}")

        metrics = evaluate(model, eval_loader, args.device)
        print(f"[EVAL] epoch={epoch + 1}, metrics={metrics}")

    print("========== Final Eval ==========")
    final_metrics = evaluate(model, eval_loader, args.device)
    print(final_metrics)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = os.path.join(args.output_dir, "target_rm_llama_lora_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] LLaMA LoRA target reward model saved to: {args.output_dir}")
    print(f"[OK] metrics saved to: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    '''新损失函数'''
    parser.add_argument("--margin", type=float, default=5.0)

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--max_eval_samples", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)

    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
    )

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())