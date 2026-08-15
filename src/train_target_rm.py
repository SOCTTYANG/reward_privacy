from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_text(prompt: str, response: str) -> str:
    return f"### Prompt:\n{prompt}\n\n### Response:\n{response}"


class PreferenceDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        rows = read_jsonl(path)
        if max_samples is not None:
            rows = rows[:max_samples]

        self.data = []
        for item in rows:
            if not all(k in item for k in ["prompt", "chosen", "rejected"]):
                raise KeyError(f"Example must contain prompt/chosen/rejected, got keys: {list(item.keys())}")

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

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, dataloader, device: str) -> Dict[str, float]:
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
        loss = -F.logsigmoid(gaps).mean()

        correct += (gaps > 0).long().sum().item()
        total += gaps.numel()
        gap_sum += gaps.detach().float().sum().item()
        loss_sum += loss.item() * gaps.numel()

    return {
        "preference_acc": correct / max(total, 1),
        "avg_gap": gap_sum / max(total, 1),
        "eval_loss": loss_sum / max(total, 1),
    }


def train(args) -> None:
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    print(f"[INFO] device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        trust_remote_code=True,
    )

    model.to(device)

    train_dataset = PreferenceDataset(args.train_path, args.max_train_samples)
    eval_dataset = PreferenceDataset(args.eval_path, args.max_eval_samples)

    print(f"[INFO] train samples = {len(train_dataset)}")
    print(f"[INFO] eval samples  = {len(eval_dataset)}")

    collator = PreferenceCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

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

    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("========== Before Training Eval ==========")
    before_metrics = evaluate(model, eval_loader, device)
    print(before_metrics)

    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}")

        for batch in pbar:
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
            loss = -F.logsigmoid(gaps).mean()

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            bs = chosen_scores.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "gap": f"{gaps.detach().mean().item():.4f}",
                }
            )

        avg_loss = total_loss / max(total_count, 1)
        print(f"[TRAIN] epoch={epoch + 1}, avg_loss={avg_loss:.6f}")

        metrics = evaluate(model, eval_loader, device)
        print(f"[EVAL] epoch={epoch + 1}, metrics={metrics}")

    print("========== Final Eval ==========")
    final_metrics = evaluate(model, eval_loader, device)
    print(final_metrics)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = os.path.join(args.output_dir, "target_rm_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] target reward model saved to: {args.output_dir}")
    print(f"[OK] metrics saved to: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--max_eval_samples", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())