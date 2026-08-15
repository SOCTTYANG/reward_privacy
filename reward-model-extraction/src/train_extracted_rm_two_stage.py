from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
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
    output = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        else:
            output[key] = value

    return output


class PreferenceDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        raw = read_jsonl(path, max_samples=max_samples)

        self.data = []
        for item in raw:
            if "prompt" not in item or "chosen" not in item or "rejected" not in item:
                raise KeyError(
                    f"Preference item must contain prompt/chosen/rejected. "
                    f"Got keys: {list(item.keys())}"
                )

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
                raise KeyError(
                    f"Scored auxiliary item must contain prompt/response/target_score. "
                    f"Got keys: {list(item.keys())}"
                )

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
        chosen_texts = [
            build_text(item["prompt"], item["chosen"])
            for item in batch
        ]

        rejected_texts = [
            build_text(item["prompt"], item["rejected"])
            for item in batch
        ]

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
        texts = [
            build_text(item["prompt"], item["response"])
            for item in batch
        ]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        target_scores = torch.tensor(
            [float(item["target_score"]) for item in batch],
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
        loss = -F.logsigmoid(gaps).mean()

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

        target_scores = batch["target_scores"]

        preds.extend(pred_scores.detach().float().cpu().tolist())
        targets.extend(target_scores.detach().float().cpu().tolist())

    n = len(preds)

    if n == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "pearson": 0.0,
        }

    mae = sum(abs(p - t) for p, t in zip(preds, targets)) / n
    rmse = math.sqrt(sum((p - t) ** 2 for p, t in zip(preds, targets)) / n)
    pearson = compute_pearson(preds, targets)

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
    }


def make_optimizer_and_scheduler(
    model,
    lr: float,
    weight_decay: float,
    total_steps: int,
    warmup_ratio: float,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
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


def train_stage_a_preference(
    model,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader],
    args,
) -> Dict[str, Any]:
    total_steps = len(train_loader) * args.pref_epochs

    optimizer, scheduler = make_optimizer_and_scheduler(
        model=model,
        lr=args.pref_lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    stage_metrics = {
        "before": None,
        "epochs": [],
        "after": None,
    }

    if eval_loader is not None:
        print("========== Stage A Before Eval: HH Preference ==========")
        before_metrics = evaluate_preference(
            model=model,
            dataloader=eval_loader,
            device=args.device,
            desc="Stage A before eval",
        )
        print(before_metrics)
        stage_metrics["before"] = before_metrics

    for epoch in range(args.pref_epochs):
        model.train()

        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Stage A Preference Training epoch {epoch + 1}")

        for batch in pbar:
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

            pbar.set_postfix(
                {
                    "rank_loss": f"{loss.item():.4f}",
                    "gap": f"{gaps.detach().mean().item():.4f}",
                }
            )

        avg_loss = total_loss / max(total_count, 1)

        epoch_record = {
            "epoch": epoch + 1,
            "train_ranking_loss": avg_loss,
        }

        print(f"[Stage A] epoch={epoch + 1}, train_ranking_loss={avg_loss:.6f}")

        if eval_loader is not None:
            eval_metrics = evaluate_preference(
                model=model,
                dataloader=eval_loader,
                device=args.device,
                desc=f"Stage A eval epoch {epoch + 1}",
            )
            print(f"[Stage A Eval] epoch={epoch + 1}, metrics={eval_metrics}")
            epoch_record["eval"] = eval_metrics

        stage_metrics["epochs"].append(epoch_record)

    if eval_loader is not None:
        print("========== Stage A After Eval: HH Preference ==========")
        after_metrics = evaluate_preference(
            model=model,
            dataloader=eval_loader,
            device=args.device,
            desc="Stage A after eval",
        )
        print(after_metrics)
        stage_metrics["after"] = after_metrics

    return stage_metrics


def train_stage_b_regression(
    model,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    args,
) -> Dict[str, Any]:
    total_steps = len(train_loader) * args.distill_epochs

    optimizer, scheduler = make_optimizer_and_scheduler(
        model=model,
        lr=args.distill_lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
    )

    stage_metrics = {
        "before": None,
        "epochs": [],
        "after": None,
    }

    print("========== Stage B Before Eval: Dolly Scored Auxiliary ==========")
    before_metrics = evaluate_regression(
        model=model,
        dataloader=eval_loader,
        device=args.device,
        desc="Stage B before eval",
    )
    print(before_metrics)
    stage_metrics["before"] = before_metrics

    for epoch in range(args.distill_epochs):
        model.train()

        total_loss = 0.0
        total_count = 0

        pbar = tqdm(train_loader, desc=f"Stage B Distillation epoch {epoch + 1}")

        for batch in pbar:
            batch = move_to_device(batch, args.device)

            pred_scores = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits.squeeze(-1)

            target_scores = batch["target_scores"]

            loss = F.mse_loss(pred_scores, target_scores)

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            bs = pred_scores.size(0)
            total_loss += loss.item() * bs
            total_count += bs

            pbar.set_postfix(
                {
                    "mse": f"{loss.item():.4f}",
                    "pred": f"{pred_scores.detach().mean().item():.4f}",
                    "target": f"{target_scores.detach().mean().item():.4f}",
                }
            )

        avg_loss = total_loss / max(total_count, 1)

        epoch_record = {
            "epoch": epoch + 1,
            "train_mse_loss": avg_loss,
        }

        print(f"[Stage B] epoch={epoch + 1}, train_mse_loss={avg_loss:.6f}")

        eval_metrics = evaluate_regression(
            model=model,
            dataloader=eval_loader,
            device=args.device,
            desc=f"Stage B eval epoch {epoch + 1}",
        )
        print(f"[Stage B Eval] epoch={epoch + 1}, metrics={eval_metrics}")

        epoch_record["eval"] = eval_metrics
        stage_metrics["epochs"].append(epoch_record)

    print("========== Stage B After Eval: Dolly Scored Auxiliary ==========")
    after_metrics = evaluate_regression(
        model=model,
        dataloader=eval_loader,
        device=args.device,
        desc="Stage B after eval",
    )
    print(after_metrics)
    stage_metrics["after"] = after_metrics

    return stage_metrics


def train(args) -> None:
    set_seed(args.seed)

    if args.skip_stage_a and args.skip_stage_b:
        raise ValueError(
            "Cannot skip both Stage A and Stage B; at least one training stage must run."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Two-stage extracted reward model training")
    print(f"[INFO] student init model:      {args.student_model_path}")
    print(f"[INFO] HH pref train path:     {args.hh_pref_train_path}")
    print(f"[INFO] HH pref eval path:      {args.hh_pref_eval_path}")
    print(f"[INFO] scored aux path:        {args.scored_aux_path}")
    print(f"[INFO] PKU pref eval path:     {args.pku_pref_eval_path}")
    print(f"[INFO] output dir:             {args.output_dir}")
    print(f"[INFO] device:                 {args.device}")
    print(f"[INFO] skip Stage A:           {args.skip_stage_a}")
    print(f"[INFO] skip Stage B:           {args.skip_stage_b}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model_path,
        trust_remote_code=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.student_model_path,
        num_labels=1,
        trust_remote_code=True,
    )

    model.to(args.device)

    # ========== Stage A data: HH preference ==========
    hh_train_dataset = PreferenceDataset(
        path=args.hh_pref_train_path,
        max_samples=args.max_hh_train_samples,
    )

    hh_eval_dataset = None
    if args.hh_pref_eval_path is not None:
        hh_eval_dataset = PreferenceDataset(
            path=args.hh_pref_eval_path,
            max_samples=args.max_hh_eval_samples,
        )

    print(f"[INFO] HH train samples = {len(hh_train_dataset)}")
    if hh_eval_dataset is not None:
        print(f"[INFO] HH eval samples  = {len(hh_eval_dataset)}")

    pref_collator = PreferenceCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    hh_train_loader = DataLoader(
        hh_train_dataset,
        batch_size=args.pref_batch_size,
        shuffle=True,
        collate_fn=pref_collator,
        num_workers=0,
    )

    hh_eval_loader = None
    if hh_eval_dataset is not None:
        hh_eval_loader = DataLoader(
            hh_eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=pref_collator,
            num_workers=0,
        )

    # ========== Stage B data: scored auxiliary ==========
    scored_aux_dataset = ScoredAuxDataset(
        path=args.scored_aux_path,
        max_samples=args.max_aux_samples,
    )

    aux_train_size = int(len(scored_aux_dataset) * args.aux_train_ratio)
    aux_eval_size = len(scored_aux_dataset) - aux_train_size

    aux_train_dataset, aux_eval_dataset = random_split(
        scored_aux_dataset,
        [aux_train_size, aux_eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"[INFO] scored aux total samples = {len(scored_aux_dataset)}")
    print(f"[INFO] scored aux train samples = {len(aux_train_dataset)}")
    print(f"[INFO] scored aux eval samples  = {len(aux_eval_dataset)}")

    aux_collator = ScoredAuxCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    aux_train_loader = DataLoader(
        aux_train_dataset,
        batch_size=args.distill_batch_size,
        shuffle=True,
        collate_fn=aux_collator,
        num_workers=0,
    )

    aux_eval_loader = DataLoader(
        aux_eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=aux_collator,
        num_workers=0,
    )

    # ========== Final PKU eval data ==========
    pku_eval_loader = None
    if args.pku_pref_eval_path is not None:
        pku_eval_dataset = PreferenceDataset(
            path=args.pku_pref_eval_path,
            max_samples=args.max_pku_eval_samples,
        )

        print(f"[INFO] PKU eval samples = {len(pku_eval_dataset)}")

        pku_eval_loader = DataLoader(
            pku_eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=pref_collator,
            num_workers=0,
        )

    final_metrics: Dict[str, Any] = {}

    # ========== Optional initial eval ==========
    if pku_eval_loader is not None:
        print("========== Initial Eval: PKU Preference ==========")
        initial_pku_metrics = evaluate_preference(
            model=model,
            dataloader=pku_eval_loader,
            device=args.device,
            desc="Initial PKU preference eval",
        )
        print(initial_pku_metrics)
        final_metrics["initial_pku_preference"] = initial_pku_metrics

    # ========== Stage A ==========
    if (not args.skip_stage_a) and args.pref_epochs > 0:
        print("\n================ Stage A: HH Preference Pretraining ================")
        stage_a_metrics = train_stage_a_preference(
            model=model,
            train_loader=hh_train_loader,
            eval_loader=hh_eval_loader,
            args=args,
        )
        final_metrics["stage_a_hh_preference"] = stage_a_metrics

        stage_a_output_dir = os.path.join(args.output_dir, "after_stage_a_hh_preference")
        os.makedirs(stage_a_output_dir, exist_ok=True)
        model.save_pretrained(stage_a_output_dir)
        tokenizer.save_pretrained(stage_a_output_dir)
        print(f"[OK] Stage A checkpoint saved to: {stage_a_output_dir}")

        if pku_eval_loader is not None:
            print("========== After Stage A Eval: PKU Preference ==========")
            pku_after_a = evaluate_preference(
                model=model,
                dataloader=pku_eval_loader,
                device=args.device,
                desc="After Stage A PKU preference eval",
            )
            print(pku_after_a)
            final_metrics["after_stage_a_pku_preference"] = pku_after_a

    # ========== Stage B ==========
    if (not args.skip_stage_b) and args.distill_epochs > 0:
        print("\n================ Stage B: Dolly Scored Auxiliary Distillation ================")
        stage_b_metrics = train_stage_b_regression(
            model=model,
            train_loader=aux_train_loader,
            eval_loader=aux_eval_loader,
            args=args,
        )
        final_metrics["stage_b_dolly_regression"] = stage_b_metrics

    # ========== Final eval ==========
    if hh_eval_loader is not None:
        print("========== Final Eval: HH Preference ==========")
        final_hh_metrics = evaluate_preference(
            model=model,
            dataloader=hh_eval_loader,
            device=args.device,
            desc="Final HH preference eval",
        )
        print(final_hh_metrics)
        final_metrics["final_hh_preference"] = final_hh_metrics

    if pku_eval_loader is not None:
        print("========== Final Eval: PKU Preference ==========")
        final_pku_metrics = evaluate_preference(
            model=model,
            dataloader=pku_eval_loader,
            device=args.device,
            desc="Final PKU preference eval",
        )
        print(final_pku_metrics)
        final_metrics["final_pku_preference"] = final_pku_metrics

    print("========== Final Eval: Dolly Scored Auxiliary Regression ==========")
    final_aux_metrics = evaluate_regression(
        model=model,
        dataloader=aux_eval_loader,
        device=args.device,
        desc="Final auxiliary regression eval",
    )
    print(final_aux_metrics)
    final_metrics["final_dolly_regression"] = final_aux_metrics

    # ========== Save ==========
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = os.path.join(args.output_dir, "two_stage_extracted_rm_metrics.json")
    save_json(metrics_path, final_metrics)

    print(f"[OK] final extracted reward model saved to: {args.output_dir}")
    print(f"[OK] metrics saved to: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--student_model_path", type=str, required=True)

    parser.add_argument("--hh_pref_train_path", type=str, required=True)
    parser.add_argument("--hh_pref_eval_path", type=str, default=None)

    parser.add_argument("--scored_aux_path", type=str, required=True)

    parser.add_argument("--pku_pref_eval_path", type=str, default=None)

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_hh_train_samples", type=int, default=5000)
    parser.add_argument("--max_hh_eval_samples", type=int, default=1000)
    parser.add_argument("--max_aux_samples", type=int, default=5000)
    parser.add_argument("--max_pku_eval_samples", type=int, default=1000)

    parser.add_argument("--aux_train_ratio", type=float, default=0.9)

    parser.add_argument("--pref_epochs", type=int, default=1)
    parser.add_argument("--distill_epochs", type=int, default=1)

    parser.add_argument("--pref_batch_size", type=int, default=8)
    parser.add_argument("--distill_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--pref_lr", type=float, default=2e-5)
    parser.add_argument("--distill_lr", type=float, default=1e-5)

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--skip_stage_a",
        action="store_true",
        help="Skip Stage A HH preference pretraining and run Stage B only.",
    )

    parser.add_argument(
        "--skip_stage_b",
        action="store_true",
        help="Skip Stage B scored auxiliary distillation and run Stage A only.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())