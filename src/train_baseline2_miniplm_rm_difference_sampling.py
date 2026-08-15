#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import random
from pathlib import Path

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_prompt(row):
    for k in ["prompt", "instruction", "query", "question"]:
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def parse_pair(row):
    prompt = get_prompt(row)

    if "chosen" in row and "rejected" in row:
        return prompt, str(row["chosen"]), str(row["rejected"]), 0

    if "response_0" in row and "response_1" in row:
        r0 = str(row["response_0"])
        r1 = str(row["response_1"])

        gold_idx = None
        for key in ["better_response_id", "safer_response_id", "preferred_response_id", "label"]:
            if key in row:
                try:
                    v = int(row[key])
                    if v in [0, 1]:
                        gold_idx = v
                except Exception:
                    gold_idx = None
                break

        return prompt, r0, r1, gold_idx

    raise ValueError(f"Cannot parse pair fields from row keys: {list(row.keys())}")


def build_text(prompt, response):
    if prompt:
        return f"Prompt:\n{prompt}\n\nResponse:\n{response}"
    return response


def load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(
        path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_target_rm(base_model_path, adapter_path, dtype):
    tok = load_tokenizer(base_model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tok.pad_token_id

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return tok, model


def load_sequence_rm(model_path, dtype, device):
    tok = load_tokenizer(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.to(device)

    return tok, model


def encode_texts(tokenizer, texts, max_length, device):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def score_texts(model, tokenizer, texts, max_length, batch_size, device, desc="Scoring"):
    model.eval()
    scores = []

    total_batches = math.ceil(len(texts) / batch_size)

    pbar = tqdm(
        range(0, len(texts), batch_size),
        total=total_batches,
        desc=desc,
        dynamic_ncols=True,
    )

    for i in pbar:
        batch_texts = texts[i:i + batch_size]
        inputs = encode_texts(tokenizer, batch_texts, max_length=max_length, device=device)
        outputs = model(**inputs)
        batch_scores = outputs.logits.view(-1).detach().float().cpu().numpy().tolist()
        scores.extend(batch_scores)

        pbar.set_postfix({
            "done": f"{len(scores)}/{len(texts)}",
            "bs": batch_size,
        })

    return scores


@torch.no_grad()
def score_pairs(model, tokenizer, examples, max_length, batch_size, device, desc="Scoring pairs"):
    texts = []
    for ex in examples:
        texts.append(build_text(ex["prompt"], ex["response_0"]))
        texts.append(build_text(ex["prompt"], ex["response_1"]))

    flat_scores = score_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        desc=desc,
    )

    return np.array(flat_scores, dtype=np.float64).reshape(-1, 2)


def prepare_examples(rows):
    examples = []
    for row in rows:
        prompt, r0, r1, gold_idx = parse_pair(row)
        examples.append({
            "prompt": prompt,
            "response_0": r0,
            "response_1": r1,
            "gold_idx": gold_idx,
        })
    return examples


class PairDataset(Dataset):
    def __init__(self, examples, use_pseudo):
        self.examples = []

        for ex in examples:
            if use_pseudo:
                self.examples.append({
                    "prompt": ex["prompt"],
                    "chosen": ex["chosen"],
                    "rejected": ex["rejected"],
                })
            else:
                gold_idx = ex.get("gold_idx", None)
                if gold_idx not in [0, 1]:
                    continue

                if gold_idx == 0:
                    chosen = ex["response_0"]
                    rejected = ex["response_1"]
                else:
                    chosen = ex["response_1"]
                    rejected = ex["response_0"]

                self.examples.append({
                    "prompt": ex["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def pair_collate(batch):
    return {
        "prompts": [x["prompt"] for x in batch],
        "chosen": [x["chosen"] for x in batch],
        "rejected": [x["rejected"] for x in batch],
    }


def pairwise_loss(model, tokenizer, batch, max_length, device):
    texts = []

    for p, c, r in zip(batch["prompts"], batch["chosen"], batch["rejected"]):
        texts.append(build_text(p, c))
        texts.append(build_text(p, r))

    inputs = encode_texts(tokenizer, texts, max_length=max_length, device=device)
    outputs = model(**inputs)
    scores = outputs.logits.view(-1).float().view(len(batch["prompts"]), 2)

    chosen_scores = scores[:, 0]
    rejected_scores = scores[:, 1]

    loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
    return loss


def train_pairwise_rm(
    model,
    tokenizer,
    train_examples,
    output_dir,
    epochs,
    batch_size,
    grad_accum,
    lr,
    weight_decay,
    warmup_ratio,
    grad_clip,
    max_length,
    device,
    use_pseudo,
    name,
):
    dataset = PairDataset(train_examples, use_pseudo=use_pseudo)

    if len(dataset) == 0:
        raise RuntimeError(f"No valid training samples for {name}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pair_collate,
        drop_last=False,
    )

    print(f"[INFO] {name} train samples = {len(dataset)}")

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    update_steps_per_epoch = math.ceil(len(loader) / grad_accum)
    total_steps = update_steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    epoch_losses = []

    for epoch in range(1, epochs + 1):
        losses = []
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"{name} epoch {epoch}")
        for step, batch in enumerate(pbar, start=1):
            loss = pairwise_loss(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                max_length=max_length,
                device=device,
            )

            (loss / grad_accum).backward()
            losses.append(float(loss.item()))

            if step % grad_accum == 0 or step == len(loader):
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({
                "loss": f"{np.mean(losses[-20:]):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg_loss = float(np.mean(losses))
        epoch_losses.append(avg_loss)
        print(f"[{name}] epoch={epoch}, train_loss={avg_loss:.6f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[OK] {name} saved to: {output_dir}")

    return {
        "num_train_samples": len(dataset),
        "epoch_losses": epoch_losses,
    }


def build_difference_sampled_data(pool_examples, target_scores, ref_scores, sampling_ratio):
    scored = []

    for ex, ts, rs in zip(pool_examples, target_scores, ref_scores):
        target_pref = int(np.argmax(ts))
        other = 1 - target_pref

        target_margin = float(ts[target_pref] - ts[other])
        ref_margin = float(rs[target_pref] - rs[other])
        diff_score = target_margin - ref_margin

        if target_pref == 0:
            chosen = ex["response_0"]
            rejected = ex["response_1"]
        else:
            chosen = ex["response_1"]
            rejected = ex["response_0"]

        scored.append({
            "prompt": ex["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "target_pref": target_pref,
            "target_score_0": float(ts[0]),
            "target_score_1": float(ts[1]),
            "reference_score_0": float(rs[0]),
            "reference_score_1": float(rs[1]),
            "target_margin": target_margin,
            "reference_margin_along_target": ref_margin,
            "diff_score": float(diff_score),
            "original_gold_idx": ex.get("gold_idx", None),
        })

    scored = sorted(scored, key=lambda x: x["diff_score"], reverse=True)
    k = max(1, int(len(scored) * sampling_ratio))
    selected = scored[:k]

    return selected, scored


@torch.no_grad()
def simple_eval_pairwise(model, tokenizer, eval_examples, max_length, batch_size, device):
    valid = [ex for ex in eval_examples if ex.get("gold_idx", None) in [0, 1]]

    if len(valid) == 0:
        return {
            "num_items": 0,
            "gold_acc": None,
            "avg_gap": None,
        }

    scores = score_pairs(
        model=model,
        tokenizer=tokenizer,
        examples=valid,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        desc="Eval scoring",
    )

    pred = np.argmax(scores, axis=1)
    gold = np.array([ex["gold_idx"] for ex in valid], dtype=np.int64)
    gaps = np.abs(scores[:, 0] - scores[:, 1])

    return {
        "num_items": int(len(valid)),
        "gold_acc": float(np.mean(pred == gold)),
        "avg_gap": float(np.mean(gaps)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_base_model", required=True)
    parser.add_argument("--target_adapter_path", required=True)

    parser.add_argument("--reference_model_path", required=True)
    parser.add_argument("--student_model_path", required=True)

    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", required=True)

    parser.add_argument("--reference_output_dir", required=True)
    parser.add_argument("--selected_output_path", required=True)
    parser.add_argument("--all_scored_output_path", required=True)
    parser.add_argument("--student_output_dir", required=True)

    parser.add_argument("--max_reference_samples", type=int, default=1000)
    parser.add_argument("--max_score_samples", type=int, default=30000)
    parser.add_argument("--sampling_ratio", type=float, default=0.5)

    parser.add_argument("--reference_epochs", type=int, default=1)
    parser.add_argument("--student_epochs", type=int, default=1)

    parser.add_argument("--reference_batch_size", type=int, default=8)
    parser.add_argument("--student_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--reference_lr", type=float, default=2e-5)
    parser.add_argument("--student_lr", type=float, default=2e-5)

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--target_batch_size", type=int, default=1)
    parser.add_argument("--reference_score_batch_size", type=int, default=16)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else torch.float32

    print("======================================================")
    print("baseline2: MiniPLM-RM Difference Sampling")
    print("Target base model:    ", args.target_base_model)
    print("Target adapter:       ", args.target_adapter_path)
    print("Reference model path: ", args.reference_model_path)
    print("Student model path:   ", args.student_model_path)
    print("Train path:           ", args.train_path)
    print("Eval path:            ", args.eval_path)
    print("Sampling ratio alpha: ", args.sampling_ratio)
    print("Device:               ", device)
    print("dtype:                ", dtype)
    print("======================================================")

    rows = read_jsonl(args.train_path)
    examples = prepare_examples(rows)

    rng = random.Random(args.seed)
    rng.shuffle(examples)

    ref_n = min(args.max_reference_samples, len(examples))
    ref_examples = examples[:ref_n]
    pool_examples = examples[ref_n:]

    if args.max_score_samples and args.max_score_samples > 0:
        pool_examples = pool_examples[:args.max_score_samples]

    print("[INFO] total train examples:", len(examples))
    print("[INFO] reference examples:", len(ref_examples))
    print("[INFO] pool examples:", len(pool_examples))

    print("======================================================")
    print("Step 1: Train weak Reference RM")
    print("======================================================")

    ref_tokenizer, ref_model = load_sequence_rm(
        args.reference_model_path,
        dtype=dtype,
        device=device,
    )

    reference_metrics = train_pairwise_rm(
        model=ref_model,
        tokenizer=ref_tokenizer,
        train_examples=ref_examples,
        output_dir=args.reference_output_dir,
        epochs=args.reference_epochs,
        batch_size=args.reference_batch_size,
        grad_accum=args.gradient_accumulation_steps,
        lr=args.reference_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        max_length=args.max_length,
        device=device,
        use_pseudo=False,
        name="baseline2 Reference RM",
    )

    print("======================================================")
    print("Step 2: Score pool with Target RM and Reference RM")
    print("======================================================")

    target_tokenizer, target_model = load_target_rm(
        args.target_base_model,
        args.target_adapter_path,
        dtype=dtype,
    )

    print("[INFO] Scoring pool with Target RM...")
    target_scores = score_pairs(
        model=target_model,
        tokenizer=target_tokenizer,
        examples=pool_examples,
        max_length=args.max_length,
        batch_size=args.target_batch_size,
        device=device,
        desc="Target RM scoring",
    )

    print("[INFO] Scoring pool with Reference RM...")
    ref_scores = score_pairs(
        model=ref_model,
        tokenizer=ref_tokenizer,
        examples=pool_examples,
        max_length=args.max_length,
        batch_size=args.reference_score_batch_size,
        device=device,
        desc="Reference RM scoring",
    )

    print("======================================================")
    print("Step 3: Difference Sampling")
    print("======================================================")

    selected, all_scored = build_difference_sampled_data(
        pool_examples=pool_examples,
        target_scores=target_scores,
        ref_scores=ref_scores,
        sampling_ratio=args.sampling_ratio,
    )

    write_jsonl(selected, args.selected_output_path)
    write_jsonl(all_scored, args.all_scored_output_path)

    selected_diff = np.array([x["diff_score"] for x in selected], dtype=np.float64)
    all_diff = np.array([x["diff_score"] for x in all_scored], dtype=np.float64)

    sampling_metrics = {
        "num_pool_examples": int(len(pool_examples)),
        "num_selected_examples": int(len(selected)),
        "sampling_ratio": float(args.sampling_ratio),
        "selected_diff_avg": float(selected_diff.mean()),
        "selected_diff_var": float(selected_diff.var()),
        "all_diff_avg": float(all_diff.mean()),
        "all_diff_var": float(all_diff.var()),
        "selected_output_path": args.selected_output_path,
        "all_scored_output_path": args.all_scored_output_path,
    }

    print(json.dumps(sampling_metrics, ensure_ascii=False, indent=2))
    print("[OK] selected data saved to:", args.selected_output_path)
    print("[OK] all scored data saved to:", args.all_scored_output_path)

    del target_model
    del ref_model
    torch.cuda.empty_cache()

    print("======================================================")
    print("Step 4: Train baseline2 Substitute RM on selected D'")
    print("======================================================")

    student_tokenizer, student_model = load_sequence_rm(
        args.student_model_path,
        dtype=dtype,
        device=device,
    )

    student_metrics = train_pairwise_rm(
        model=student_model,
        tokenizer=student_tokenizer,
        train_examples=selected,
        output_dir=args.student_output_dir,
        epochs=args.student_epochs,
        batch_size=args.student_batch_size,
        grad_accum=args.gradient_accumulation_steps,
        lr=args.student_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        max_length=args.max_length,
        device=device,
        use_pseudo=True,
        name="baseline2 Substitute RM",
    )

    print("======================================================")
    print("Step 5: Simple eval on eval_path")
    print("======================================================")

    eval_examples = prepare_examples(read_jsonl(args.eval_path))

    eval_metrics = simple_eval_pairwise(
        model=student_model,
        tokenizer=student_tokenizer,
        eval_examples=eval_examples,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        device=device,
    )

    print(json.dumps(eval_metrics, ensure_ascii=False, indent=2))

    final_metrics = {
        "args": vars(args),
        "reference_metrics": reference_metrics,
        "sampling_metrics": sampling_metrics,
        "student_metrics": student_metrics,
        "eval_metrics": eval_metrics,
    }

    metrics_path = str(Path(args.student_output_dir) / "baseline2_miniplm_rm_metrics.json")
    save_json(final_metrics, metrics_path)

    print("======================================================")
    print("[OK] baseline2 finished")
    print("[OK] Substitute RM saved to:", args.student_output_dir)
    print("[OK] Metrics saved to:", metrics_path)
    print("======================================================")


if __name__ == "__main__":
    main()
