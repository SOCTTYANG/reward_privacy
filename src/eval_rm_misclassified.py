from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_text(prompt: str, response: str) -> str:
    return f"### Prompt:\n{prompt}\n\n### Response:\n{response}"


class PairwisePreferenceDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        raw = read_jsonl(path, max_samples=max_samples)

        self.data = []

        for idx, item in enumerate(raw):
            if "prompt" not in item or "chosen" not in item or "rejected" not in item:
                raise KeyError(
                    f"Each item must contain prompt/chosen/rejected. Got keys: {list(item.keys())}"
                )

            self.data.append(
                {
                    "idx": idx,
                    "prompt": str(item["prompt"]),
                    "chosen": str(item["chosen"]),
                    "rejected": str(item["rejected"]),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class PairwiseCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosen_texts = [build_text(x["prompt"], x["chosen"]) for x in batch]
        rejected_texts = [build_text(x["prompt"], x["rejected"]) for x in batch]

        chosen_enc = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        rejected_enc = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"],
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
            "raw_items": batch,
        }


def move_tensor_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    output = {}

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device)
        else:
            output[key] = value

    return output


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=1,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        device_map=None,
    )

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False

    if args.adapter_path:
        print(f"[INFO] Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print("[INFO] No adapter_path provided. Using base/full reward model directly.")
        model = base_model

    model.to(args.device)
    model.eval()

    return tokenizer, model


@torch.no_grad()
def main(args) -> None:
    print(f"[INFO] base_model_path = {args.base_model_path}")
    print(f"[INFO] adapter_path    = {args.adapter_path}")
    print(f"[INFO] eval_path       = {args.eval_path}")
    print(f"[INFO] output_dir      = {args.output_dir}")
    print(f"[INFO] max_length      = {args.max_length}")
    print(f"[INFO] device          = {args.device}")

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, model = load_model(args)

    dataset = PairwisePreferenceDataset(
        path=args.eval_path,
        max_samples=args.max_samples,
    )

    print(f"[INFO] eval samples = {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PairwiseCollator(tokenizer=tokenizer, max_length=args.max_length),
        num_workers=0,
    )

    all_rows = []
    wrong_rows = []

    total = 0
    correct = 0
    gap_sum = 0.0

    for batch in tqdm(dataloader, desc="Evaluating RM and collecting mistakes"):
        batch = move_tensor_batch_to_device(batch, args.device)

        chosen_scores = model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        ).logits.squeeze(-1)

        rejected_scores = model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        ).logits.squeeze(-1)

        gaps = chosen_scores - rejected_scores

        chosen_scores = chosen_scores.detach().float().cpu().tolist()
        rejected_scores = rejected_scores.detach().float().cpu().tolist()
        gaps = gaps.detach().float().cpu().tolist()

        if isinstance(chosen_scores, float):
            chosen_scores = [chosen_scores]
        if isinstance(rejected_scores, float):
            rejected_scores = [rejected_scores]
        if isinstance(gaps, float):
            gaps = [gaps]

        for item, sc, sr, gap in zip(batch["raw_items"], chosen_scores, rejected_scores, gaps):
            is_correct = gap > 0

            row = {
                "idx": item["idx"],
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
                "chosen_score": float(sc),
                "rejected_score": float(sr),
                "gap": float(gap),
                "correct": bool(is_correct),
            }

            all_rows.append(row)

            if not is_correct:
                wrong_rows.append(row)

            total += 1
            correct += int(is_correct)
            gap_sum += float(gap)

    acc = correct / max(total, 1)
    wrong_count = total - correct
    avg_gap = gap_sum / max(total, 1)

    summary = {
        "total": total,
        "correct": correct,
        "wrong": wrong_count,
        "accuracy": acc,
        "wrong_rate": wrong_count / max(total, 1),
        "avg_gap": avg_gap,
        "base_model_path": args.base_model_path,
        "adapter_path": args.adapter_path,
        "eval_path": args.eval_path,
        "max_length": args.max_length,
    }

    all_path = os.path.join(args.output_dir, "all_eval_samples_with_scores.jsonl")
    wrong_path = os.path.join(args.output_dir, "misclassified_samples.jsonl")
    summary_path = os.path.join(args.output_dir, "misclassified_summary.json")

    write_jsonl(all_rows, all_path)
    write_jsonl(wrong_rows, wrong_path)
    write_json(summary, summary_path)

    print("========== Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] all samples saved to: {all_path}")
    print(f"[OK] wrong samples saved to: {wrong_path}")
    print(f"[OK] summary saved to: {summary_path}")

    if wrong_rows:
        print("========== First Wrong Example ==========")
        print(json.dumps(wrong_rows[0], ensure_ascii=False, indent=2)[:2000])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())