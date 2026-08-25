"""Query a chat-template reward model on prompt/response records.

This is intentionally separate from ``score_aux_with_target.py``: PublicRewardModel's
reward model was trained on its tokenizer's chat template, not on the legacy
``### Prompt`` format used by the locally trained RMs.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples is not None and len(rows) >= max_samples:
                    break
    return rows


class AuxDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int]):
        self.rows = read_jsonl(path, max_samples)
        for row in self.rows:
            if "prompt" not in row or "response" not in row:
                raise KeyError("Auxiliary JSONL records require prompt and response fields.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def format_conversation(tokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(response)},
    ]
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The target tokenizer has no chat_template; use the generic scorer instead.")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def make_collator(tokenizer, max_length):
    def collate(batch):
        texts = [format_conversation(tokenizer, row["prompt"], row["response"]) for row in batch]
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        encoded["raw_rows"] = batch
        return encoded
    return collate


@torch.inference_mode()
def main(args):
    dtype = torch.float32 if args.device == "cpu" else (torch.bfloat16 if args.bf16 else torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=1, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(args.device).eval()

    loader = DataLoader(AuxDataset(args.aux_path, args.max_samples), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=make_collator(tokenizer, args.max_length))
    rows = []
    for batch in tqdm(loader, desc="Querying PublicRewardModel reward model"):
        raw_rows = batch.pop("raw_rows")
        logits = model(**{key: value.to(args.device) for key, value in batch.items()}).logits.view(-1)
        for row, score in zip(raw_rows, logits.float().cpu().tolist()):
            scored = {"prompt": str(row["prompt"]), "response": str(row["response"]),
                      "target_score": float(score), "category": str(row.get("category", "")),
                      "target_model": args.model_name}
            # Preserve the pair provenance emitted by the disjoint-Attacker Preference
            # preparer so a run can be audited for cross-stage leakage.
            for key in ("source_pair_index",):
                if key in row:
                    scored[key] = row[key]
            rows.append(scored)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    scores = [row["target_score"] for row in rows]
    print(f"[OK] wrote {len(rows)} queries to {args.output_path}")
    if scores:
        print(f"[INFO] score min/mean/max: {min(scores):.6f} / {sum(scores)/len(scores):.6f} / {max(scores):.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Score auxiliary prompts with a chat-template reward model.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--aux_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", default="PublicRewardModel-Reward-Llama-3.1-8B")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
