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


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_text(prompt: str, response: str) -> str:
    return f"### Prompt:\n{prompt}\n\n### Response:\n{response}"


class AuxPairDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        raw = read_jsonl(path, max_samples=max_samples)

        self.data = []
        for item in raw:
            if "prompt" not in item or "response" not in item:
                raise KeyError(
                    f"Each aux item must contain prompt/response. Got keys: {list(item.keys())}"
                )

            self.data.append(
                {
                    "prompt": str(item["prompt"]),
                    "response": str(item["response"]),
                    "category": str(item.get("category", "")),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


class AuxPairCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        texts = [build_text(item["prompt"], item["response"]) for item in batch]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
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


@torch.no_grad()
def main(args) -> None:
    print(f"[INFO] base model path:    {args.base_model_path}")
    print(f"[INFO] lora adapter path: {args.lora_adapter_path}")
    print(f"[INFO] aux path:          {args.aux_path}")
    print(f"[INFO] output path:       {args.output_path}")
    print(f"[INFO] device:            {args.device}")

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

    model = PeftModel.from_pretrained(
        base_model,
        args.lora_adapter_path,
    )

    model.to(args.device)
    model.eval()

    dataset = AuxPairDataset(
        path=args.aux_path,
        max_samples=args.max_samples,
    )

    print(f"[INFO] aux samples = {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=AuxPairCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
        ),
        num_workers=0,
    )

    scored_rows = []

    for batch in tqdm(dataloader, desc="Scoring aux data with LLaMA2 LoRA RM"):
        batch = move_tensor_batch_to_device(batch, args.device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        scores = outputs.logits.squeeze(-1).detach().float().cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        for item, score in zip(batch["raw_items"], scores):
            scored_rows.append(
                {
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "target_score": float(score),
                    "category": item.get("category", ""),
                    "target_model": args.target_model_name,
                }
            )

    save_jsonl(scored_rows, args.output_path)

    print(f"[OK] saved scored aux data to: {args.output_path}")
    print(f"[OK] total scored samples: {len(scored_rows)}")

    if scored_rows:
        scores = [x["target_score"] for x in scored_rows]
        print("[INFO] score statistics:")
        print(f"  min  = {min(scores):.6f}")
        print(f"  max  = {max(scores):.6f}")
        print(f"  mean = {sum(scores) / len(scores):.6f}")

        print("[INFO] first scored example:")
        print(json.dumps(scored_rows[0], ensure_ascii=False, indent=2)[:1200])


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--lora_adapter_path", type=str, required=True)
    parser.add_argument("--aux_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--target_model_name",
        type=str,
        default="llama2_7b_lora_rm",
        help="Identifier recorded in the output JSONL metadata.",
    )

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
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
