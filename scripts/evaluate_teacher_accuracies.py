"""Batch-evaluate teacher reward models on preference train/test splits.

Accuracy here is preference accuracy: ``score(chosen) > score(rejected)``.
For LoRA teachers, provide a local base-model mapping with ``--base-model``;
the base-model paths saved in adapter_config.json may be paths from the training
machine and therefore unavailable on the current machine.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_TEACHERS = [
    "target_rm_llama2_13b_full_margin5_e1",
    "target_rm_llama32_3b_full_margin5_e1",
    "target_rm_mistral_7b_full_margin5_e1",
    "target_rm_qwen3_8b_full_margin5_e1",
    "target_rm_roberta",
]


def build_text(prompt: str, response: str) -> str:
    return f"### Prompt:\n{prompt}\n\n### Response:\n{response}"


class PreferenceDataset(Dataset):
    def __init__(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as file:
            self.rows = [json.loads(line) for line in file if line.strip()]
        for row in self.rows:
            missing = {"prompt", "chosen", "rejected"} - row.keys()
            if missing:
                raise KeyError(f"{path} contains a row without {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        return {key: str(row[key]) for key in ("prompt", "chosen", "rejected")}


def make_collator(tokenizer: Any, max_length: int):
    def collate(rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        chosen = tokenizer(
            [build_text(row["prompt"], row["chosen"]) for row in rows],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        rejected = tokenizer(
            [build_text(row["prompt"], row["rejected"]) for row in rows],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }

    return collate


def parse_base_models(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("--base-model must have the form TEACHER_NAME=LOCAL_MODEL_PATH")
        result[name] = path
    return result


def load_teacher(model_dir: Path, base_models: dict[str, str], device: str):
    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_path = base_models.get(model_dir.name, config["base_model_name_or_path"])
        if not Path(base_path).exists():
            raise FileNotFoundError(
                f"Base model for {model_dir.name} is unavailable: {base_path}. "
                f"Pass --base-model {model_dir.name}=LOCAL_MODEL_PATH"
            )
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if device.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32
        base = AutoModelForSequenceClassification.from_pretrained(
            base_path, num_labels=1, trust_remote_code=True, torch_dtype=dtype
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=1, trust_remote_code=True
        )
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def evaluate(model: Any, tokenizer: Any, dataset_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    loader = DataLoader(
        PreferenceDataset(dataset_path), batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collator(tokenizer, args.max_length), num_workers=0,
    )
    correct = total = 0
    for batch in loader:
        batch = {name: tensor.to(args.device) for name, tensor in batch.items()}
        chosen = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits.squeeze(-1)
        rejected = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits.squeeze(-1)
        correct += (chosen > rejected).sum().item()
        total += chosen.numel()
    return {"correct": correct, "total": total, "preference_acc": correct / total if total else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--train-path", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--teachers", nargs="+", default=DEFAULT_TEACHERS)
    parser.add_argument("--base-model", action="append", default=[], metavar="TEACHER=PATH")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result-path", type=Path, default=Path("results/teacher_accuracy.json"))
    args = parser.parse_args()
    base_models = parse_base_models(args.base_model)
    rows = []
    for teacher in args.teachers:
        print(f"\n[INFO] Evaluating {teacher}")
        model, tokenizer = load_teacher(args.output_root / teacher, base_models, args.device)
        train = evaluate(model, tokenizer, args.train_path, args)
        test = evaluate(model, tokenizer, args.test_path, args)
        row = {"teacher": teacher, "train": train, "test": test}
        rows.append(row)
        print(f"  train: {train['correct']}/{train['total']} = {train['preference_acc']:.4%}")
        print(f"  test:  {test['correct']}/{test['total']} = {test['preference_acc']:.4%}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.result_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["teacher", "train_accuracy", "test_accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"teacher": row["teacher"], "train_accuracy": row["train"]["preference_acc"], "test_accuracy": row["test"]["preference_acc"]})
    print(f"\n[OK] Saved results to {args.result_path} and {csv_path}")


if __name__ == "__main__":
    main()
