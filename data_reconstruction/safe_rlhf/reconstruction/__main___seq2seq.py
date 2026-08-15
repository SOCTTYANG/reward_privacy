from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, fields

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments

from safe_rlhf.reconstruction.dataset_seq2seq import (
    ReconstructionCollator,
    ReconstructionTripletDataset,
)
from safe_rlhf.reconstruction.model_seq2seq import ReconstructionModel
from safe_rlhf.reconstruction.trainer_seq2seq import ReconstructionTrainer


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/home/vipuser/Desktop/model/llama2-7b")
    extractor_name_or_path: str = field(default="/home/vipuser/Desktop/model/bge-small-en-v1.5")
    torch_dtype: str = field(default="auto")
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)


@dataclass
class DataArguments:
    dataset_name: str = field(default="PKU-Alignment/PKU-SafeRLHF-10K")
    train_file: str = field(default="")
    eval_file: str = field(default="")
    processed_dir: str = field(default="/home/vipuser/Desktop/yang-safe-rlhf/data/pku_saferlhf_10k_triplet")
    max_length: int = field(default=320)
    eval_ratio: float = field(default=0.05)
    split_seed: int = field(default=42)
    force_rebuild: bool = field(default=False)


@dataclass
class CustomArguments:
    lambda_mle: float = field(default=1.0)
    lambda_cos: float = field(default=0.05)
    cosine_max_new_tokens: int = field(default=96)
    cosine_temperature: float = field(default=1.0)
    cosine_top_p: float = field(default=0.95)
    reward_baseline_momentum: float = field(default=0.9)


def normalize_training_argument_aliases() -> None:
    training_arg_names = {item.name for item in fields(TrainingArguments)}
    if "eval_strategy" not in training_arg_names and "evaluation_strategy" in training_arg_names:
        sys.argv = [
            "--evaluation_strategy" if arg == "--eval_strategy" else arg
            for arg in sys.argv
        ]
    elif "evaluation_strategy" not in training_arg_names and "eval_strategy" in training_arg_names:
        sys.argv = [
            "--eval_strategy" if arg == "--evaluation_strategy" else arg
            for arg in sys.argv
        ]

    if "overwrite_output_dir" not in training_arg_names:
        filtered = [sys.argv[0]]
        index = 1
        while index < len(sys.argv):
            arg = sys.argv[index]
            if arg == "--overwrite_output_dir":
                index += 2 if index + 1 < len(sys.argv) and not sys.argv[index + 1].startswith("--") else 1
                continue
            filtered.append(arg)
            index += 1
        sys.argv = filtered


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch_dtype: {dtype_name}")


def convert_example_to_triplet(example: dict) -> dict | None:

    x = str(example.get("prompt", "")).strip()
    r0 = str(example.get("response_0", "")).strip()
    r1 = str(example.get("response_1", "")).strip()
    better_id = example.get("better_response_id", None)

    if not x or not r0 or not r1 or better_id is None:
        return None

    better_id = int(better_id)
    if better_id == 0:
        y_plus, y_minus = r0, r1
    elif better_id == 1:
        y_plus, y_minus = r1, r0
    else:
        return None

    return {
        "x": x,
        "y_plus": y_plus,
        "y_minus": y_minus,
    }


def build_local_triplet_files(
    dataset_name: str,
    processed_dir: str,
    eval_ratio: float,
    split_seed: int,
    force_rebuild: bool,
) -> tuple[str, str]:
    os.makedirs(processed_dir, exist_ok=True)
    train_file = os.path.join(processed_dir, "train.jsonl")
    eval_file = os.path.join(processed_dir, "eval.jsonl")

    if (
        not force_rebuild
        and os.path.exists(train_file)
        and os.path.exists(eval_file)
    ):
        print("[INFO] Reusing existing processed triplet files.")
        return train_file, eval_file

    print(f"[INFO] Downloading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    print(f"[INFO] Raw dataset size = {len(ds)}")
    ds = ds.train_test_split(test_size=eval_ratio, seed=split_seed)

    train_ds = ds["train"]
    eval_ds = ds["test"]

    train_count = 0
    eval_count = 0

    with open(train_file, "w", encoding="utf-8") as f:
        for ex in train_ds:
            row = convert_example_to_triplet(ex)
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            train_count += 1

    with open(eval_file, "w", encoding="utf-8") as f:
        for ex in eval_ds:
            row = convert_example_to_triplet(ex)
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            eval_count += 1

    print(f"[INFO] Saved processed train triplets: {train_count}")
    print(f"[INFO] Saved processed eval triplets : {eval_count}")
    print(f"[INFO] train_file = {train_file}")
    print(f"[INFO] eval_file  = {eval_file}")

    return train_file, eval_file


def main():
    normalize_training_argument_aliases()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, CustomArguments))
    model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()

    print("=" * 80)
    print("[INFO] Parsed arguments successfully.")
    print(f"[INFO] model_name_or_path      = {model_args.model_name_or_path}")
    print(f"[INFO] extractor_name_or_path = {model_args.extractor_name_or_path}")
    print(f"[INFO] torch_dtype            = {model_args.torch_dtype}")
    print(f"[INFO] dataset_name           = {data_args.dataset_name}")
    print(f"[INFO] train_file             = {data_args.train_file}")
    print(f"[INFO] eval_file              = {data_args.eval_file}")
    print(f"[INFO] processed_dir          = {data_args.processed_dir}")
    print(f"[INFO] output_dir             = {training_args.output_dir}")
    print(f"[INFO] lambda_mle             = {custom_args.lambda_mle}")
    print(f"[INFO] lambda_cos             = {custom_args.lambda_cos}")
    print(f"[INFO] cosine_max_new_tokens  = {custom_args.cosine_max_new_tokens}")
    print("=" * 80)

    has_train_file = bool(data_args.train_file.strip())
    has_eval_file = bool(data_args.eval_file.strip())
    if has_train_file or has_eval_file:
        if not (has_train_file and has_eval_file):
            raise ValueError("Please pass both --train_file and --eval_file, or neither.")
        train_file = data_args.train_file
        eval_file = data_args.eval_file
        if not os.path.isfile(train_file):
            raise FileNotFoundError(f"--train_file does not exist: {train_file}")
        if not os.path.isfile(eval_file):
            raise FileNotFoundError(f"--eval_file does not exist: {eval_file}")
        print("[INFO] Using explicit local triplet files.")
    else:
        train_file, eval_file = build_local_triplet_files(
            dataset_name=data_args.dataset_name,
            processed_dir=data_args.processed_dir,
            eval_ratio=data_args.eval_ratio,
            split_seed=data_args.split_seed,
            force_rebuild=data_args.force_rebuild,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Tokenizer loaded.")

    train_dataset = ReconstructionTripletDataset(
        train_file,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        min_target_tokens=4,
        target_reserved_tokens=128,
        max_prompt_tokens=56,
        max_preferred_tokens=24,
    )
    eval_dataset = ReconstructionTripletDataset(
        eval_file,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        min_target_tokens=4,
        target_reserved_tokens=128,
        max_prompt_tokens=56,
        max_preferred_tokens=24,
    )

    print(f"[INFO] train_dataset size = {len(train_dataset)}")
    print(f"[INFO] eval_dataset size  = {len(eval_dataset)}")

    first_sample = train_dataset[0]
    print("[INFO] First sample source preview:")
    print(first_sample["source_text"][:500])
    print("-" * 80)
    print("[INFO] First sample target preview:")
    print(first_sample["target_text"][:300])
    print("=" * 80)

    model = ReconstructionModel(
        model_name_or_path=model_args.model_name_or_path,
        extractor_name_or_path=model_args.extractor_name_or_path,
        torch_dtype=resolve_torch_dtype(model_args.torch_dtype),
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )

    print("[INFO] ReconstructionModel loaded.")

    data_collator = ReconstructionCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_length,
    )

    trainer = ReconstructionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        lambda_mle=custom_args.lambda_mle,
        lambda_cos=custom_args.lambda_cos,
        generation_tokenizer=tokenizer,
        cosine_max_new_tokens=custom_args.cosine_max_new_tokens,
        cosine_temperature=custom_args.cosine_temperature,
        cosine_top_p=custom_args.cosine_top_p,
        reward_baseline_momentum=custom_args.reward_baseline_momentum,
    )

    print("[INFO] Starting trainer.train() ...")
    train_result = trainer.train()
    print("[INFO] trainer.train() finished.")
    print(train_result)

    os.makedirs(training_args.output_dir, exist_ok=True)
    model.save_reconstruction_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"[INFO] Saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
