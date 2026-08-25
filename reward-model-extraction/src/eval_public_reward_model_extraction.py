"""Evaluate ranking agreement between PublicRewardModel RM and an extracted student RM."""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_pairs(path: str, max_samples: int | None, seed: int) -> List[Dict[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if {"prompt", "chosen", "rejected"} <= row.keys():
                pairs.append({key: str(row[key]) for key in ("prompt", "chosen", "rejected")})
            elif {"x", "y_plus", "y_minus"} <= row.keys():
                # MIA member records use a different field convention. These
                # values are mapped only for external evaluation; neither the
                # y_plus/y_minus names nor membership metadata enter a model.
                pairs.append({
                    "prompt": str(row["x"]),
                    "chosen": str(row["y_plus"]),
                    "rejected": str(row["y_minus"]),
                })
    if max_samples is not None and len(pairs) > max_samples:
        pairs = random.Random(seed).sample(pairs, max_samples)
    if not pairs:
        raise ValueError("No prompt/chosen/rejected preference pairs found.")
    return pairs


def target_text(tokenizer, prompt: str, response: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )


def student_text(prompt: str, response: str) -> str:
    return f"### Prompt:\n{prompt}\n\n### Response:\n{response}"


@torch.inference_mode()
def score(tokenizer, model, texts: List[str], batch_size: int, max_length: int, device: str) -> List[float]:
    result = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start : start + batch_size], padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits.view(-1)
        result.extend(logits.float().cpu().tolist())
    return result


def load(path: str, dtype, device: str, use_fast: bool):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=use_fast)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=1, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model.to(device).eval()


def main(args):
    device = args.device
    target_dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if args.bf16 else torch.float16)
    # ``max_samples`` is applied per file.  Passing train.jsonl and test.jsonl
    # with 500 therefore reproduces the legacy 500+500 Diff protocol.
    split_pairs = {
        path: read_pairs(path, args.max_samples, args.seed + index)
        for index, path in enumerate(args.preference_path)
    }
    pairs = [pair for items in split_pairs.values() for pair in items]
    # PublicRewardModel is loaded with its documented slow tokenizer; the student uses
    # the default fast tokenizer, exactly as train_extracted_rm_two_stage.py.
    target_tok, target_model = load(args.target_model_path, target_dtype, device, use_fast=False)
    # Stage-A/B training keeps this RoBERTa model in FP32.  Its final score
    # gaps are often around 1e-4; loading it in BF16 can round chosen and
    # rejected scores to the same value and turn every strict comparison false.
    student_tok, student_model = load(args.student_model_path, torch.float32, device, use_fast=True)

    target_chosen = score(target_tok, target_model, [target_text(target_tok, x["prompt"], x["chosen"]) for x in tqdm(pairs, desc="Formatting target chosen")], args.batch_size, args.target_max_length, device)
    target_rejected = score(target_tok, target_model, [target_text(target_tok, x["prompt"], x["rejected"]) for x in pairs], args.batch_size, args.target_max_length, device)
    student_chosen = score(student_tok, student_model, [student_text(x["prompt"], x["chosen"]) for x in pairs], args.batch_size, args.student_max_length, device)
    student_rejected = score(student_tok, student_model, [student_text(x["prompt"], x["rejected"]) for x in pairs], args.batch_size, args.student_max_length, device)

    teacher_pref = [a > b for a, b in zip(target_chosen, target_rejected)]
    student_pref = [a > b for a, b in zip(student_chosen, student_rejected)]
    # Match the repository's established metric: both RMs rank the known y+
    # (chosen) above y- (rejected), rather than merely making the same choice.
    agreement = sum(a and b for a, b in zip(teacher_pref, student_pref)) / len(pairs)
    teacher_accuracy = sum(teacher_pref) / len(pairs)
    student_accuracy = sum(student_pref) / len(pairs)
    chosen_diffs = [abs(target - student) for target, student in zip(target_chosen, student_chosen)]
    rejected_diffs = [abs(target - student) for target, student in zip(target_rejected, student_rejected)]
    # Legacy tables report an item-level difference: for each preference
    # triple, average its chosen/rejected score differences before summarizing.
    item_diffs = [(chosen + rejected) / 2 for chosen, rejected in zip(chosen_diffs, rejected_diffs)]
    split_metrics = {}
    offset = 0
    for path, items in split_pairs.items():
        end = offset + len(items)
        split_teacher = teacher_pref[offset:end]
        split_student = student_pref[offset:end]
        split_diffs = item_diffs[offset:end]
        split_metrics[path] = {
            "count": len(items),
            "Diff_avg": statistics.fmean(split_diffs),
            "Diff_var": statistics.pvariance(split_diffs),
            "Agreement": sum(a and b for a, b in zip(split_teacher, split_student)) / len(items),
            "teacher_chosen_accuracy": sum(split_teacher) / len(items),
            "student_chosen_accuracy": sum(split_student) / len(items),
        }
        offset = end
    result = {"count": len(pairs), "Diff_avg": statistics.fmean(item_diffs),
              "Diff_var": statistics.pvariance(item_diffs), "Agreement": agreement,
              "teacher_chosen_accuracy": teacher_accuracy, "student_chosen_accuracy": student_accuracy,
              "splits": split_metrics}
    print(json.dumps(result, indent=2))
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"[OK] wrote {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model_path", required=True)
    parser.add_argument("--student_model_path", required=True)
    parser.add_argument("--preference_path", required=True, nargs="+")
    parser.add_argument("--output_path")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--target_max_length", type=int, default=2048)
    parser.add_argument("--student_max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
