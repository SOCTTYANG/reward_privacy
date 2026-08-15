from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCTION_PREFIX = (
    "You are given a user prompt and a preferred response. "
    "Generate an alternative response that is less preferred than the preferred response. "
    "The response should still be a plausible answer to the prompt, "
    "but it should be less helpful, less direct, less complete, or less useful.\n\n"
)
PROMPT_PREFIX = "User prompt:\n"
PREFERRED_PREFIX = "\n\nPreferred response:\n"
TARGET_PREFIX = "\n\nLess-preferred alternative response:\n"


def build_source_text(x: str, y_plus: str) -> str:
    return (
        INSTRUCTION_PREFIX
        + PROMPT_PREFIX + x
        + PREFERRED_PREFIX + y_plus
        + TARGET_PREFIX
    )


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict], path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def extract_candidate(decoded_text: str) -> str:
    if TARGET_PREFIX in decoded_text:
        return decoded_text.split(TARGET_PREFIX, 1)[1].strip()
    return decoded_text.strip()


def resolve_torch_dtype(torch_dtype: str, device: str):
    if torch_dtype == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    if torch_dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def load_generator(base_model_path: str, adapter_dir: str, device: str, torch_dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_torch_dtype(torch_dtype, device)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    model.to(device)

    return tokenizer, model


@torch.no_grad()
def generate_candidate_yminus(
    model,
    tokenizer,
    x: str,
    y_plus: str,
    device: str,
    num_candidates: int = 3,
    max_new_tokens: int = 96,
    temperature: float = 1.0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
) -> List[str]:
    source_text = build_source_text(x, y_plus)
    inputs = tokenizer(source_text, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_return_sequences=max(num_candidates * 2, num_candidates),
        pad_token_id=tokenizer.eos_token_id,
    )

    candidates = []
    seen = set()

    for out in outputs:
        text = tokenizer.decode(out, skip_special_tokens=True)
        cand = extract_candidate(text)
        cand_norm = normalize_text(cand)

        if not cand_norm or cand_norm in seen:
            continue

        seen.add(cand_norm)
        candidates.append(cand)

        if len(candidates) >= num_candidates:
            break

    retry = 0
    while len(candidates) < num_candidates and retry < 10:
        retry += 1
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature + 0.1 * retry,
            top_p=min(0.99, top_p + 0.01 * retry),
            repetition_penalty=repetition_penalty,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cand = extract_candidate(text)
        cand_norm = normalize_text(cand)

        if not cand_norm or cand_norm in seen:
            continue

        seen.add(cand_norm)
        candidates.append(cand)

    return candidates[:num_candidates]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    args = parser.parse_args()

    print("=" * 80)
    print("[INFO] Loading generator...")
    tokenizer, model = load_generator(
        base_model_path=args.base_model_path,
        adapter_dir=args.adapter_dir,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    print("[INFO] Model loaded.")
    print("=" * 80)

    data = load_jsonl(args.input_file)
    if args.max_samples > 0:
        data = data[:args.max_samples]

    print(f"[INFO] Loaded {len(data)} samples from: {args.input_file}")

    output_rows = []

    for idx, item in enumerate(data):
        x = str(item.get("x", "")).strip()
        y_plus = str(item.get("y_plus", "")).strip()

        if not x or not y_plus:
            continue

        candidate_y_minus_list = generate_candidate_yminus(
            model=model,
            tokenizer=tokenizer,
            x=x,
            y_plus=y_plus,
            device=args.device,
            num_candidates=args.num_candidates,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        out_row = {
            "x": x,
            "candidate_y_minus_list": candidate_y_minus_list,
        }

        output_rows.append(out_row)

        if idx < 3:
            print(f"\n[Preview sample {idx}]")
            print("-" * 80)
            print("x:")
            print(x[:300])
            print("-" * 80)
            print("candidate_y_minus_list:")
            for j, cand in enumerate(candidate_y_minus_list, start=1):
                print(f"[y_minus_{j}] {cand[:220]}")
            print("-" * 80)

        if (idx + 1) % 50 == 0:
            print(f"[INFO] Processed {idx + 1}/{len(data)} samples")

    save_jsonl(output_rows, args.output_file)

    print("=" * 80)
    print(f"[INFO] Saved {len(output_rows)} rows to:")
    print(args.output_file)
    print("=" * 80)


if __name__ == "__main__":
    main()
