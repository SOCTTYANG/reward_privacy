from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from step1_finetune_seq2seq import build_source_text_with_budget


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


def load_generator(
    base_model_path: str, adapter_dir: str, device: str, torch_dtype: str
):
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
    max_source_tokens: int = 192,
    max_new_tokens: int = 96,
) -> List[str]:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")

    source_text = build_source_text_with_budget(
        x=x,
        y_plus=y_plus,
        tokenizer=tokenizer,
        max_source_tokens=max_source_tokens,
    )
    inputs = tokenizer(source_text, return_tensors="pt").to(device)
    prompt_width = inputs["input_ids"].shape[1]

    sequences = inputs["input_ids"].expand(num_candidates, -1).clone()
    attention_mask = inputs["attention_mask"].expand(num_candidates, -1).clone()
    finished = torch.zeros(num_candidates, dtype=torch.bool, device=device)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError(
            "The generation tokenizer must define pad_token_id or eos_token_id."
        )

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=sequences,
            attention_mask=attention_mask,
            return_dict=True,
        )
        next_token_probabilities = torch.softmax(
            outputs.logits[:, -1, :].float(), dim=-1
        )
        next_tokens = torch.multinomial(
            next_token_probabilities, num_samples=1
        ).squeeze(-1)
        active = ~finished
        next_tokens = torch.where(
            active,
            next_tokens,
            torch.full_like(next_tokens, pad_id),
        )
        sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, active.long().unsqueeze(-1)],
            dim=1,
        )
        if tokenizer.eos_token_id is not None:
            finished = finished | (active & next_tokens.eq(tokenizer.eos_token_id))
            if bool(finished.all()):
                break

    generated_ids = sequences[:, prompt_width:]
    candidates = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return [candidate.strip() for candidate in candidates]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_candidates", type=int, default=3)
    parser.add_argument("--max_source_tokens", type=int, default=192)
    parser.add_argument("--max_new_tokens", type=int, default=96)
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
        data = data[: args.max_samples]

    print(f"[INFO] Loaded {len(data)} samples from: {args.input_file}")

    output_rows = []

    for idx, item in enumerate(data):
        forbidden_reference_fields = {"y_minus", "real_y_minus", "reference_y_minus"}
        leaked_fields = forbidden_reference_fields.intersection(item)
        if leaked_fields:
            raise ValueError(
                "Stage 2 input must contain only attack-visible target data; "
                f"found hidden reference fields: {sorted(leaked_fields)}"
            )
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
            max_source_tokens=args.max_source_tokens,
            max_new_tokens=args.max_new_tokens,
        )

        out_row = {
            "sample_id": item.get("sample_id", idx),
            "x": x,
            "y_plus": y_plus,
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
