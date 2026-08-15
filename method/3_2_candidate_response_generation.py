import os
import json
import gc
import random
import argparse
from typing import List, Dict, Any, Tuple

import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import PeftModel


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
    return data


def save_json(data: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Saved to: {path}")
    print(f"[INFO] Total records: {len(data)}")


def normalize_item(obj: Dict[str, Any]) -> Dict[str, Any]:

    x = obj.get("x", obj.get("X", obj.get("prompt", "")))
    y_plus = obj.get("y_plus", obj.get("Y+", obj.get("chosen", obj.get("better", ""))))
    y_minus = obj.get("y_minus", obj.get("Y-", obj.get("rejected", obj.get("worse", ""))))

    return {
        "x": str(x),
        "y_plus": str(y_plus),
        "y_minus": str(y_minus),
    }


def build_step2_input(
    member_path: str,
    nonmember_path: str,
    member_size: int,
    nonmember_size: int,
    seed: int,
) -> List[Dict[str, Any]]:

    random.seed(seed)

    member_raw = load_jsonl(member_path)
    nonmember_raw = load_jsonl(nonmember_path)

    if len(member_raw) < member_size:
        raise ValueError(f"member 数据不足：需要 {member_size}，但只有 {len(member_raw)}")

    if len(nonmember_raw) < nonmember_size:
        raise ValueError(f"nonmember 数据不足：需要 {nonmember_size}，但只有 {len(nonmember_raw)}")


    member_selected = member_raw[:member_size]


    nonmember_selected = random.sample(nonmember_raw, nonmember_size)

    records = []

    for i, obj in enumerate(member_selected):
        item = normalize_item(obj)
        records.append(
            {
                "id": len(records),
                "membership": "member",
                "source_dataset": "member_train_512",
                "source_index": i,
                "x": item["x"],
                "y_plus": item["y_plus"],
                "y_minus": item["y_minus"],
            }
        )

    for i, obj in enumerate(nonmember_selected):
        item = normalize_item(obj)
        records.append(
            {
                "id": len(records),
                "membership": "nonmember",
                "source_dataset": "test_random_512",
                "source_index": i,
                "x": item["x"],
                "y_plus": item["y_plus"],
                "y_minus": item["y_minus"],
            }
        )

    print(f"[INFO] Built Step 2 input records: {len(records)}")
    print(f"[INFO] member records: {sum(1 for x in records if x['membership'] == 'member')}")
    print(f"[INFO] nonmember records: {sum(1 for x in records if x['membership'] == 'nonmember')}")

    return records


def build_generation_prompt(x: str) -> str:
    return x.strip()


@torch.no_grad()
def generate_candidate_responses(
    model,
    tokenizer,
    prompts: List[str],
    m: int,
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[List[str]]:
    all_candidates = []
    model.eval()

    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating candidate responses"):
        batch_prompts = prompts[start:start + batch_size]
        generation_prompts = [build_generation_prompt(x) for x in batch_prompts]

        inputs = tokenizer(
            generation_prompts,
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_length = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=m,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_only = outputs[:, input_length:]
        decoded = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

        for i in range(len(batch_prompts)):
            candidates = decoded[i * m:(i + 1) * m]
            candidates = [y.strip() for y in candidates]
            all_candidates.append(candidates)

    return all_candidates


def format_reward_text(prompt: str, response: str) -> str:

    return (
        "### Human:\n"
        + prompt.strip()
        + "\n\n### Assistant:\n"
        + response.strip()
    )


@torch.no_grad()
def score_pairs(
    model,
    tokenizer,
    pairs: List[Tuple[str, str]],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> List[float]:
    scores = []
    model.eval()

    for start in tqdm(range(0, len(pairs), batch_size), desc="Scoring candidate pairs"):
        batch_pairs = pairs[start:start + batch_size]

        texts = [format_reward_text(x, y) for x, y in batch_pairs]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits

        if logits.shape[-1] == 1:
            batch_scores = logits.squeeze(-1)
        elif logits.shape[-1] == 2:
            batch_scores = logits[:, 1]
        else:
            batch_scores = logits.max(dim=-1).values

        scores.extend(batch_scores.detach().cpu().float().tolist())

    return scores


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--member_path",
        type=str,
        default="/run/media/vipuser/data/yang-safe/data/pku30k_filtered/pku30k_train_cosine_le_085_xyz.jsonl",
    )

    parser.add_argument(
        "--nonmember_path",
        type=str,
        default="/run/media/vipuser/data/yang-safe/data/pku30k_filtered/pku30k_test_cosine_le_085_xyz.jsonl",
    )

    parser.add_argument(
        "--pretrained_llm_path",
        type=str,
        default="/home/vipuser/Desktop/model/llama2-7b",
    )

    parser.add_argument(
        "--reward_base_model",
        type=str,
        default="/home/vipuser/Desktop/model/llama2-7b",
    )

    parser.add_argument(
        "--reward_adapter_path",
        type=str,
        default="/run/media/vipuser/data/yang-safe/output/rm_lora_overfit_member512_e30/checkpoint-5000",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/run/media/vipuser/data/yang-safe/membership inference/3.2-Candidate Response Generation",
    )

    parser.add_argument(
        "--output_filename",
        type=str,
        default="3.2-Candidate Response Generation.json",
    )

    parser.add_argument("--member_size", type=int, default=512)
    parser.add_argument("--nonmember_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--m", type=int, default=3)

    # 生成参数
    parser.add_argument("--generation_batch_size", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)

    # reward scoring 参数
    parser.add_argument("--reward_batch_size", type=int, default=1)
    parser.add_argument("--reward_max_length", type=int, default=512)

    args = parser.parse_args()

    output_path = os.path.join(args.output_dir, args.output_filename)

    print("[INFO] Step 2: Candidate Response Generation")
    print(f"[INFO] Member path: {args.member_path}")
    print(f"[INFO] Nonmember path: {args.nonmember_path}")
    print(f"[INFO] Pretrained LLM: {args.pretrained_llm_path}")
    print(f"[INFO] Reward base model: {args.reward_base_model}")
    print(f"[INFO] Reward LoRA adapter: {args.reward_adapter_path}")
    print(f"[INFO] member_size = {args.member_size}")
    print(f"[INFO] nonmember_size = {args.nonmember_size}")
    print(f"[INFO] m = {args.m}")
    print(f"[INFO] Output path: {output_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    step2_data = build_step2_input(
        member_path=args.member_path,
        nonmember_path=args.nonmember_path,
        member_size=args.member_size,
        nonmember_size=args.nonmember_size,
        seed=args.seed,
    )

    prompts = [item["x"] for item in step2_data]


    print("\n[INFO] Loading LLaMA2-7B for generation...")

    gen_tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_llm_path,
        use_fast=False,
        trust_remote_code=True,
    )

    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    gen_tokenizer.padding_side = "left"

    gen_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_llm_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    print("\n[INFO] Generating candidate responses...")
    candidate_responses = generate_candidate_responses(
        model=gen_model,
        tokenizer=gen_tokenizer,
        prompts=prompts,
        m=args.m,
        batch_size=args.generation_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    del gen_model
    del gen_tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[INFO] Candidate generation finished.")


    candidate_pairs = []

    for item, candidates in zip(step2_data, candidate_responses):
        x = item["x"]
        for y in candidates:
            candidate_pairs.append((x, y))

    print(f"[INFO] Total candidate pairs to score: {len(candidate_pairs)}")


    print("\n[INFO] Loading target reward model: base + LoRA adapter...")

    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_base_model,
        use_fast=False,
        trust_remote_code=True,
    )

    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_base_model,
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    reward_model = PeftModel.from_pretrained(
        reward_model,
        args.reward_adapter_path,
    )

    reward_model.to(device)
    reward_model.eval()

    print("\n[INFO] Scoring generated candidate responses...")
    candidate_scores = score_pairs(
        model=reward_model,
        tokenizer=reward_tokenizer,
        pairs=candidate_pairs,
        batch_size=args.reward_batch_size,
        max_length=args.reward_max_length,
        device=device,
    )


    results = []
    score_idx = 0

    for idx, item in enumerate(step2_data):
        x = item["x"]
        candidates = []

        for j in range(args.m):
            y_i = candidate_responses[idx][j]
            r_i = float(candidate_scores[score_idx])
            score_idx += 1

            candidates.append(
                {
                    "candidate_id": j,
                    "y_i": y_i,
                    "r_i": r_i,
                    "pair": {
                        "x": x,
                        "y": y_i,
                        "reward_score": r_i,
                    },
                }
            )

        results.append(
            {
                "id": idx,
                "membership": item["membership"],
                "source_dataset": item["source_dataset"],
                "source_index": item["source_index"],

                "x": x,
                "y_plus": item.get("y_plus"),
                "y_minus": item.get("y_minus"),

                "m": args.m,
                "candidate_responses": candidates,
            }
        )

    save_json(results, output_path)

    print("\n[DONE] Step 2 finished successfully.")
    print(f"[DONE] Output file: {output_path}")


if __name__ == "__main__":
    main()
