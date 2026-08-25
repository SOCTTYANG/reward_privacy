import argparse
import gc
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tool.paper_mia import load_artifact, policy_prompt_text, save_artifact, score_reward_pairs


def trim_generated_tokens(tokens, pad_token_id):
    tokens = [int(token) for token in tokens]
    while tokens and tokens[-1] == pad_token_id:
        tokens.pop()
    if not tokens:
        raise ValueError("Generator produced an empty candidate response")
    return tokens


@torch.no_grad()
def generate_candidates(
    model,
    tokenizer,
    prompts,
    m,
    batch_size,
    max_prompt_length,
    max_new_tokens,
    temperature,
    top_p,
    device,
):
    generated = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        source_prompts = prompts[start : start + batch_size]
        texts = [policy_prompt_text(tokenizer, prompt) for prompt in source_prompts]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        prompt_width = batch["input_ids"].shape[1]
        outputs = model.generate(
            **batch,
            do_sample=True,
            num_return_sequences=m,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for index in range(len(source_prompts)):
            prompt_ids = batch["input_ids"][index][batch["attention_mask"][index].bool()].tolist()
            candidates = []
            for candidate_index in range(m):
                output_index = index * m + candidate_index
                token_ids = trim_generated_tokens(
                    outputs[output_index, prompt_width:].detach().cpu().tolist(),
                    tokenizer.pad_token_id,
                )
                text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                if not text:
                    raise ValueError("Generator produced a response containing only special tokens")
                candidates.append(
                    {
                        "candidate_id": candidate_index,
                        "y_i": text,
                        "policy_token_ids": token_ids,
                    }
                )
            generated.append({"policy_prompt_token_ids": prompt_ids, "candidates": candidates})
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--pretrained_llm_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--generation_batch_size", type=int, default=1)
    parser.add_argument("--reward_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()
    if args.m < 1:
        raise ValueError("m must be positive")
    data, step1_metadata = load_artifact(args.input_path, expected_stage=1)
    reward_spec = step1_metadata["reward_model"]
    reward_max_length = int(step1_metadata["reward_max_length"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_llm_path, trust_remote_code=True)
    if policy_tokenizer.pad_token_id is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    policy_tokenizer.padding_side = "left"
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_llm_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device)
    generated = generate_candidates(
        policy_model,
        policy_tokenizer,
        [record["x"] for record in data],
        args.m,
        args.generation_batch_size,
        args.max_prompt_length,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        device,
    )
    del policy_model, policy_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_spec["base_model"], trust_remote_code=True)
    if reward_tokenizer.pad_token_id is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_spec["base_model"],
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if reward_spec.get("adapter_path"):
        reward_model = PeftModel.from_pretrained(reward_model, reward_spec["adapter_path"])
    reward_model.to(device).eval()
    pairs = [
        (record["x"], candidate["y_i"])
        for record, generated_record in zip(data, generated)
        for candidate in generated_record["candidates"]
    ]
    scores = score_reward_pairs(
        reward_model,
        reward_tokenizer,
        pairs,
        args.reward_batch_size,
        reward_max_length,
        device,
    )
    score_index = 0
    for record, generated_record in zip(data, generated):
        record["m"] = args.m
        record["policy_prompt_token_ids"] = generated_record["policy_prompt_token_ids"]
        record["candidate_responses"] = generated_record["candidates"]
        for candidate in record["candidate_responses"]:
            candidate["r_i"] = float(scores[score_index])
            score_index += 1
    metadata = {
        "stage": 2,
        "reward_model": reward_spec,
        "reward_max_length": reward_max_length,
        "policy_model": {"base_model": args.pretrained_llm_path},
        "generation": {
            "m": args.m,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    save_artifact(data, metadata, args.output_path)
    print(f"[DONE] Step 2 wrote {len(data)} records to {args.output_path}")


if __name__ == "__main__":
    main()
