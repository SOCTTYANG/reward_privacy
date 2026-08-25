import argparse
import copy
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tool.paper_mia import (
    advantages,
    encode_policy_tokens,
    gradient_l2,
    load_artifact,
    ppo_clipped_loss,
    save_artifact,
    sequence_logprob,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.clip_eps < 1.0:
        raise ValueError("clip_eps must lie in (0,1)")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    torch.manual_seed(args.seed)
    data, step2_metadata = load_artifact(args.input_path, expected_stage=2)
    policy_spec = step2_metadata["policy_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(policy_spec["base_model"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    policy = AutoModelForCausalLM.from_pretrained(
        policy_spec["base_model"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device)
    policy.config.use_cache = False
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    reference = copy.deepcopy(policy).to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD(policy.parameters(), lr=args.learning_rate)
    for step, record in enumerate(tqdm(data, desc="Independent paper PPO probes"), start=1):
        policy.load_state_dict(reference.state_dict())
        policy.eval()
        reference.load_state_dict(policy.state_dict())
        candidates = record.get("candidate_responses", [])
        if len(candidates) != int(record.get("m", len(candidates))) or not candidates:
            raise ValueError(f"record {record.get('id')} has an invalid candidate set")
        rewards = torch.tensor(
            [float(candidate["r_i"]) for candidate in candidates],
            dtype=torch.float32,
            device=device,
        )
        advantage = advantages(rewards)
        sequence_length = max(
            len(record["policy_prompt_token_ids"]) + len(candidate["policy_token_ids"])
            for candidate in candidates
        )
        batch = encode_policy_tokens(
            tokenizer,
            record["policy_prompt_token_ids"],
            [candidate["policy_token_ids"] for candidate in candidates],
            sequence_length,
            device,
        )
        with torch.no_grad():
            reference_logprob = sequence_logprob(reference, **batch).detach()
        optimizer.zero_grad(set_to_none=True)
        current_logprob = sequence_logprob(policy, **batch)
        loss = ppo_clipped_loss(
            current_logprob,
            reference_logprob,
            advantage,
            args.clip_eps,
        )
        loss.backward()
        norm = gradient_l2(policy.parameters())
        optimizer.step()
        record["advantages"] = [float(value) for value in advantage.detach().cpu().tolist()]
        record["ppo_loss"] = float(loss.detach().float().cpu())
        record["grad_norm"] = float(norm)
        record["ppo_step"] = step
    metadata = {
        "stage": 3,
        "reward_model": step2_metadata["reward_model"],
        "reward_max_length": step2_metadata["reward_max_length"],
        "policy_model": policy_spec,
        "generation": step2_metadata["generation"],
        "ppo": {
            "learning_rate": args.learning_rate,
            "clip_eps": args.clip_eps,
            "complete_response_sequence": True,
            "parameter_scope": "all",
            "optimizer": "SGD",
            "probe_initialization": "same_pretrained_policy_for_every_target",
        },
    }
    save_artifact(data, metadata, args.output_path)
    print(f"[DONE] Step 3 wrote {len(data)} records to {args.output_path}")


if __name__ == "__main__":
    main()
