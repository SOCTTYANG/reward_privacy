import os
import json
import math
import copy
import argparse
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved: {path}")


def build_policy_prompt(x: str) -> str:
    return str(x).strip()


def flatten_step2_data(step2_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples = []

    for item in step2_data:
        x = item["x"]
        candidates = item.get("candidate_responses", [])

        if len(candidates) == 0:
            continue

        rewards = [float(c["r_i"]) for c in candidates]
        mean_reward = sum(rewards) / len(rewards)

        for c in candidates:
            y = str(c.get("y_i", "")).strip()
            if y == "":
                continue

            r = float(c["r_i"])
            adv = r - mean_reward

            samples.append(
                {
                    "id": item.get("id"),
                    "membership": item.get("membership"),
                    "source_dataset": item.get("source_dataset"),
                    "source_index": item.get("source_index"),
                    "candidate_id": c.get("candidate_id"),
                    "x": x,
                    "y": y,
                    "reward": r,
                    "mean_reward": mean_reward,
                    "advantage": adv,
                }
            )

    return samples


class PPOCandidateDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 640,
        normalize_advantage: bool = False,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

        if normalize_advantage:
            advs = torch.tensor([float(x["advantage"]) for x in self.samples], dtype=torch.float32)
            mean = advs.mean().item()
            std = advs.std().item()
            std = std if std > 1e-6 else 1.0

            for item in self.samples:
                item["advantage"] = (float(item["advantage"]) - mean) / std

            print(f"[INFO] Advantage normalized: mean={mean:.6f}, std={std:.6f}")

    def __len__(self):
        return len(self.samples)

    def encode_prompt_response(self, prompt: str, response: str):
        prompt_text = build_policy_prompt(prompt)
        response_text = str(response).strip()

        full_text = prompt_text + response_text

        prompt_enc = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        full_enc = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        prompt_len = len(prompt_enc["input_ids"])
        prompt_len = min(prompt_len, len(input_ids))

        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len

        if all(x == -100 for x in labels):
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.encode_prompt_response(item["x"], item["y"])

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": enc["labels"],
            "advantage": float(item["advantage"]),
            "reward": float(item["reward"]),
        }


class PPOCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]):
        model_features = [
            {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]

        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]

        labels = []
        for f in features:
            label = f["labels"]
            pad_len = max_len - len(label)
            labels.append(label + [-100] * pad_len)

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["advantages"] = torch.tensor([f["advantage"] for f in features], dtype=torch.float32)
        batch["rewards"] = torch.tensor([f["reward"] for f in features], dtype=torch.float32)

        return batch


def sequence_logprob(model, input_ids, attention_mask, labels):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    mask = shift_labels.ne(-100)

    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    log_probs = F.log_softmax(shift_logits, dim=-1)

    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)

    token_log_probs = token_log_probs * mask.float()

    # pi(y|x) is the probability of the complete response sequence, so its
    # log-probability is the SUM of response-token log-probabilities.  Do not
    # length-normalize here: doing so would implement a geometric-mean token
    # probability rather than Eq. (3)'s sequence probability ratio.
    seq_log_probs = token_log_probs.sum(dim=-1)

    return seq_log_probs


def ppo_clipped_loss(new_logp, old_logp, advantages, clip_eps: float):
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    obj1 = ratio * advantages
    obj2 = clipped_ratio * advantages

    loss = -torch.min(obj1, obj2).mean()

    approx_kl = (old_logp - new_logp).mean().detach()
    clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().detach()

    return loss, ratio.detach(), approx_kl, clip_frac


def save_checkpoint(model, tokenizer, output_dir: str, step: int):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    print(f"[INFO] Saved checkpoint: {ckpt_dir}")


def infer_checkpoint_step(checkpoint_path: str) -> int:
    name = os.path.basename(os.path.normpath(checkpoint_path))
    if name.startswith("checkpoint-"):
        value = name[len("checkpoint-"):]
        if value.isdigit():
            return int(value)
    return 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="/mnt/bai_data/yang-safe/membership inference/3.2-Candidate Response Generation/3.2-Candidate Response Generation.json",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="/home/vipuser/Desktop/model/llama2-7b",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/bai_data/yang-safe/membership inference/3.3-LLM Update-FullPPO",
    )

    parser.add_argument("--max_length", type=int, default=640)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--clip_eps", type=float, default=0.2)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_global_step", type=int, default=None)
    parser.add_argument("--resume_skip_batches", type=int, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Step 3: paper-exact full-parameter PPO")
    print(f"[INFO] input_path: {args.input_path}")
    print(f"[INFO] base_model: {args.base_model}")
    print(f"[INFO] output_dir: {args.output_dir}")
    print(f"[INFO] clip_eps: {args.clip_eps}")
    print(f"[INFO] resume_from_checkpoint: {args.resume_from_checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    step2_data = load_json(args.input_path)
    print(f"[INFO] Loaded 3.2 records: {len(step2_data)}")

    samples = flatten_step2_data(step2_data)
    print(f"[INFO] Flattened PPO samples: {len(samples)}")

    member_count = sum(1 for x in samples if x.get("membership") == "member")
    nonmember_count = sum(1 for x in samples if x.get("membership") == "nonmember")

    print(f"[INFO] PPO member candidate samples: {member_count}")
    print(f"[INFO] PPO nonmember candidate samples: {nonmember_count}")

    save_json(samples, os.path.join(args.output_dir, "3.3_ppo_training_samples.json"))

    print("\n[INFO] Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"


    print("\n[INFO] Loading trainable policy model...")

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    policy_model.config.pad_token_id = tokenizer.pad_token_id
    policy_model.config.use_cache = False
    if args.resume_from_checkpoint is not None:
        print(f"[INFO] Loading full policy checkpoint: {args.resume_from_checkpoint}")
        policy_model = AutoModelForCausalLM.from_pretrained(
            args.resume_from_checkpoint,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
    for parameter in policy_model.parameters():
        parameter.requires_grad_(True)
    policy_model.to(device)
    policy_model.train()


    print("\n[INFO] Loading frozen old policy model...")

    # Both policies start from the same pretrained f, exactly as stated in the
    # paper.  This model is refreshed from pi_phi at each PPO epoch below.
    old_policy_model = copy.deepcopy(policy_model)

    old_policy_model.config.pad_token_id = tokenizer.pad_token_id
    old_policy_model.config.use_cache = False
    old_policy_model.to(device)
    old_policy_model.eval()

    for p in old_policy_model.parameters():
        p.requires_grad_(False)

    print("[INFO] Frozen old policy loaded.")

    dataset = PPOCandidateDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        normalize_advantage=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=PPOCollator(tokenizer),
    )

    # Eq. (3): phi <- phi - eta * g_phi.  Plain SGD implements that update
    # directly; AdamW and weight decay would define a different update rule.
    optimizer = torch.optim.SGD(
        policy_model.parameters(),
        lr=args.learning_rate,
    )

    total_update_steps = math.ceil(
        len(dataloader) * args.num_train_epochs / args.gradient_accumulation_steps
    )

    print(f"[INFO] Total dataloader batches: {len(dataloader)}")
    print(f"[INFO] Total optimizer update steps: {total_update_steps}")

    config_to_save = vars(args)
    config_to_save["num_ppo_samples"] = len(samples)
    config_to_save["member_candidate_samples"] = member_count
    config_to_save["nonmember_candidate_samples"] = nonmember_count
    config_to_save["ppo_type"] = "paper_exact_full_parameter_ppo"

    save_json(config_to_save, os.path.join(args.output_dir, "3.3_config.json"))

    resume_global_step = 0
    if args.resume_from_checkpoint is not None:
        resume_global_step = (
            args.resume_global_step
            if args.resume_global_step is not None
            else infer_checkpoint_step(args.resume_from_checkpoint)
        )

    resume_skip_batches = (
        args.resume_skip_batches
        if args.resume_skip_batches is not None
        else resume_global_step * args.gradient_accumulation_steps
    )

    if args.resume_from_checkpoint is not None:
        print(f"[INFO] Resuming from global_step={resume_global_step}")
        print(f"[INFO] Skipping already processed dataloader batches={resume_skip_batches}")

    global_step = resume_global_step

    running_loss = 0.0
    running_reward = 0.0
    running_adv = 0.0
    running_ratio = 0.0
    running_kl = 0.0
    running_clip_frac = 0.0
    log_count = 0

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.num_train_epochs):
        # At the beginning of every PPO step, reset pi_ref to the current
        # pi_phi, then keep it frozen during the updates in this step.
        old_policy_model.load_state_dict(policy_model.state_dict())
        old_policy_model.eval()
        progress = tqdm(dataloader, desc=f"Full PPO epoch {epoch + 1}/{args.num_train_epochs}")

        for batch_idx, batch in enumerate(progress):
            absolute_batch_idx = epoch * len(dataloader) + batch_idx
            if absolute_batch_idx < resume_skip_batches:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            advantages = batch["advantages"].to(device)
            rewards = batch["rewards"].to(device)

            with torch.no_grad():
                old_logp = sequence_logprob(
                    model=old_policy_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).detach()

            new_logp = sequence_logprob(
                model=policy_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss, ratio, approx_kl, clip_frac = ppo_clipped_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                advantages=advantages,
                clip_eps=args.clip_eps,
            )

            loss_to_backward = loss / args.gradient_accumulation_steps
            loss_to_backward.backward()

            running_loss += float(loss.detach().cpu())
            running_reward += float(rewards.mean().detach().cpu())
            running_adv += float(advantages.mean().detach().cpu())
            running_ratio += float(ratio.mean().detach().cpu())
            running_kl += float(approx_kl.cpu())
            running_clip_frac += float(clip_frac.cpu())
            log_count += 1

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / max(log_count, 1)
                    avg_reward = running_reward / max(log_count, 1)
                    avg_adv = running_adv / max(log_count, 1)
                    avg_ratio = running_ratio / max(log_count, 1)
                    avg_kl = running_kl / max(log_count, 1)
                    avg_clip_frac = running_clip_frac / max(log_count, 1)

                    print(
                        {
                            "step": global_step,
                            "epoch": epoch + 1,
                            "loss": round(avg_loss, 6),
                            "reward": round(avg_reward, 6),
                            "advantage": round(avg_adv, 6),
                            "ratio": round(avg_ratio, 6),
                            "approx_kl": round(avg_kl, 6),
                            "clip_frac": round(avg_clip_frac, 6),
                        }
                    )

                    running_loss = 0.0
                    running_reward = 0.0
                    running_adv = 0.0
                    running_ratio = 0.0
                    running_kl = 0.0
                    running_clip_frac = 0.0
                    log_count = 0

                if global_step % args.save_steps == 0:
                    save_checkpoint(policy_model, tokenizer, args.output_dir, global_step)

    save_checkpoint(policy_model, tokenizer, args.output_dir, global_step)

    final_dir = os.path.join(args.output_dir, "final_policy")
    policy_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    print("\n[DONE] Full PPO Step 3 finished.")
    print(f"[DONE] Final full policy saved to: {final_dir}")


if __name__ == "__main__":
    main()
