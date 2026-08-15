"""Train a LoRA reward model with record-level DP-SGD.

One preference pair (prompt, chosen, rejected) is one private record.  Each
update uses Poisson sampling, clips each record's pairwise-ranking gradient,
adds Gaussian noise to the summed gradient, and records privacy loss with a
self-contained RDP accountant. Only LoRA adapters and the score head are trained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def build_text(prompt: str, response: str) -> str:
    return f"### Prompt:\n{prompt}\n\n### Response:\n{response}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PreferenceDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None) -> None:
        self.data = []
        for item in read_jsonl(path, max_samples):
            missing = {"prompt", "chosen", "rejected"} - item.keys()
            if missing:
                raise KeyError(f"{path} item is missing {sorted(missing)}")
            self.data.append({key: str(item[key]) for key in ("prompt", "chosen", "rejected")})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.data[index]


@dataclass
class PreferenceCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, rows: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        def tokenize(key: str) -> Dict[str, torch.Tensor]:
            return self.tokenizer(
                [build_text(row["prompt"], row[key]) for row in rows], padding=True,
                truncation=True, max_length=self.max_length, return_tensors="pt",
            )
        chosen, rejected = tokenize("chosen"), tokenize("rejected")
        return {
            "chosen_input_ids": chosen["input_ids"], "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"], "rejected_attention_mask": rejected["attention_mask"],
        }


def move(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def evaluate(model: Any, dataset: PreferenceDataset, collator: PreferenceCollator, args: argparse.Namespace) -> Dict[str, float]:
    model.eval()
    correct = total = 0
    gap_sum = loss_sum = 0.0
    for start in range(0, len(dataset), args.eval_batch_size):
        batch = move(collator(dataset.data[start : start + args.eval_batch_size]), args.device)
        chosen = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits.squeeze(-1)
        rejected = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits.squeeze(-1)
        gaps = chosen - rejected
        loss = -F.logsigmoid(gaps.float() - args.margin).mean()
        correct += (gaps > 0).sum().item()
        total += gaps.numel()
        gap_sum += gaps.float().sum().item()
        loss_sum += loss.item() * gaps.numel()
    return {"preference_acc": correct / max(total, 1), "avg_gap": gap_sum / max(total, 1), "ranking_loss": loss_sum / max(total, 1)}


def _log_add(log_x: float, log_y: float) -> float:
    if log_x == -math.inf:
        return log_y
    if log_y == -math.inf:
        return log_x
    maximum = max(log_x, log_y)
    return maximum + math.log(math.exp(log_x - maximum) + math.exp(log_y - maximum))


def _single_step_rdp_integer_order(noise_multiplier: float, sample_rate: float, order: int) -> float:
    """Exact RDP for Poisson-subsampled Gaussian noise at an integer order.

    This is the binomial expansion of the sampled-Gaussian moment.  Restricting
    to integer Renyi orders is a conservative accountant, so the reported
    epsilon is a valid upper bound without requiring Opacus or a Torch upgrade.
    """
    if noise_multiplier <= 0:
        return math.inf
    log_q = math.log(sample_rate)
    log_one_minus_q = math.log1p(-sample_rate)
    log_a = -math.inf
    for i in range(order + 1):
        log_binomial = math.lgamma(order + 1) - math.lgamma(i + 1) - math.lgamma(order - i + 1)
        log_term = log_binomial + i * log_q + (order - i) * log_one_minus_q
        log_term += (i * i - i) / (2 * noise_multiplier * noise_multiplier)
        log_a = _log_add(log_a, log_term)
    return log_a / (order - 1)


def epsilon_for(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    return min(
        steps * _single_step_rdp_integer_order(noise_multiplier, sample_rate, order)
        + math.log(1 / delta) / (order - 1)
        for order in range(2, 65)
    )


def calibrate_noise(target_epsilon: float, sample_rate: float, steps: int, delta: float) -> float:
    """Find the smallest Gaussian noise multiplier meeting the requested budget."""
    low, high = 0.01, 1.0
    while epsilon_for(high, sample_rate, steps, delta) > target_epsilon:
        high *= 2
        if high > 1_000:
            raise RuntimeError("Unable to calibrate a finite noise multiplier")
    for _ in range(40):
        middle = (low + high) / 2
        if epsilon_for(middle, sample_rate, steps, delta) <= target_epsilon:
            high = middle
        else:
            low = middle
    return high


def train(args: argparse.Namespace) -> None:
    if not 0 < args.delta < 1 or args.target_epsilon <= 0:
        raise ValueError("--delta must be in (0, 1) and --target_epsilon must be positive")
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=1, torch_dtype=dtype, trust_remote_code=True, device_map=None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.lora_target_modules.split(","),
        modules_to_save=["score"], bias="none",
    ))
    model.to(args.device)
    train_data = PreferenceDataset(args.train_path, args.max_train_samples)
    eval_data = PreferenceDataset(args.eval_path, args.max_eval_samples)
    if not train_data:
        raise ValueError("Training set is empty")
    collator = PreferenceCollator(tokenizer, args.max_length)
    steps = math.ceil(len(train_data) * args.epochs / args.lot_size)
    sample_rate = args.lot_size / len(train_data)
    noise_multiplier = args.noise_multiplier or calibrate_noise(args.target_epsilon, sample_rate, steps, args.delta)
    planned_epsilon = epsilon_for(noise_multiplier, sample_rate, steps, args.delta)
    print(f"[DP] records={len(train_data)} steps={steps} q={sample_rate:.8f} C={args.max_grad_norm} sigma={noise_multiplier:.6f}")
    print(f"[DP] planned epsilon={planned_epsilon:.6f}, delta={args.delta}")
    if planned_epsilon > args.target_epsilon * 1.0001:
        raise ValueError("Specified --noise_multiplier exceeds --target_epsilon; omit it to calibrate automatically")
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * args.warmup_ratio), steps)
    print("[EVAL] before", evaluate(model, eval_data, collator, args))
    model.train()
    progress = tqdm(range(steps), desc="DP-SGD")
    for update in progress:
        # Poisson sampling matches the subsampled-Gaussian RDP accountant.
        selected = torch.nonzero(torch.rand(len(train_data)) < sample_rate, as_tuple=False).flatten().tolist()
        summed_grads = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in params]
        batch_loss = 0.0
        for index in selected:
            batch = move(collator([train_data[index]]), args.device)
            chosen = model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits.squeeze(-1)
            rejected = model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits.squeeze(-1)
            loss = -F.logsigmoid((chosen - rejected).float() - args.margin).mean()
            loss.backward()
            squared_norm = sum((parameter.grad.detach().float().norm(2) ** 2 for parameter in params if parameter.grad is not None), torch.zeros((), device=args.device))
            clip = min(1.0, args.max_grad_norm / (squared_norm.sqrt().item() + 1e-12))
            for accumulator, parameter in zip(summed_grads, params):
                if parameter.grad is not None:
                    accumulator.add_(parameter.grad.detach().float(), alpha=clip)
            optimizer.zero_grad(set_to_none=True)
            batch_loss += loss.item()
        # Empty Poisson lots are retained as a noise-only update.  Resampling
        # them would condition the mechanism and invalidate this accountant.
        denominator = float(max(len(selected), 1))
        for accumulator, parameter in zip(summed_grads, params):
            noise = torch.randn_like(accumulator) * (noise_multiplier * args.max_grad_norm)
            parameter.grad = ((accumulator + noise) / denominator).to(dtype=parameter.dtype)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        epsilon = epsilon_for(noise_multiplier, sample_rate, update + 1, args.delta)
        progress.set_postfix(loss=f"{batch_loss / max(len(selected), 1):.4f}", records=len(selected), eps=f"{epsilon:.3f}")
    metrics = evaluate(model, eval_data, collator, args)
    spent_epsilon = epsilon_for(noise_multiplier, sample_rate, steps, args.delta)
    privacy = {
        "epsilon": spent_epsilon, "delta": args.delta, "target_epsilon": args.target_epsilon,
        "noise_multiplier": noise_multiplier, "max_grad_norm": args.max_grad_norm,
        "sample_rate": sample_rate, "steps": steps, "record_definition": "one prompt/chosen/rejected preference pair",
    }
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "target_rm_llama_lora_dp_metrics.json"), "w", encoding="utf-8") as file:
        json.dump({"metrics": metrics, "privacy": privacy}, file, ensure_ascii=False, indent=2)
    print(f"[OK] DP teacher saved to {args.output_dir}; metrics={metrics}; privacy={privacy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_samples", type=int, default=30_000)
    parser.add_argument("--max_eval_samples", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lot_size", type=int, default=16, help="Expected Poisson-sampled records per DP update")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--margin", type=float, default=5.0)
    parser.add_argument("--target_epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--noise_multiplier", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
