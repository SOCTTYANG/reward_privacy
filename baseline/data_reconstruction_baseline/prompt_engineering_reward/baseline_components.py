from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse the reward-model implementation published with the main method.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_RECONSTRUCTION_DIR = REPOSITORY_ROOT / "data_reconstruction"
if str(DATA_RECONSTRUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_RECONSTRUCTION_DIR))

from safe_rlhf.dual_reward.dataset import format_prompt_response
from safe_rlhf.dual_reward.model import DualRewardModel


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from error
            if isinstance(row, dict):
                rows.append(row)
    return rows


def save_json(value: Any, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def first_nonempty(row: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def resolve_torch_dtype(name: str):
    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


class DualRewardScorer:
    def __init__(
        self,
        reward_model_path: str,
        device: str,
        max_length: int,
        score_head: str,
        lambda_safe: float,
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.score_head = score_head
        self.lambda_safe = lambda_safe
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = DualRewardModel.from_pretrained_reward(reward_model_path)
        self.model.eval().to(device)

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> Tuple[float, float, float]:
        encoded = self.tokenizer(
            format_prompt_response(prompt, response),
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        output = self.model(**encoded, return_dict=True)
        help_score = float(output.help_scores.reshape(-1)[0].cpu())
        safe_score = float(output.safe_scores.reshape(-1)[0].cpu())
        if self.score_head == "help":
            reward_score = help_score
        elif self.score_head == "safe":
            reward_score = safe_score
        elif self.score_head == "sum":
            reward_score = help_score + self.lambda_safe * safe_score
        else:
            raise ValueError(f"Unsupported score head: {self.score_head}")
        return reward_score, help_score, safe_score


class PromptEngineeringGenerator:
    def __init__(
        self,
        base_model_path: str,
        device: str,
        torch_dtype: str,
        device_map: str,
        max_input_length: int,
    ) -> None:
        self.device = device
        self.max_input_length = max_input_length
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": resolve_torch_dtype(torch_dtype),
            "low_cpu_mem_usage": True,
        }
        if device_map != "none":
            model_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
        if device_map == "none":
            self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate_one(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        retry_id: int = 0,
    ) -> str:
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        output = self.model.generate(
            **encoded,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature + 0.05 * retry_id,
            top_p=min(0.99, top_p + 0.01 * retry_id),
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        continuation = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(continuation, skip_special_tokens=True).strip()


def build_generation_prompt(
    x: str,
    y_plus: str,
    y_plus_reward: float,
    target_y_minus_reward: float,
    attempts: Sequence[Dict[str, Any]],
    score_head: str,
    max_feedback_chars: int,
) -> str:
    feedback = ""
    if attempts:
        lines = ["Previous attempts and reward feedback:"]
        for attempt in attempts:
            response = str(attempt["generated_y_minus"]).replace("\n", " ")
            response = response[:max_feedback_chars]
            lines.append(
                f"- Round {attempt['round']}: reward={float(attempt['reward_score']):.6f}; "
                f"response={response}"
            )
        feedback = "\n" + "\n".join(lines) + "\n"

    return (
        "Generate one plausible response that is less preferred than the provided preferred response.\n"
        "Its reward score should be as close as possible to the target reward score.\n"
        "Return only the new response, without explanations or score annotations.\n\n"
        f"User prompt:\n{x}\n\n"
        f"Preferred response:\n{y_plus}\n\n"
        f"Preferred-response {score_head} reward: {y_plus_reward:.6f}\n"
        f"Target dispreferred-response {score_head} reward: {target_y_minus_reward:.6f}\n"
        f"{feedback}\n"
        "Less-preferred response:\n"
    )


def build_summary(rows: Sequence[Dict[str, Any]], skipped: int, args) -> Dict[str, Any]:
    stopped = sum(bool(row.get("stopped_by_delta")) for row in rows)
    differences = [float(row["selected_abs_diff_to_real_reward"]) for row in rows]
    return {
        "num_output_rows": len(rows),
        "num_skipped_rows": skipped,
        "num_stopped_by_delta": stopped,
        "stop_rate": stopped / len(rows) if rows else 0.0,
        "mean_abs_reward_difference": sum(differences) / len(differences) if differences else None,
        "delta": args.delta,
        "max_rounds": args.max_rounds,
        "score_head": args.score_head,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative prompt-engineering reward baseline")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--reward_model_path", required=True)
    parser.add_argument("--triplet_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--summary_file", default="")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--generator_device", default="cuda:0")
    parser.add_argument("--reward_device", default="cuda:0")
    parser.add_argument("--generator_device_map", default="none")
    parser.add_argument("--torch_dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--generator_max_input_length", type=int, default=2048)
    parser.add_argument("--reward_max_length", type=int, default=512)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--score_head", choices=("help", "safe", "sum"), default="help")
    parser.add_argument("--lambda_safe", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--empty_retries", type=int, default=2)
    parser.add_argument("--max_feedback_chars", type=int, default=500)
    parser.add_argument("--save_generation_prompts", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
