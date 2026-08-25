from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer


@dataclass
class DualRewardOutput:
    help_scores: torch.Tensor
    safe_scores: torch.Tensor


@dataclass
class DualRewardConfig:
    backbone_name_or_path: str
    trust_remote_code: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    )


class DualRewardModel(nn.Module):
    def __init__(self, base_model: nn.Module, config: DualRewardConfig):
        super().__init__()
        self.base_model = base_model
        self.dual_reward_config = config

        hidden_size = self.base_model.config.hidden_size
        self.score_head = nn.Linear(hidden_size, 2)

    @classmethod
    def from_pretrained_backbone(
        cls,
        model_name_or_path: str,
        trust_remote_code: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ),
    ) -> "DualRewardModel":
        cfg = DualRewardConfig(
            backbone_name_or_path=model_name_or_path,
            trust_remote_code=trust_remote_code,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )

        base_model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
        )

        if use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
                bias="none",
            )
            base_model = get_peft_model(base_model, lora_cfg)

        model = cls(base_model=base_model, config=cfg)

        model.score_head = model.score_head.float()

        return model

    @classmethod
    def from_pretrained_reward(
        cls,
        reward_model_path: str,
        trust_remote_code: bool = True,
    ) -> "DualRewardModel":
        config_path = os.path.join(reward_model_path, "dual_reward_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing config file: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        cfg = DualRewardConfig(
            backbone_name_or_path=cfg_dict["backbone_name_or_path"],
            trust_remote_code=cfg_dict.get("trust_remote_code", True),
            use_lora=cfg_dict.get("use_lora", True),
            lora_r=cfg_dict.get("lora_r", 16),
            lora_alpha=cfg_dict.get("lora_alpha", 32),
            lora_dropout=cfg_dict.get("lora_dropout", 0.05),
            lora_target_modules=tuple(
                cfg_dict.get(
                    "lora_target_modules",
                    [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "up_proj",
                        "down_proj",
                        "gate_proj",
                    ],
                )
            ),
        )

        base_model = AutoModel.from_pretrained(
            cfg.backbone_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
        )

        if cfg.use_lora:
            base_model = PeftModel.from_pretrained(base_model, reward_model_path)

        model = cls(base_model=base_model, config=cfg)

        score_head_path = os.path.join(reward_model_path, "score_head.pt")
        if not os.path.exists(score_head_path):
            raise FileNotFoundError(f"Missing score head checkpoint: {score_head_path}")

        state_dict = torch.load(score_head_path, map_location="cpu")
        model.score_head.load_state_dict(state_dict)
        model.score_head = model.score_head.float()

        return model

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)

        if hasattr(self.base_model, "save_pretrained"):
            self.base_model.save_pretrained(save_directory)

        torch.save(
            self.score_head.state_dict(), os.path.join(save_directory, "score_head.pt")
        )

        with open(
            os.path.join(save_directory, "dual_reward_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(asdict(self.dual_reward_config), f, indent=2, ensure_ascii=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        return_dict: bool = True,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            last_token_idx = attention_mask.long().sum(dim=1) - 1
            last_token_hidden = hidden_states[
                torch.arange(hidden_states.size(0), device=hidden_states.device),
                last_token_idx,
            ]
        else:
            last_token_hidden = hidden_states[:, -1, :]

        last_token_hidden = last_token_hidden.to(self.score_head.weight.dtype)

        scores = self.score_head(last_token_hidden)
        help_scores = scores[:, 0]
        safe_scores = scores[:, 1]

        if return_dict:
            return DualRewardOutput(
                help_scores=help_scores,
                safe_scores=safe_scores,
            )

        return help_scores, safe_scores

    def print_trainable_parameters(self):
        total_params = 0
        trainable_params = 0
        for _, param in self.named_parameters():
            num = param.numel()
            total_params += num
            if param.requires_grad:
                trainable_params += num

        ratio = 100 * trainable_params / total_params if total_params > 0 else 0.0
        print(
            f"trainable params: {trainable_params:,} || "
            f"all params: {total_params:,} || "
            f"trainable%: {ratio:.4f}"
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.base_model.gradient_checkpointing_enable()
            else:
                self.base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )

    def gradient_checkpointing_disable(self):
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.base_model, "is_gradient_checkpointing", False)


def format_prompt_response(prompt: str, response: str) -> str:
    return f"Human: {prompt}\nAssistant: {response}"


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict], path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: object) -> str:
    return str(text).strip()


class DualRewardScorer:
    def __init__(
        self,
        reward_model_path: str,
        device: str = "cuda:0",
        max_length: int = 512,
        score_head: str = "help",
        lambda_safe: float = 1.0,
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.score_head = score_head
        self.lambda_safe = lambda_safe

        self.tokenizer = AutoTokenizer.from_pretrained(
            reward_model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = DualRewardModel.from_pretrained_reward(
            reward_model_path=reward_model_path,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> Tuple[float, float, float]:
        text = format_prompt_response(prompt, response)
        tok = self.tokenizer(
            text,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        if tok["attention_mask"].sum(dim=1).max().item() > self.max_length:
            raise ValueError("A complete (x, candidate) pair exceeds max_length.")
        tok = {k: v.to(self.device) for k, v in tok.items()}

        outputs = self.model(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            return_dict=True,
        )

        help_score = float(outputs.help_scores.reshape(-1)[0].detach().cpu().item())
        safe_score = float(outputs.safe_scores.reshape(-1)[0].detach().cpu().item())

        if self.score_head == "help":
            reward_score = help_score
        elif self.score_head == "safe":
            reward_score = safe_score
        elif self.score_head == "sum":
            reward_score = help_score + self.lambda_safe * safe_score
        else:
            raise ValueError(f"Unsupported score_head: {self.score_head}")

        return reward_score, help_score, safe_score


def select_lowest_reward(
    scorer: DualRewardScorer,
    x: str,
    candidate_y_minus_list: List[str],
) -> Tuple[List[float], List[float], List[float], int]:
    if not candidate_y_minus_list:
        raise ValueError("candidate_y_minus_list must contain at least one response.")

    candidate_reward_scores = []
    help_scores = []
    safe_scores = []

    for y_hat_i_minus in candidate_y_minus_list:
        r_hat_i, help_score, safe_score = scorer.score(x, y_hat_i_minus)
        if not math.isfinite(r_hat_i):
            raise ValueError("The target reward model returned a non-finite score.")
        candidate_reward_scores.append(r_hat_i)
        help_scores.append(help_score)
        safe_scores.append(safe_score)

    j = min(
        range(len(candidate_y_minus_list)),
        key=lambda i: candidate_reward_scores[i],
    )
    return candidate_reward_scores, help_scores, safe_scores, j


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the final reconstructed y_minus by the lowest reward-model score."
    )
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--stage2_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--expected_num_candidates", type=int, default=3)
    parser.add_argument(
        "--score_head",
        type=str,
        default="help",
        choices=("help", "safe", "sum"),
        help="Reward scalar used for argmin selection.",
    )
    parser.add_argument("--lambda_safe", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 80)
    print("[INFO] Loading dual reward model...")
    scorer = DualRewardScorer(
        reward_model_path=args.reward_model_path,
        device=args.device,
        max_length=args.max_length,
        score_head=args.score_head,
        lambda_safe=args.lambda_safe,
    )
    print("[INFO] Dual reward model loaded.")
    print(f"[INFO] Selection rule: argmin {args.score_head} reward score")
    print("=" * 80)

    stage2_rows = load_jsonl(args.stage2_file)

    if args.max_samples > 0:
        stage2_rows = stage2_rows[: args.max_samples]

    print(f"[INFO] Loaded stage2 rows : {len(stage2_rows)}")

    output_rows = []
    skipped = 0

    for idx, row in enumerate(stage2_rows):
        leaked_fields = {"y_minus", "real_y_minus", "reference_y_minus"}.intersection(
            row
        )
        if leaked_fields:
            raise ValueError(
                "Stage 3 must not receive the unknown y_minus; "
                f"found hidden reference fields: {sorted(leaked_fields)}"
            )
        x = normalize_text(row.get("x", ""))
        candidate_y_minus_list = row.get("candidate_y_minus_list", [])

        if (
            not x
            or not isinstance(candidate_y_minus_list, list)
            or len(candidate_y_minus_list) == 0
        ):
            skipped += 1
            continue
        if len(candidate_y_minus_list) != args.expected_num_candidates:
            raise ValueError(
                f"Sample {idx} has {len(candidate_y_minus_list)} candidates; "
                f"expected exactly {args.expected_num_candidates}."
            )

        y_plus = row.get("y_plus", "")
        candidate_reward_scores, help_scores, safe_scores, j = select_lowest_reward(
            scorer=scorer,
            x=x,
            candidate_y_minus_list=candidate_y_minus_list,
        )

        y_hat_star_minus = candidate_y_minus_list[j]
        selected_reward = candidate_reward_scores[j]

        output_rows.append(
            {
                "sample_id": row.get("sample_id", idx),
                "x": x,
                "y_plus": y_plus,
                "candidate_y_minus_list": candidate_y_minus_list,
                "candidate_reward_scores": candidate_reward_scores,
                "candidate_help_scores": help_scores,
                "candidate_safe_scores": safe_scores,
                "selection_strategy": f"lowest_{args.score_head}_reward",
                "lowest_reward_index": j,
                "lowest_reward_generated_y_minus": y_hat_star_minus,
                "lowest_reward_score": selected_reward,
                "selected_generated_y_minus": y_hat_star_minus,
                "selected_reward_score": selected_reward,
            }
        )

        if idx < 5:
            print(f"\n[Preview sample {idx}]")
            print("-" * 80)
            print("x:")
            print(x[:300])
            print("-" * 80)
            for i, cand in enumerate(candidate_y_minus_list):
                marker = " <-- selected" if i == j else ""
                print(
                    f"[candidate {i}] reward={candidate_reward_scores[i]:.6f}, "
                    f"help={help_scores[i]:.6f}, safe={safe_scores[i]:.6f}{marker}"
                )
                print(cand[:220])
                print("-" * 80)

        if (idx + 1) % 50 == 0:
            print(f"[INFO] Processed {idx + 1}/{len(stage2_rows)} samples")

    save_jsonl(output_rows, args.output_file)

    print("=" * 80)
    print(f"[INFO] Saved {len(output_rows)} rows to:")
    print(args.output_file)
    print(f"[INFO] Skipped rows: {skipped}")
    print("=" * 80)


if __name__ == "__main__":
    main()
