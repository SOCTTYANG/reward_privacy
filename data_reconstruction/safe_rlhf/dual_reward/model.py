from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModel


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
    """
    LoRA 版双头奖励模型：
    - backbone: AutoModel (decoder-only backbone)
    - score_head: Linear(hidden_size -> 2)
        score[:, 0] = help_score
        score[:, 1] = safe_score
    """

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

        # score_head 用 float32 更稳一点，参数量很小，显存影响可忽略
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

        # 保存 LoRA adapter / base model
        if hasattr(self.base_model, "save_pretrained"):
            self.base_model.save_pretrained(save_directory)

        # 保存 score head
        torch.save(self.score_head.state_dict(), os.path.join(save_directory, "score_head.pt"))

        # 保存 dual reward 配置
        with open(os.path.join(save_directory, "dual_reward_config.json"), "w", encoding="utf-8") as f:
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

        hidden_states = outputs.last_hidden_state  # [B, T, H]

        if attention_mask is not None:
            last_token_idx = attention_mask.long().sum(dim=1) - 1
            last_token_hidden = hidden_states[
                torch.arange(hidden_states.size(0), device=hidden_states.device),
                last_token_idx,
            ]
        else:
            last_token_hidden = hidden_states[:, -1, :]

        # 避免 bf16/float32 冲突
        last_token_hidden = last_token_hidden.to(self.score_head.weight.dtype)

        scores = self.score_head(last_token_hidden)  # [B, 2]
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

    # 兼容 HF Trainer gradient checkpointing
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
