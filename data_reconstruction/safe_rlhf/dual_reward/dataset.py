from __future__ import annotations

from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


def format_prompt_response(prompt: str, response: str) -> str:
    return f"Human: {prompt}\nAssistant: {response}"


class PKUDualRewardDataset(Dataset):
    """
    PKU-SafeRLHF 双目标偏好数据集：
    - helpfulness preference: better_response_id
    - safety preference: safer_response_id

    每个样本返回四路输入：
    - help_chosen
    - help_rejected
    - safe_chosen
    - safe_rejected
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer,
        max_length: int = 64,
    ):
        self.raw = load_dataset(dataset_name, split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.raw)

    def _encode(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
        }

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, torch.Tensor]]:
        item = self.raw[idx]

        prompt = item["prompt"]
        response_0 = item["response_0"]
        response_1 = item["response_1"]

        better_id = int(item["better_response_id"])
        safer_id = int(item["safer_response_id"])

        help_pos = response_0 if better_id == 0 else response_1
        help_neg = response_1 if better_id == 0 else response_0

        safe_pos = response_0 if safer_id == 0 else response_1
        safe_neg = response_1 if safer_id == 0 else response_0

        return {
            "help_chosen": self._encode(format_prompt_response(prompt, help_pos)),
            "help_rejected": self._encode(format_prompt_response(prompt, help_neg)),
            "safe_chosen": self._encode(format_prompt_response(prompt, safe_pos)),
            "safe_rejected": self._encode(format_prompt_response(prompt, safe_neg)),
        }


class DualRewardCollator:
    """
    将四路样本分别 pad 成 batch。
    输出结构保持不变，供 trainer.compute_loss 直接取用。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

    def __call__(self, batch: List[Dict[str, Dict[str, torch.Tensor]]]):
        help_chosen = [x["help_chosen"] for x in batch]
        help_rejected = [x["help_rejected"] for x in batch]
        safe_chosen = [x["safe_chosen"] for x in batch]
        safe_rejected = [x["safe_rejected"] for x in batch]

        return {
            "help_chosen": self._pad(help_chosen),
            "help_rejected": self._pad(help_rejected),
            "safe_chosen": self._pad(safe_chosen),
            "safe_rejected": self._pad(safe_rejected),
        }
