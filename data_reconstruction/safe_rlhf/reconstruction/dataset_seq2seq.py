from __future__ import annotations

import json
from typing import Dict, List

from torch.utils.data import Dataset


INSTRUCTION_PREFIX = (
    "You are given a user prompt and a preferred response. "
    "Generate an alternative response that is less preferred than the preferred response. "
    "The response should still be a plausible answer to the prompt, "
    "but it should be less helpful, less direct, less complete, or less useful.\n\n"
)
PROMPT_PREFIX = "User prompt:\n"
PREFERRED_PREFIX = "\n\nPreferred response:\n"
TARGET_PREFIX = "\n\nLess-preferred alternative response:\n"


def build_source_text(x: str, y_plus: str) -> str:
    return (
        INSTRUCTION_PREFIX
        + PROMPT_PREFIX + x
        + PREFERRED_PREFIX + y_plus
        + TARGET_PREFIX
    )


class ReconstructionTripletDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 320,
        min_target_tokens: int = 4,
        target_reserved_tokens: int = 128,
        max_prompt_tokens: int = 56,
        max_preferred_tokens: int = 24,
    ) -> None:
        self.samples: List[Dict[str, str]] = []
        dropped = 0

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_target_tokens = min_target_tokens
        self.target_reserved_tokens = target_reserved_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.max_preferred_tokens = max_preferred_tokens

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                x = str(obj.get("x", "")).strip()
                y_plus = str(obj.get("y_plus", "")).strip()
                y_minus = str(obj.get("y_minus", "")).strip()

                if not x or not y_plus or not y_minus:
                    dropped += 1
                    continue

                x_trunc = self._truncate_text_from_end(x, self.max_prompt_tokens)
                y_plus_trunc = self._truncate_text_from_end(y_plus, self.max_preferred_tokens)

                source_text = build_source_text(x_trunc, y_plus_trunc)

                y_minus_ids = self.tokenizer(
                    y_minus,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.target_reserved_tokens,
                )["input_ids"]

                if len(y_minus_ids) < self.min_target_tokens:
                    dropped += 1
                    continue

                source_ids = self.tokenizer(
                    source_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]

                available_for_target = self.max_length - len(source_ids)
                if available_for_target < self.min_target_tokens:
                    dropped += 1
                    continue

                y_minus_trunc = self.tokenizer.decode(y_minus_ids, skip_special_tokens=True)

                self.samples.append(
                    {
                        "source_text": source_text,
                        "target_text": y_minus_trunc,
                    }
                )

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {file_path}")

        print(f"[INFO] Loaded {len(self.samples)} valid samples from {file_path}")
        print(f"[INFO] Dropped {dropped} invalid/too-long samples from {file_path}")

    def _truncate_text_from_end(self, text: str, max_tokens: int) -> str:
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if len(ids) <= max_tokens:
            return text

        ids = ids[-max_tokens:]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]


class ReconstructionCollator:
    def __init__(self, tokenizer, max_length: int = 320) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        sources = [x["source_text"] for x in batch]
        targets = [x["target_text"] for x in batch]
        full_texts = [s + t for s, t in zip(sources, targets)]

        tokenized_full = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized_source = self.tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        labels = input_ids.clone()

        source_lengths = tokenized_source["attention_mask"].sum(dim=1)
        for i in range(len(batch)):
            labels[i, : source_lengths[i]] = -100

        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source_texts": sources,
            "target_texts": targets,
        }
