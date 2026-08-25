from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


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
        + PROMPT_PREFIX
        + x
        + PREFERRED_PREFIX
        + y_plus
        + TARGET_PREFIX
    )


def _truncate_text_from_end(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True)


def build_source_text_with_budget(
    x: str,
    y_plus: str,
    tokenizer,
    max_source_tokens: Optional[int],
) -> str:
    source_text = build_source_text(x, y_plus)
    if max_source_tokens is None:
        return source_text

    source_ids = tokenizer(
        source_text,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    if len(source_ids) <= max_source_tokens:
        return source_text

    fixed_text = build_source_text("", "")
    fixed_tokens = len(
        tokenizer(fixed_text, add_special_tokens=True, truncation=False)["input_ids"]
    )
    field_budget = max_source_tokens - fixed_tokens
    if field_budget < 2:
        raise ValueError(
            "max_source_tokens is too small to preserve the reconstruction prompt format."
        )

    x_ids = tokenizer(x, add_special_tokens=False, truncation=False)["input_ids"]
    y_plus_ids = tokenizer(y_plus, add_special_tokens=False, truncation=False)[
        "input_ids"
    ]
    total_field_tokens = max(len(x_ids) + len(y_plus_ids), 1)
    x_budget = max(1, round(field_budget * len(x_ids) / total_field_tokens))
    y_plus_budget = max(1, field_budget - x_budget)

    while True:
        x_trunc = _truncate_text_from_end(tokenizer, x, x_budget)
        y_plus_trunc = _truncate_text_from_end(tokenizer, y_plus, y_plus_budget)
        source_text = build_source_text(x_trunc, y_plus_trunc)
        source_length = len(
            tokenizer(source_text, add_special_tokens=True, truncation=False)[
                "input_ids"
            ]
        )
        if source_length <= max_source_tokens:
            return source_text
        if x_budget >= y_plus_budget and x_budget > 1:
            x_budget -= 1
        elif y_plus_budget > 1:
            y_plus_budget -= 1
        else:
            raise ValueError(
                "Unable to fit reconstruction source into max_source_tokens."
            )


class ReconstructionTripletDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 320,
        min_target_tokens: int = 4,
        target_reserved_tokens: int = 128,
        max_samples: Optional[int] = None,
    ) -> None:
        self.samples: List[Dict[str, str]] = []
        dropped = 0

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_target_tokens = min_target_tokens
        self.target_reserved_tokens = target_reserved_tokens
        self.max_source_tokens = max_length - target_reserved_tokens
        self.max_samples = max_samples

        if self.max_source_tokens <= 0:
            raise ValueError("target_reserved_tokens must be smaller than max_length.")

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

                source_text = build_source_text_with_budget(
                    x=x,
                    y_plus=y_plus,
                    tokenizer=self.tokenizer,
                    max_source_tokens=self.max_source_tokens,
                )

                full_y_minus_ids = self.tokenizer(
                    y_minus,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
                y_minus_ids = full_y_minus_ids[: self.target_reserved_tokens]

                if len(y_minus_ids) < self.min_target_tokens:
                    dropped += 1
                    continue

                source_ids = self.tokenizer(
                    source_text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]

                available_for_target = self.max_length - len(source_ids)
                if available_for_target < self.min_target_tokens:
                    dropped += 1
                    continue

                y_minus_target = (
                    y_minus
                    if len(full_y_minus_ids) <= self.target_reserved_tokens
                    else self.tokenizer.decode(y_minus_ids, skip_special_tokens=True)
                )

                self.samples.append(
                    {
                        "source_text": source_text,
                        "target_text": y_minus_target,
                    }
                )

                if (
                    self.max_samples is not None
                    and len(self.samples) >= self.max_samples
                ):
                    break

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {file_path}")

        print(f"[INFO] Loaded {len(self.samples)} valid samples from {file_path}")
        print(f"[INFO] Dropped {dropped} invalid/too-long samples from {file_path}")

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


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class ReconstructionModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        extractor_name_or_path: str,
        torch_dtype: torch.dtype | str = "auto",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.lm = get_peft_model(self.lm, lora_config)
        self.config = self.lm.config

        self.extractor_tokenizer = AutoTokenizer.from_pretrained(
            extractor_name_or_path,
            trust_remote_code=True,
        )
        self.extractor = AutoModel.from_pretrained(
            extractor_name_or_path,
            trust_remote_code=True,
        )
        self.extractor.eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        self.extractor_name_or_path = extractor_name_or_path

    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        return self.lm(*args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            return self.lm.gradient_checkpointing_enable()
        return self.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        return self.lm.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.lm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.lm.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None):
        return self.lm.resize_token_embeddings(
            new_num_tokens=new_num_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.lm.prepare_inputs_for_generation(*args, **kwargs)

    @property
    def device(self):
        return self.lm.device

    @torch.no_grad()
    def encode_texts_with_extractor(
        self, texts: List[str], device: torch.device
    ) -> torch.Tensor:
        tok = self.extractor_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}

        out = self.extractor(**tok, return_dict=True)

        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            emb = masked_mean(out.last_hidden_state, tok["attention_mask"])
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            raise ValueError("Extractor output does not contain usable embeddings.")

        emb = F.normalize(emb, dim=-1)
        return emb

    def save_reconstruction_pretrained(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.lm.save_pretrained(output_dir)

        with open(
            os.path.join(output_dir, "reconstruction_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {
                    "type": "mle_plus_sampled_response_cosine_reinforce",
                    "extractor_name_or_path": self.extractor_name_or_path,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


class ReconstructionTrainer(Trainer):
    def __init__(
        self,
        *args,
        lambda_mle: float = 1.0,
        lambda_cos: float = 0.05,
        generation_tokenizer=None,
        cosine_max_new_tokens: int = 96,
        reward_baseline_momentum: float = 0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if generation_tokenizer is None:
            raise ValueError(
                "generation_tokenizer is required for sampled cosine loss."
            )
        self.lambda_mle = lambda_mle
        self.lambda_cos = lambda_cos
        self.generation_tokenizer = generation_tokenizer
        self.cosine_max_new_tokens = cosine_max_new_tokens
        self.reward_baseline_momentum = reward_baseline_momentum
        self._reward_baseline = 0.0

    @staticmethod
    def _core_model(model):
        while hasattr(model, "module") and not hasattr(model, "lm"):
            model = model.module
        return model

    @torch.no_grad()
    def _sample_responses(
        self,
        model,
        source_texts: List[str],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        tokenizer = self.generation_tokenizer
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            source_batch = tokenizer(
                source_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = old_padding_side

        core_model = self._core_model(model)
        device = next(core_model.lm.parameters()).device
        source_batch = {key: value.to(device) for key, value in source_batch.items()}
        prompt_width = source_batch["input_ids"].shape[1]

        was_training = core_model.lm.training
        core_model.lm.eval()
        sequences = source_batch["input_ids"]
        generation_attention_mask = source_batch["attention_mask"]
        finished = torch.zeros(sequences.shape[0], dtype=torch.bool, device=device)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError(
                "The generation tokenizer must define pad_token_id or eos_token_id."
            )

        for _ in range(self.cosine_max_new_tokens):
            generation_outputs = core_model.lm(
                input_ids=sequences,
                attention_mask=generation_attention_mask,
                return_dict=True,
            )
            next_token_probabilities = F.softmax(
                generation_outputs.logits[:, -1, :].float(),
                dim=-1,
            )
            next_tokens = torch.multinomial(next_token_probabilities, num_samples=1)
            next_tokens = next_tokens.squeeze(-1)
            active = ~finished
            next_tokens = torch.where(
                active,
                next_tokens,
                torch.full_like(next_tokens, pad_id),
            )
            sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=1)
            generation_attention_mask = torch.cat(
                [generation_attention_mask, active.long().unsqueeze(-1)],
                dim=1,
            )

            if tokenizer.eos_token_id is not None:
                finished = finished | (active & next_tokens.eq(tokenizer.eos_token_id))
                if bool(finished.all()):
                    break
        if was_training:
            core_model.lm.train()

        generated_ids = sequences[:, prompt_width:]
        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        generated_texts = [text.strip() for text in generated_texts]
        return generated_texts, sequences, source_batch["attention_mask"]

    def _sample_log_prob(
        self,
        model,
        sequences: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokenizer = self.generation_tokenizer
        prompt_width = source_attention_mask.shape[1]
        generated_ids = sequences[:, prompt_width:]

        generated_mask = torch.ones_like(generated_ids, dtype=torch.long)
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            for row_index, row in enumerate(generated_ids):
                eos_positions = (row == eos_id).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    first_eos = int(eos_positions[0].item())
                    generated_mask[row_index, first_eos + 1 :] = 0

        attention_mask = torch.cat([source_attention_mask, generated_mask], dim=1)
        outputs = model(input_ids=sequences, attention_mask=attention_mask)
        shift_logits = outputs.logits[:, :-1, :]
        shift_tokens = sequences[:, 1:]
        token_log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        token_log_probs = token_log_probs.gather(
            dim=-1,
            index=shift_tokens.unsqueeze(-1),
        ).squeeze(-1)

        score_mask = attention_mask[:, 1:].bool()
        score_mask[:, : prompt_width - 1] = False
        return (token_log_probs * score_mask).sum(dim=-1)

    def compute_loss(
        self, model, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs
    ):
        source_texts = inputs.pop("source_texts")
        target_texts = inputs.pop("target_texts")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        shift_logits = outputs.logits[:, :-1, :].float()
        shift_labels = inputs["labels"][:, 1:]
        target_mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~target_mask, 0)
        target_log_probs = (
            F.log_softmax(shift_logits, dim=-1)
            .gather(
                dim=-1,
                index=safe_labels.unsqueeze(-1),
            )
            .squeeze(-1)
        )
        sequence_negative_log_likelihood = -(target_log_probs * target_mask).sum(dim=-1)
        l_mle = sequence_negative_log_likelihood.mean()

        if self.lambda_cos == 0.0:
            cos_sim = l_mle.new_zeros(())
            l_cos = l_mle.new_zeros(())
            reinforce_loss = l_mle.new_zeros(())
        else:
            generated_texts, sequences, source_attention_mask = self._sample_responses(
                model=model,
                source_texts=source_texts,
            )
            core_model = self._core_model(model)
            generated_repr = core_model.encode_texts_with_extractor(
                texts=generated_texts,
                device=l_mle.device,
            )
            target_repr = core_model.encode_texts_with_extractor(
                texts=target_texts,
                device=l_mle.device,
            )
            cosine_rewards = F.cosine_similarity(generated_repr, target_repr, dim=-1)
            cos_sim = cosine_rewards.mean()
            paper_l_cos = -cos_sim

            sequence_log_probs = self._sample_log_prob(
                model=model,
                sequences=sequences,
                source_attention_mask=source_attention_mask,
            )
            advantages = cosine_rewards.detach() - self._reward_baseline
            reinforce_loss = -(advantages * sequence_log_probs).mean()

            batch_reward = float(cos_sim.detach().cpu())
            self._reward_baseline = (
                self.reward_baseline_momentum * self._reward_baseline
                + (1.0 - self.reward_baseline_momentum) * batch_reward
            )

            l_cos = paper_l_cos.detach() + reinforce_loss - reinforce_loss.detach()

        total_loss = self.lambda_mle * l_mle + self.lambda_cos * l_cos
        self.log(
            {
                "train/loss_mle": float(l_mle.detach().cpu()),
                "train/loss_cos": float(l_cos.detach().cpu()),
                "train/cos_sim": float(cos_sim.detach().cpu()),
                "train/loss_cos_reinforce": float(reinforce_loss.detach().cpu()),
                "train/reward_baseline": self._reward_baseline,
                "train/loss_total": float(total_loss.detach().cpu()),
            }
        )

        if return_outputs:
            return total_loss, outputs
        return total_loss


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="")
    extractor_name_or_path: str = field(default="")
    torch_dtype: str = field(default="auto")
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)


@dataclass
class DataArguments:
    dataset_name: str = field(default="")
    train_file: str = field(default="")
    eval_file: str = field(default="")
    processed_dir: str = field(default="./data/preference_triplets")
    max_length: int = field(default=320)
    target_reserved_tokens: int = field(default=128)
    max_train_samples: int = field(default=2000)
    eval_ratio: float = field(default=0.05)
    split_seed: int = field(default=42)
    force_rebuild: bool = field(default=False)


@dataclass
class CustomArguments:
    lambda_mle: float = field(default=1.0)
    lambda_cos: float = field(default=0.05)
    cosine_max_new_tokens: int = field(default=96)
    reward_baseline_momentum: float = field(default=0.9)


def normalize_training_argument_aliases() -> None:
    training_arg_names = {item.name for item in fields(TrainingArguments)}
    if (
        "eval_strategy" not in training_arg_names
        and "evaluation_strategy" in training_arg_names
    ):
        sys.argv = [
            "--evaluation_strategy" if arg == "--eval_strategy" else arg
            for arg in sys.argv
        ]
    elif (
        "evaluation_strategy" not in training_arg_names
        and "eval_strategy" in training_arg_names
    ):
        sys.argv = [
            "--eval_strategy" if arg == "--evaluation_strategy" else arg
            for arg in sys.argv
        ]

    if "overwrite_output_dir" not in training_arg_names:
        filtered = [sys.argv[0]]
        index = 1
        while index < len(sys.argv):
            arg = sys.argv[index]
            if arg == "--overwrite_output_dir":
                index += (
                    2
                    if index + 1 < len(sys.argv)
                    and not sys.argv[index + 1].startswith("--")
                    else 1
                )
                continue
            filtered.append(arg)
            index += 1
        sys.argv = filtered


def resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch_dtype: {dtype_name}")


def convert_example_to_triplet(example: dict) -> dict | None:
    x = str(example.get("prompt", "")).strip()
    r0 = str(example.get("response_0", "")).strip()
    r1 = str(example.get("response_1", "")).strip()
    better_id = example.get("better_response_id", None)

    if not x or not r0 or not r1 or better_id is None:
        return None

    better_id = int(better_id)
    if better_id == 0:
        y_plus, y_minus = r0, r1
    elif better_id == 1:
        y_plus, y_minus = r1, r0
    else:
        return None

    return {
        "x": x,
        "y_plus": y_plus,
        "y_minus": y_minus,
    }


def build_local_triplet_files(
    dataset_name: str,
    processed_dir: str,
    eval_ratio: float,
    split_seed: int,
    force_rebuild: bool,
) -> tuple[str, str]:
    os.makedirs(processed_dir, exist_ok=True)
    train_file = os.path.join(processed_dir, "train.jsonl")
    eval_file = os.path.join(processed_dir, "eval.jsonl")

    if not force_rebuild and os.path.exists(train_file) and os.path.exists(eval_file):
        print("[INFO] Reusing existing processed triplet files.")
        return train_file, eval_file

    print(f"[INFO] Downloading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")

    print(f"[INFO] Raw dataset size = {len(ds)}")
    ds = ds.train_test_split(test_size=eval_ratio, seed=split_seed)

    train_ds = ds["train"]
    eval_ds = ds["test"]

    train_count = 0
    eval_count = 0

    with open(train_file, "w", encoding="utf-8") as f:
        for ex in train_ds:
            row = convert_example_to_triplet(ex)
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            train_count += 1

    with open(eval_file, "w", encoding="utf-8") as f:
        for ex in eval_ds:
            row = convert_example_to_triplet(ex)
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            eval_count += 1

    print(f"[INFO] Saved processed train triplets: {train_count}")
    print(f"[INFO] Saved processed eval triplets : {eval_count}")
    print(f"[INFO] train_file = {train_file}")
    print(f"[INFO] eval_file  = {eval_file}")

    return train_file, eval_file


def main():
    normalize_training_argument_aliases()
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, CustomArguments)
    )
    model_args, data_args, training_args, custom_args = (
        parser.parse_args_into_dataclasses()
    )

    print("=" * 80)
    print("[INFO] Parsed arguments successfully.")
    print(f"[INFO] model_name_or_path      = {model_args.model_name_or_path}")
    print(f"[INFO] extractor_name_or_path = {model_args.extractor_name_or_path}")
    print(f"[INFO] torch_dtype            = {model_args.torch_dtype}")
    print(f"[INFO] dataset_name           = {data_args.dataset_name}")
    print(f"[INFO] train_file             = {data_args.train_file}")
    print(f"[INFO] eval_file              = {data_args.eval_file}")
    print(f"[INFO] processed_dir          = {data_args.processed_dir}")
    print(f"[INFO] output_dir             = {training_args.output_dir}")
    print(f"[INFO] lambda_mle             = {custom_args.lambda_mle}")
    print(f"[INFO] lambda_cos             = {custom_args.lambda_cos}")
    print(f"[INFO] cosine_max_new_tokens  = {custom_args.cosine_max_new_tokens}")
    print("=" * 80)

    has_train_file = bool(data_args.train_file.strip())
    has_eval_file = bool(data_args.eval_file.strip())
    if has_train_file or has_eval_file:
        if not (has_train_file and has_eval_file):
            raise ValueError(
                "Please pass both --train_file and --eval_file, or neither."
            )
        train_file = data_args.train_file
        eval_file = data_args.eval_file
        if not os.path.isfile(train_file):
            raise FileNotFoundError(f"--train_file does not exist: {train_file}")
        if not os.path.isfile(eval_file):
            raise FileNotFoundError(f"--eval_file does not exist: {eval_file}")
        print("[INFO] Using explicit local triplet files.")
    else:
        train_file, eval_file = build_local_triplet_files(
            dataset_name=data_args.dataset_name,
            processed_dir=data_args.processed_dir,
            eval_ratio=data_args.eval_ratio,
            split_seed=data_args.split_seed,
            force_rebuild=data_args.force_rebuild,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Tokenizer loaded.")

    train_dataset = ReconstructionTripletDataset(
        train_file,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        min_target_tokens=4,
        target_reserved_tokens=data_args.target_reserved_tokens,
        max_samples=(
            data_args.max_train_samples if data_args.max_train_samples > 0 else None
        ),
    )
    eval_dataset = ReconstructionTripletDataset(
        eval_file,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        min_target_tokens=4,
        target_reserved_tokens=data_args.target_reserved_tokens,
    )

    print(f"[INFO] train_dataset size = {len(train_dataset)}")
    print(f"[INFO] eval_dataset size  = {len(eval_dataset)}")

    first_sample = train_dataset[0]
    print("[INFO] First sample source preview:")
    print(first_sample["source_text"][:500])
    print("-" * 80)
    print("[INFO] First sample target preview:")
    print(first_sample["target_text"][:300])
    print("=" * 80)

    model = ReconstructionModel(
        model_name_or_path=model_args.model_name_or_path,
        extractor_name_or_path=model_args.extractor_name_or_path,
        torch_dtype=resolve_torch_dtype(model_args.torch_dtype),
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )

    print("[INFO] ReconstructionModel loaded.")

    data_collator = ReconstructionCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_length,
    )

    trainer = ReconstructionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        lambda_mle=custom_args.lambda_mle,
        lambda_cos=custom_args.lambda_cos,
        generation_tokenizer=tokenizer,
        cosine_max_new_tokens=custom_args.cosine_max_new_tokens,
        reward_baseline_momentum=custom_args.reward_baseline_momentum,
    )

    print("[INFO] Starting trainer.train() ...")
    train_result = trainer.train()
    print("[INFO] trainer.train() finished.")
    print(train_result)

    os.makedirs(training_args.output_dir, exist_ok=True)
    model.save_reconstruction_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"[INFO] Saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
