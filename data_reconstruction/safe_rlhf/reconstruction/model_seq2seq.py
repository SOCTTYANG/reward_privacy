from __future__ import annotations

import json
import os
from typing import List

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


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

        # 冻结的 embedding extractor
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
    def encode_texts_with_extractor(self, texts: List[str], device: torch.device) -> torch.Tensor:
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

        with open(os.path.join(output_dir, "reconstruction_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "mle_plus_sampled_response_cosine_reinforce",
                    "extractor_name_or_path": self.extractor_name_or_path,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
