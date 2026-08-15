from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer


class ReconstructionTrainer(Trainer):
    """MLE plus the paper's sampled-response cosine objective.

    The forward value is exactly ``-cos(E(y_hat), E(y_minus))``. Because
    sampling discrete tokens is non-differentiable, its gradient is estimated
    with REINFORCE so the cosine term can update the generator.
    """

    def __init__(
        self,
        *args,
        lambda_mle: float = 1.0,
        lambda_cos: float = 0.05,
        generation_tokenizer=None,
        cosine_max_new_tokens: int = 96,
        cosine_temperature: float = 1.0,
        cosine_top_p: float = 0.95,
        reward_baseline_momentum: float = 0.9,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if generation_tokenizer is None:
            raise ValueError("generation_tokenizer is required for sampled cosine loss.")
        self.lambda_mle = lambda_mle
        self.lambda_cos = lambda_cos
        self.generation_tokenizer = generation_tokenizer
        self.cosine_max_new_tokens = cosine_max_new_tokens
        self.cosine_temperature = cosine_temperature
        self.cosine_top_p = cosine_top_p
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
        sequences = core_model.lm.generate(
            **source_batch,
            do_sample=True,
            temperature=self.cosine_temperature,
            top_p=self.cosine_top_p,
            max_new_tokens=self.cosine_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
        )
        if was_training:
            core_model.lm.train()

        generated_ids = sequences[:, prompt_width:]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_texts = [text.strip() for text in generated_texts]
        return generated_texts, sequences, source_batch["attention_mask"]

    def _sample_log_prob(
        self,
        model,
        sequences: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log p_theta(y_hat | x, y_plus) for sampled responses."""
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

    def compute_loss(self, model, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs):
        source_texts = inputs.pop("source_texts")
        target_texts = inputs.pop("target_texts")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        l_mle = outputs.loss

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

            # Preserve the exact paper loss value while supplying its unbiased
            # score-function gradient estimator through the zero-valued term.
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
