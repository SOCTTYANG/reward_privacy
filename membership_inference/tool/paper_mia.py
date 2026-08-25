import json
import math
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def load_data(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        if path.lower().endswith(".jsonl"):
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def save_json(value: Any, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def save_artifact(records: List[Dict[str, Any]], metadata: Dict[str, Any], path: str):
    save_json({"schema": "paper_mia_v1", "metadata": metadata, "records": records}, path)


def load_artifact(path: str, expected_stage: int):
    artifact = load_data(path)
    if not isinstance(artifact, dict) or artifact.get("schema") != "paper_mia_v1":
        raise ValueError("Input is not a paper_mia_v1 pipeline artifact")
    metadata = artifact.get("metadata")
    records = artifact.get("records")
    if not isinstance(metadata, dict) or metadata.get("stage") != expected_stage:
        raise ValueError(f"Expected Step {expected_stage} output")
    if not isinstance(records, list):
        raise ValueError("Artifact records must be a list")
    return records, metadata


def first(record: Dict[str, Any], names: Iterable[str]):
    return next((record[name] for name in names if name in record and record[name] is not None), None)


def normalize_triplet(record: Dict[str, Any]) -> Tuple[str, str, str]:
    x = first(record, ("x", "X", "prompt", "instruction", "question", "query"))
    y_plus = first(record, ("y_plus", "Y_plus", "Y+", "chosen", "better", "preferred"))
    y_minus = first(record, ("y_minus", "Y_minus", "Y-", "rejected", "worse", "dispreferred"))
    if (y_plus is None or y_minus is None) and all(
        key in record for key in ("response_0", "response_1", "better_response_id")
    ):
        better = int(record["better_response_id"])
        if better not in (0, 1):
            raise ValueError("better_response_id must be 0 or 1")
        y_plus = record[f"response_{better}"]
        y_minus = record[f"response_{1 - better}"]
    if x is None or y_plus is None or y_minus is None:
        raise ValueError(f"Missing triplet in keys {list(record)}")
    return str(x), str(y_plus), str(y_minus)


def reward_text(tokenizer, prompt: str, response: str) -> str:
    messages = [
        {"role": "user", "content": str(prompt).strip()},
        {"role": "assistant", "content": str(response).strip()},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"### Human:\n{messages[0]['content']}\n\n### Assistant:\n{messages[1]['content']}"


def policy_prompt_text(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": str(prompt).strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return str(prompt).strip()


@torch.no_grad()
def score_reward_pairs(model, tokenizer, pairs, batch_size, max_length, device):
    scores = []
    model.eval()
    for start in range(0, len(pairs), batch_size):
        texts = [reward_text(tokenizer, x, y) for x, y in pairs[start : start + batch_size]]
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
        if logits.shape[-1] != 1:
            raise ValueError("Reward model must output one scalar R(x,y)")
        scores.extend(logits.squeeze(-1).detach().float().cpu().tolist())
    return scores


def advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a non-empty vector")
    return rewards - rewards.mean()


def encode_policy_tokens(tokenizer, prompt_ids, response_token_ids, max_length, device):
    prompt_ids = [int(token) for token in prompt_ids]
    if len(prompt_ids) >= max_length:
        raise ValueError("Policy prompt leaves no room for a response")
    rows = []
    for response in response_token_ids:
        response = [int(token) for token in response][: max_length - len(prompt_ids)]
        if not response:
            raise ValueError("Candidate response has no policy tokens")
        rows.append((prompt_ids + response, [-100] * len(prompt_ids) + response))
    width = max(len(ids) for ids, _ in rows)
    input_ids = []
    attention_masks = []
    labels = []
    for ids, row_labels in rows:
        padding = width - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * padding)
        attention_masks.append([1] * len(ids) + [0] * padding)
        labels.append(row_labels + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def sequence_logprob(model, input_ids, attention_mask, labels):
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1]
    target = labels[:, 1:]
    mask = target.ne(-100)
    safe_target = target.masked_fill(~mask, 0)
    token_logprob = F.log_softmax(logits, dim=-1).gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    return (token_logprob * mask).sum(dim=-1)


def ppo_clipped_loss(current_logprob, reference_logprob, advantage, epsilon):
    rho = torch.exp(current_logprob - reference_logprob)
    clipped_rho = torch.clamp(rho, 1.0 - epsilon, 1.0 + epsilon)
    return -torch.minimum(rho * advantage, clipped_rho * advantage).mean()


def gradient_l2(parameters):
    squared_norm = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.requires_grad and parameter.grad is not None:
            squared_norm += parameter.grad.detach().double().pow(2).sum().cpu()
    return math.sqrt(squared_norm.item())


def reward_margin(r_plus, r_minus):
    return r_plus - r_minus


def pairwise_ranking_loss_from_margin(margin):
    return -F.logsigmoid(margin)


def reward_margin_loss_derivative(margin):
    return torch.sigmoid(margin) - 1.0


def reward_parameter_after_step(parameters, margin, margin_gradient, learning_rate):
    return parameters + learning_rate * torch.sigmoid(-margin) * margin_gradient


def first_order_margin_after_update(margin, learning_rate, margin_gradient_l2):
    return margin + learning_rate * torch.sigmoid(-margin) * margin_gradient_l2**2


def local_policy_gradient(advantage, score_gradients):
    shape = (advantage.shape[0],) + (1,) * (score_gradients.ndim - 1)
    return -(advantage.reshape(shape) * score_gradients).mean(dim=0)


def policy_gradient_cauchy_bound(rewards, score_gradient_l2):
    centered = advantages(rewards)
    return torch.sqrt(centered.pow(2).mean()) * torch.sqrt(score_gradient_l2.pow(2).mean())
