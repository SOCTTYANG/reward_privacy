from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Iterable[Dict], path: str) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(data: Dict, path: str) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp(
        min=1e-6
    )


class EmbeddingExtractor:
    def __init__(
        self,
        model_path: str,
        device: str,
        max_length: int,
        torch_dtype: str,
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=resolve_torch_dtype(torch_dtype, device),
        )
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        outputs = self.model(**tokens, return_dict=True)
        if getattr(outputs, "last_hidden_state", None) is not None:
            embeddings = masked_mean(
                outputs.last_hidden_state, tokens["attention_mask"]
            )
        elif getattr(outputs, "pooler_output", None) is not None:
            embeddings = outputs.pooler_output
        else:
            raise ValueError("Embedding extractor output has no usable representation.")
        return F.normalize(embeddings.float(), dim=-1)


def ngrams(tokens: Sequence[str], order: int) -> Counter:
    return Counter(
        tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1)
    )


def corpus_bleu(
    references: Sequence[str],
    predictions: Sequence[str],
    max_order: int,
) -> float:
    matches = [0] * max_order
    possible = [0] * max_order
    reference_length = 0
    prediction_length = 0

    for reference, prediction in zip(references, predictions):
        reference_tokens = reference.split()
        prediction_tokens = prediction.split()
        reference_length += len(reference_tokens)
        prediction_length += len(prediction_tokens)
        for order in range(1, max_order + 1):
            reference_ngrams = ngrams(reference_tokens, order)
            prediction_ngrams = ngrams(prediction_tokens, order)
            matches[order - 1] += sum((reference_ngrams & prediction_ngrams).values())
            possible[order - 1] += max(len(prediction_tokens) - order + 1, 0)

    precisions = [
        match_count / possible_count if possible_count else 0.0
        for match_count, possible_count in zip(matches, possible)
    ]
    if min(precisions) <= 0.0 or prediction_length == 0:
        return 0.0
    brevity_penalty = (
        1.0
        if prediction_length > reference_length
        else math.exp(1.0 - reference_length / prediction_length)
    )
    return brevity_penalty * math.exp(
        sum(math.log(value) for value in precisions) / max_order
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate semantic reconstruction using the paper criterion "
            "cos(E(y_hat_minus), E(y_minus)) >= zeta."
        )
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument(
        "--reference_file",
        required=True,
        help="Held-out triplets containing y_minus; read only by evaluation.",
    )
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--summary_file", required=True)
    parser.add_argument("--reference_field", default="y_minus")
    parser.add_argument("--pred_field", default="lowest_reward_generated_y_minus")
    parser.add_argument("--embedder_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--embed_max_length", type=int, default=256)
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        required=True,
        help="The predefined semantic-reconstruction threshold zeta from the paper.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
    )
    parser.add_argument("--bleu_order", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    if not -1.0 <= args.similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1].")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.bleu_order <= 0:
        raise ValueError("bleu_order must be positive.")

    rows = load_jsonl(args.input_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    reference_rows = load_jsonl(args.reference_file)
    reference_by_id = {
        str(row.get("sample_id", row_index)): row
        for row_index, row in enumerate(reference_rows)
    }

    evaluated_rows: List[Dict] = []
    references: List[str] = []
    predictions: List[str] = []
    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", row_index))
        reference_row = reference_by_id.get(sample_id)
        if reference_row is None:
            continue
        reference = str(reference_row.get(args.reference_field, "")).strip()
        prediction = str(row.get(args.pred_field, "")).strip()
        if not reference:
            continue
        detail_row = dict(row)
        detail_row["reference_y_minus"] = reference
        evaluated_rows.append(detail_row)
        references.append(reference)
        predictions.append(prediction)

    if not evaluated_rows:
        raise ValueError(
            "No selected rows could be matched to a reference y_minus by sample_id."
        )

    extractor = EmbeddingExtractor(
        model_path=args.embedder_path,
        device=args.device,
        max_length=args.embed_max_length,
        torch_dtype=args.torch_dtype,
    )
    cosine_values: List[float] = []
    for start in range(0, len(evaluated_rows), args.batch_size):
        end = start + args.batch_size
        reference_embeddings = extractor.encode(references[start:end])
        prediction_embeddings = extractor.encode(predictions[start:end])
        batch_cosines = F.cosine_similarity(
            prediction_embeddings,
            reference_embeddings,
            dim=-1,
        )
        cosine_values.extend(batch_cosines.detach().cpu().tolist())

    success_count = 0
    for row, cosine_similarity in zip(evaluated_rows, cosine_values):
        reconstructed = cosine_similarity >= args.similarity_threshold
        success_count += int(reconstructed)
        row["reconstruction_cosine_similarity"] = cosine_similarity
        row["similarity_threshold_zeta"] = args.similarity_threshold
        row["semantic_reconstruction_success"] = reconstructed

    mean_cosine = sum(cosine_values) / len(cosine_values)
    summary = {
        "num_evaluated": len(evaluated_rows),
        "bleu_order": args.bleu_order,
        "bleu_score": corpus_bleu(references, predictions, args.bleu_order),
        "cosine_similarity": mean_cosine,
        "similarity_threshold_zeta": args.similarity_threshold,
        "semantic_reconstruction_success_count": success_count,
        "semantic_reconstruction_success_rate": success_count / len(evaluated_rows),
        "embedding_extractor": args.embedder_path,
    }
    save_jsonl(evaluated_rows, args.output_file)
    save_json(summary, args.summary_file)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
