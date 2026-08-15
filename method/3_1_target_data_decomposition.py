import os
import json
import random
import argparse
from typing import Dict, Any, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    return data


def get_first_existing(example: Dict[str, Any], keys: List[str]):
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return None


def normalize_triplet(example: Dict[str, Any]) -> Tuple[str, str, str]:

    x = get_first_existing(
        example,
        [
            "x",
            "X",
            "prompt",
            "instruction",
            "question",
            "query",
        ],
    )

    y_plus = get_first_existing(
        example,
        [
            "y_plus",
            "Y_plus",
            "Y+",
            "chosen",
            "better",
            "preferred",
            "preferred_response",
        ],
    )

    y_minus = get_first_existing(
        example,
        [
            "y_minus",
            "Y_minus",
            "Y-",
            "rejected",
            "worse",
            "dispreferred",
            "rejected_response",
            "dispreferred_response",
        ],
    )


    if y_plus is None or y_minus is None:
        if (
            "response_0" in example
            and "response_1" in example
            and "better_response_id" in example
        ):
            better_id = int(example["better_response_id"])

            if better_id == 0:
                y_plus = example["response_0"]
                y_minus = example["response_1"]
            elif better_id == 1:
                y_plus = example["response_1"]
                y_minus = example["response_0"]
            else:
                raise ValueError(f"Invalid better_response_id: {better_id}")

    if x is None or y_plus is None or y_minus is None:
        raise ValueError(f"Cannot normalize example keys: {list(example.keys())}")

    return str(x), str(y_plus), str(y_minus)


def load_and_sample(
    path: str,
    dataset_name: str,
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    raw_data = read_jsonl(path)

    normalized_data = []
    skipped = 0

    for idx, example in enumerate(raw_data):
        try:
            x, y_plus, y_minus = normalize_triplet(example)

            normalized_data.append(
                {
                    "source_dataset": dataset_name,
                    "source_index": idx,
                    "x": x,
                    "y_plus": y_plus,
                    "y_minus": y_minus,
                }
            )

        except Exception:
            skipped += 1

    if len(normalized_data) < sample_size:
        raise ValueError(
            f"{dataset_name} 可用数据不足：需要 {sample_size} 条，但只解析出 {len(normalized_data)} 条。"
        )

    rng = random.Random(seed)
    sampled_data = rng.sample(normalized_data, sample_size)

    print(f"\n[INFO] Dataset: {dataset_name}")
    print(f"[INFO] Path: {path}")
    print(f"[INFO] Raw samples: {len(raw_data)}")
    print(f"[INFO] Valid triplets: {len(normalized_data)}")
    print(f"[INFO] Skipped samples: {skipped}")
    print(f"[INFO] Sampled samples: {len(sampled_data)}")

    return sampled_data


@torch.no_grad()
def score_pairs(
    model,
    tokenizer,
    pairs: List[Tuple[str, str]],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> List[float]:


    scores = []
    model.eval()

    for start in tqdm(range(0, len(pairs), batch_size), desc="Scoring pairs"):
        batch_pairs = pairs[start:start + batch_size]

        prompts = [item[0] for item in batch_pairs]
        responses = [item[1] for item in batch_pairs]

        inputs = tokenizer(
            prompts,
            responses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits


        if logits.shape[-1] == 1:
            batch_scores = logits.squeeze(-1)


        elif logits.shape[-1] == 2:
            batch_scores = logits[:, 1]


        else:
            batch_scores = logits.max(dim=-1).values

        scores.extend(batch_scores.detach().cpu().float().tolist())

    return scores


def save_json(data: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Saved to: {output_path}")
    print(f"[INFO] Total decomposed records: {len(data)}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target_rm_path",
        type=str,
        default="/mnt/bai_data/projects/bai-rm-extraction-exp/output/target_rm_roberta",
    )

    parser.add_argument(
        "--hh_path",
        type=str,
        default="/run/media/vipuser/data/yang-safe/data/hh_rlhf_triplet_train.jsonl",
    )

    parser.add_argument(
        "--pku_path",
        type=str,
        default="/mnt/bai_data/yang-safe/data/pku30k_filtered/pku30k_train_cosine_le_085_xyz.jsonl",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/bai_data/yang-safe/membership inference/3.1-Target Data Decomposition",
    )

    parser.add_argument(
        "--output_filename",
        type=str,
        default="3.1-Target Data Decomposition.json",
    )

    parser.add_argument("--sample_per_dataset", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()

    output_path = os.path.join(args.output_dir, args.output_filename)

    print("[INFO] Step 1: Target Data Decomposition")
    print(f"[INFO] Target reward model: {args.target_rm_path}")
    print(f"[INFO] HH dataset: {args.hh_path}")
    print(f"[INFO] PKU dataset: {args.pku_path}")
    print(f"[INFO] Output path: {output_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    print("\n[INFO] Loading target reward model...")

    tokenizer = AutoTokenizer.from_pretrained(args.target_rm_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.target_rm_path
    )

    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    hh_samples = load_and_sample(
        path=args.hh_path,
        dataset_name="hh_rlhf_triplet_train",
        sample_size=args.sample_per_dataset,
        seed=args.seed,
    )

    pku_samples = load_and_sample(
        path=args.pku_path,
        dataset_name="pku30k_train_cosine_le_085_xyz",
        sample_size=args.sample_per_dataset,
        seed=args.seed + 1,
    )

    all_samples = hh_samples + pku_samples

    print(f"\n[INFO] Total target data points d=(x,y+,y-): {len(all_samples)}")
    print(f"[INFO] Total prompt-response pairs to query: {len(all_samples) * 2}")


    plus_pairs = [(item["x"], item["y_plus"]) for item in all_samples]
    minus_pairs = [(item["x"], item["y_minus"]) for item in all_samples]

    print("\n[INFO] Scoring positive pairs: r_plus = R(x, y_plus)")
    r_plus_scores = score_pairs(
        model=model,
        tokenizer=tokenizer,
        pairs=plus_pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    print("\n[INFO] Scoring negative pairs: r_minus = R(x, y_minus)")
    r_minus_scores = score_pairs(
        model=model,
        tokenizer=tokenizer,
        pairs=minus_pairs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    results = []

    for idx, item in enumerate(all_samples):
        r_plus = float(r_plus_scores[idx])
        r_minus = float(r_minus_scores[idx])

        results.append(
            {
                "id": idx,
                "source_dataset": item["source_dataset"],
                "source_index": item["source_index"],

                "x": item["x"],
                "y_plus": item["y_plus"],
                "y_minus": item["y_minus"],

                "r_plus": r_plus,
                "r_minus": r_minus,
                "reward_gap": r_plus - r_minus,

                "decomposed_pairs": {
                    "positive_pair": {
                        "x": item["x"],
                        "y": item["y_plus"],
                        "reward_score": r_plus,
                    },
                    "negative_pair": {
                        "x": item["x"],
                        "y": item["y_minus"],
                        "reward_score": r_minus,
                    },
                },
            }
        )

    save_json(results, output_path)

    print("\n[DONE] Step 1 finished successfully.")
    print(f"[DONE] Output file: {output_path}")


if __name__ == "__main__":
    main()
