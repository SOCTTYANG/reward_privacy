#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_prompt(row):
    for k in ["prompt", "instruction", "query", "question"]:
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def get_responses(row):
    """
    支持以下常见格式：
    1. chosen / rejected
    2. response_0 / response_1
    3. response
    4. text
    """
    responses = []

    if "chosen" in row and "rejected" in row:
        responses.append(("chosen", str(row["chosen"])))
        responses.append(("rejected", str(row["rejected"])))
    elif "response_0" in row and "response_1" in row:
        responses.append(("response_0", str(row["response_0"])))
        responses.append(("response_1", str(row["response_1"])))
    elif "response" in row:
        responses.append(("response", str(row["response"])))
    elif "text" in row:
        responses.append(("text", str(row["text"])))
    else:
        raise ValueError(f"Cannot find response fields in row keys: {list(row.keys())}")

    return responses


def build_text(prompt, response):
    if prompt:
        return f"Prompt:\n{prompt}\n\nResponse:\n{response}"
    return response


def load_target_rm(base_model_path, adapter_path, dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def load_substitute_rm(model_path, dtype, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def score_texts(tokenizer, model, texts, max_length, batch_size, device):
    scores = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits.view(-1).detach().float().cpu().numpy().tolist()
        scores.extend(logits)

    return scores


def summarize(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "Diff_avg": float(arr.mean()) if len(arr) else None,
        "Diff_var": float(arr.var(ddof=0)) if len(arr) else None,
        "Diff_std": float(arr.std(ddof=0)) if len(arr) else None,
        "Diff_min": float(arr.min()) if len(arr) else None,
        "Diff_max": float(arr.max()) if len(arr) else None,
    }


def summarize_agreement(pair_records):
    """Compute Agreement: teacher and student both rank y+ strictly above y-."""
    if not pair_records:
        return {"Agreement": None, "agreement_count": 0, "agreed_count": 0}

    agreed_count = sum(
        record["target_positive_score"] > record["target_negative_score"]
        and record["substitute_positive_score"] > record["substitute_negative_score"]
        for record in pair_records
    )
    return {
        "Agreement": agreed_count / len(pair_records),
        "agreement_count": len(pair_records),
        "agreed_count": agreed_count,
    }


def evaluate_split(
    split_name,
    data_path,
    sample_size,
    seed,
    target_tokenizer,
    target_model,
    substitute_tokenizer,
    substitute_model,
    max_length,
    target_batch_size,
    substitute_batch_size,
    device,
):
    rows = read_jsonl(data_path)

    rng = random.Random(seed)
    if sample_size is not None and sample_size > 0 and len(rows) > sample_size:
        rows = rng.sample(rows, sample_size)

    item_meta = []
    texts = []

    for idx, row in enumerate(rows):
        prompt = get_prompt(row)
        responses = get_responses(row)

        for resp_name, resp in responses:
            text = build_text(prompt, resp)
            item_meta.append({
                "row_id": idx,
                "response_name": resp_name,
            })
            texts.append(text)

    target_scores = score_texts(
        target_tokenizer,
        target_model,
        texts,
        max_length=max_length,
        batch_size=target_batch_size,
        device=device,
    )

    substitute_scores = score_texts(
        substitute_tokenizer,
        substitute_model,
        texts,
        max_length=max_length,
        batch_size=substitute_batch_size,
        device=device,
    )

    response_records = []
    row_to_diffs = {}

    for meta, ts, ss in zip(item_meta, target_scores, substitute_scores):
        diff = abs(float(ts) - float(ss))

        response_records.append({
            "split": split_name,
            "row_id": meta["row_id"],
            "response_name": meta["response_name"],
            "target_score": float(ts),
            "substitute_score": float(ss),
            "Diff": diff,
        })

        row_to_diffs.setdefault(meta["row_id"], []).append(diff)

    # chosen/rejected has an explicit y+/y- direction. Other row formats are
    # still included in Diff, but cannot contribute to this directional metric.
    scores_by_row = {}
    for record in response_records:
        scores_by_row.setdefault(record["row_id"], {})[record["response_name"]] = record

    agreement_records = []
    for row_scores in scores_by_row.values():
        if "chosen" not in row_scores or "rejected" not in row_scores:
            continue
        positive = row_scores["chosen"]
        negative = row_scores["rejected"]
        agreement_records.append({
            "target_positive_score": positive["target_score"],
            "target_negative_score": negative["target_score"],
            "substitute_positive_score": positive["substitute_score"],
            "substitute_negative_score": negative["substitute_score"],
        })

    # response-level Diff：每个 response 单独算一个 Diff
    response_diffs = [r["Diff"] for r in response_records]

    # item-level Diff：每条原始数据内多个 response 的 Diff 取平均
    item_diffs = [float(np.mean(v)) for v in row_to_diffs.values()]

    result = {
        "split": split_name,
        "data_path": str(data_path),
        "num_original_items": len(rows),
        "num_scored_responses": len(response_records),
        "response_level": summarize(response_diffs),
        "item_level": summarize(item_diffs),
        **summarize_agreement(agreement_records),
    }

    return result, response_records


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_base_model", required=True)
    parser.add_argument("--target_adapter_path", required=True)
    parser.add_argument("--substitute_model_path", required=True)

    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--sample_train", type=int, default=500)
    parser.add_argument("--sample_test", type=int, default=500)

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--target_batch_size", type=int, default=1)
    parser.add_argument("--substitute_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else torch.float32

    print("======================================================")
    print("Target RM base:      ", args.target_base_model)
    print("Target RM adapter:   ", args.target_adapter_path)
    print("Substitute RM:       ", args.substitute_model_path)
    print("Train path:          ", args.train_path)
    print("Test path:           ", args.test_path)
    print("Device:              ", device)
    print("dtype:               ", dtype)
    print("======================================================")

    print("[INFO] Loading target RM...")
    target_tokenizer, target_model = load_target_rm(
        args.target_base_model,
        args.target_adapter_path,
        dtype=dtype,
    )

    print("[INFO] Loading substitute RM...")
    substitute_tokenizer, substitute_model = load_substitute_rm(
        args.substitute_model_path,
        dtype=dtype,
        device=device,
    )

    all_results = []
    all_records = []

    for split_name, path, sample_size in [
        ("train", args.train_path, args.sample_train),
        ("test", args.test_path, args.sample_test),
    ]:
        print(f"======================================================")
        print(f"[INFO] Evaluating split: {split_name}")
        print(f"======================================================")

        result, records = evaluate_split(
            split_name=split_name,
            data_path=path,
            sample_size=sample_size,
            seed=args.seed,
            target_tokenizer=target_tokenizer,
            target_model=target_model,
            substitute_tokenizer=substitute_tokenizer,
            substitute_model=substitute_model,
            max_length=args.max_length,
            target_batch_size=args.target_batch_size,
            substitute_batch_size=args.substitute_batch_size,
            device=device,
        )

        all_results.append(result)
        all_records.extend(records)

        print(json.dumps(result, ensure_ascii=False, indent=2))

    # all split summary
    response_diffs_all = [r["Diff"] for r in all_records]

    row_key_to_diffs = {}
    for r in all_records:
        key = f'{r["split"]}_{r["row_id"]}'
        row_key_to_diffs.setdefault(key, []).append(r["Diff"])
    item_diffs_all = [float(np.mean(v)) for v in row_key_to_diffs.values()]

    agreement_count_all = sum(result["agreement_count"] for result in all_results)
    agreed_count_all = sum(result["agreed_count"] for result in all_results)

    all_summary = {
        "split": "all",
        "num_scored_responses": len(all_records),
        "num_original_items": len(row_key_to_diffs),
        "response_level": summarize(response_diffs_all),
        "item_level": summarize(item_diffs_all),
        "Agreement": (
            agreed_count_all / agreement_count_all if agreement_count_all else None
        ),
        "agreement_count": agreement_count_all,
        "agreed_count": agreed_count_all,
    }

    final_summary = {
        "target_base_model": args.target_base_model,
        "target_adapter_path": args.target_adapter_path,
        "substitute_model_path": args.substitute_model_path,
        "sample_train": args.sample_train,
        "sample_test": args.sample_test,
        "max_length": args.max_length,
        "splits": all_results,
        "all": all_summary,
    }

    summary_path = output_dir / "diff_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    csv_path = output_dir / "diff_records.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "row_id",
                "response_name",
                "target_score",
                "substitute_score",
                "Diff",
            ],
        )
        writer.writeheader()
        writer.writerows(all_records)

    print("======================================================")
    print("[OK] diff summary saved to:", summary_path)
    print("[OK] diff records saved to:", csv_path)
    print("======================================================")
    print(json.dumps(final_summary["all"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
