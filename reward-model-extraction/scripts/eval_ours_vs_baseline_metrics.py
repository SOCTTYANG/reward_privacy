#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from src.extraction_metrics import compute_positive_pair_agreement


def read_jsonl(path, max_samples=None, seed=42):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if max_samples is not None and max_samples > 0 and len(rows) > max_samples:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_samples)

    return rows


def get_prompt(row):
    for k in ["prompt", "instruction", "query", "question"]:
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def get_pair(row):
    prompt = get_prompt(row)

    if "chosen" in row and "rejected" in row:
        return prompt, str(row["chosen"]), str(row["rejected"]), 0

    if "response_0" in row and "response_1" in row:
        r0 = str(row["response_0"])
        r1 = str(row["response_1"])

        gold_idx = None
        for key in ["better_response_id", "safer_response_id", "preferred_response_id", "label"]:
            if key in row:
                try:
                    gold_idx = int(row[key])
                    if gold_idx not in [0, 1]:
                        gold_idx = None
                except Exception:
                    gold_idx = None
                break

        return prompt, r0, r1, gold_idx

    raise ValueError(f"Cannot parse pair fields. keys={list(row.keys())}")


def get_aux_texts(row):
    prompt = get_prompt(row)
    responses = []

    if "response" in row:
        responses.append(str(row["response"]))
    elif "output" in row:
        responses.append(str(row["output"]))
    elif "answer" in row:
        responses.append(str(row["answer"]))
    elif "text" in row:
        responses.append(str(row["text"]))
    elif "chosen" in row and "rejected" in row:
        responses.append(str(row["chosen"]))
        responses.append(str(row["rejected"]))
    elif "response_0" in row and "response_1" in row:
        responses.append(str(row["response_0"]))
        responses.append(str(row["response_1"]))
    else:
        raise ValueError(f"Cannot parse aux response fields. keys={list(row.keys())}")

    return [build_text(prompt, r) for r in responses]


def build_text(prompt, response):
    if prompt:
        return f"Prompt:\n{prompt}\n\nResponse:\n{response}"
    return response


def load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(
        path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_target_rm(base_model_path, adapter_path, dtype):
    tok = load_tokenizer(base_model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return tok, model


def load_substitute_rm(model_path, tokenizer_path, dtype, device):
    tok_path = tokenizer_path if tokenizer_path else model_path
    tok = load_tokenizer(tok_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.to(device)
    model.eval()

    return tok, model


@torch.no_grad()
def score_texts(model, tokenizer, texts, max_length, batch_size, device):
    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        batch_scores = outputs.logits.view(-1).detach().float().cpu().numpy().tolist()
        scores.extend(batch_scores)

    return scores


def pearson_corr(x, y):
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    if len(x) < 2:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None

    return float(np.corrcoef(x, y)[0, 1])


def eval_pairwise(
    split_name,
    data_path,
    target_model,
    target_tok,
    sub_model,
    sub_tok,
    max_samples,
    max_length,
    target_batch_size,
    sub_batch_size,
    device,
    seed,
):
    rows = read_jsonl(data_path, max_samples=max_samples, seed=seed)

    texts = []
    golds = []

    for row in rows:
        prompt, r0, r1, gold_idx = get_pair(row)
        texts.append(build_text(prompt, r0))
        texts.append(build_text(prompt, r1))
        golds.append(gold_idx)

    target_scores_flat = score_texts(
        target_model,
        target_tok,
        texts,
        max_length=max_length,
        batch_size=target_batch_size,
        device=device,
    )
    sub_scores_flat = score_texts(
        sub_model,
        sub_tok,
        texts,
        max_length=max_length,
        batch_size=sub_batch_size,
        device=device,
    )

    target_scores = np.array(target_scores_flat, dtype=np.float64).reshape(-1, 2)
    sub_scores = np.array(sub_scores_flat, dtype=np.float64).reshape(-1, 2)

    target_pref = target_scores.argmax(axis=1)
    sub_pref = sub_scores.argmax(axis=1)

    agreement_acc = float(np.mean(target_pref == sub_pref))

    # Agreement follows the extraction metric definition: both teacher and
    # student must rank the labelled y+ strictly above y-. Unlabelled pairs do
    # not have a known y+ direction and are therefore excluded.
    valid_gold = [i for i, g in enumerate(golds) if g in [0, 1]]
    agreement = compute_positive_pair_agreement(
        target_scores[valid_gold],
        sub_scores[valid_gold],
        [golds[i] for i in valid_gold],
    )

    if valid_gold:
        gold_arr = np.array([golds[i] for i in valid_gold])
        target_gold_acc = float(np.mean(target_pref[valid_gold] == gold_arr))
        sub_gold_acc = float(np.mean(sub_pref[valid_gold] == gold_arr))
    else:
        target_gold_acc = None
        sub_gold_acc = None

    item_diffs = np.abs(target_scores - sub_scores).mean(axis=1)

    result = {
        "split": split_name,
        "data_path": data_path,
        "num_items": int(len(rows)),
        "target_gold_acc": target_gold_acc,
        "substitute_gold_acc": sub_gold_acc,
        "target_student_agreement_acc": agreement_acc,
        "Agreement": agreement,
        "agreement_count": len(valid_gold),
        "Diff_avg": float(item_diffs.mean()),
        "Diff_var": float(item_diffs.var()),
        "Diff_std": float(item_diffs.std()),
    }

    return result


def eval_aux_pearson(
    aux_path,
    target_model,
    target_tok,
    sub_model,
    sub_tok,
    max_samples,
    max_length,
    target_batch_size,
    sub_batch_size,
    device,
    seed,
):
    rows = read_jsonl(aux_path, max_samples=max_samples, seed=seed)

    texts = []
    for row in rows:
        texts.extend(get_aux_texts(row))

    target_scores = score_texts(
        target_model,
        target_tok,
        texts,
        max_length=max_length,
        batch_size=target_batch_size,
        device=device,
    )
    sub_scores = score_texts(
        sub_model,
        sub_tok,
        texts,
        max_length=max_length,
        batch_size=sub_batch_size,
        device=device,
    )

    diffs = np.abs(np.array(target_scores, dtype=np.float64) - np.array(sub_scores, dtype=np.float64))

    result = {
        "aux_path": aux_path,
        "num_texts": int(len(texts)),
        "pearson": pearson_corr(target_scores, sub_scores),
        "Diff_avg": float(diffs.mean()),
        "Diff_var": float(diffs.var()),
        "Diff_std": float(diffs.std()),
    }

    return result


def evaluate_method(
    method_name,
    sub_path,
    args,
    target_model,
    target_tok,
    dtype,
    device,
):
    print("=" * 80)
    print(f"[INFO] Evaluating method: {method_name}")
    print(f"[INFO] substitute path: {sub_path}")
    print("=" * 80)

    sub_tok, sub_model = load_substitute_rm(
        sub_path,
        tokenizer_path=args.student_tokenizer_path,
        dtype=dtype,
        device=device,
    )

    pku_result = eval_pairwise(
        split_name="PKU",
        data_path=args.pku_path,
        target_model=target_model,
        target_tok=target_tok,
        sub_model=sub_model,
        sub_tok=sub_tok,
        max_samples=args.max_pairwise_samples,
        max_length=args.max_length,
        target_batch_size=args.target_batch_size,
        sub_batch_size=args.substitute_batch_size,
        device=device,
        seed=args.seed,
    )

    hh_result = eval_pairwise(
        split_name="HH",
        data_path=args.hh_path,
        target_model=target_model,
        target_tok=target_tok,
        sub_model=sub_model,
        sub_tok=sub_tok,
        max_samples=args.max_pairwise_samples,
        max_length=args.max_length,
        target_batch_size=args.target_batch_size,
        sub_batch_size=args.substitute_batch_size,
        device=device,
        seed=args.seed,
    )

    aux_result = eval_aux_pearson(
        aux_path=args.aux_path,
        target_model=target_model,
        target_tok=target_tok,
        sub_model=sub_model,
        sub_tok=sub_tok,
        max_samples=args.max_aux_samples,
        max_length=args.max_length,
        target_batch_size=args.target_batch_size,
        sub_batch_size=args.substitute_batch_size,
        device=device,
        seed=args.seed,
    )

    result = {
        "method": method_name,
        "substitute_path": sub_path,
        "PKU": pku_result,
        "HH": hh_result,
        "AUX": aux_result,
    }

    del sub_model
    torch.cuda.empty_cache()

    return result


def print_table(results):
    print("\n")
    print("| Method | PKU Acc | PKU Agree | HH Acc | HH Agree | Dolly Pearson | PKU Diff_avg | PKU Diff_var |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")

    for r in results:
        method = r["method"]
        pku = r["PKU"]
        hh = r["HH"]
        aux = r["AUX"]

        print(
            f"| {method} | "
            f"{pku['substitute_gold_acc']:.6f} | "
            f"{pku['target_student_agreement_acc']:.6f} | "
            f"{hh['substitute_gold_acc']:.6f} | "
            f"{hh['target_student_agreement_acc']:.6f} | "
            f"{aux['pearson']:.6f} | "
            f"{pku['Diff_avg']:.6f} | "
            f"{pku['Diff_var']:.6f} |"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_name", required=True)
    parser.add_argument("--target_base_model", required=True)
    parser.add_argument("--target_adapter_path", required=True)

    parser.add_argument("--ours_substitute_path", required=True)
    parser.add_argument("--baseline_substitute_path", required=True)
    parser.add_argument("--student_tokenizer_path", default="models/roberta-base")

    parser.add_argument("--pku_path", default="data/test.jsonl")
    parser.add_argument("--hh_path", default="data/hh_pref_test.jsonl")
    parser.add_argument("--aux_path", default="data/aux_dolly.jsonl")

    parser.add_argument("--max_pairwise_samples", type=int, default=1000)
    parser.add_argument("--max_aux_samples", type=int, default=5000)

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--target_batch_size", type=int, default=1)
    parser.add_argument("--substitute_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else torch.float32

    print("=" * 80)
    print("[INFO] Target:", args.target_name)
    print("[INFO] Target base:", args.target_base_model)
    print("[INFO] Target adapter:", args.target_adapter_path)
    print("[INFO] Device:", device)
    print("[INFO] dtype:", dtype)
    print("=" * 80)

    target_tok, target_model = load_target_rm(
        args.target_base_model,
        args.target_adapter_path,
        dtype=dtype,
    )

    results = []

    results.append(
        evaluate_method(
            "Ours",
            args.ours_substitute_path,
            args,
            target_model,
            target_tok,
            dtype,
            device,
        )
    )

    results.append(
        evaluate_method(
            "MiniLLM-RM",
            args.baseline_substitute_path,
            args,
            target_model,
            target_tok,
            dtype,
            device,
        )
    )

    final = {
        "target_name": args.target_name,
        "target_base_model": args.target_base_model,
        "target_adapter_path": args.target_adapter_path,
        "results": results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "ours_vs_baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print_table(results)
    print("\n[OK] saved to:", out_path)


if __name__ == "__main__":
    main()
