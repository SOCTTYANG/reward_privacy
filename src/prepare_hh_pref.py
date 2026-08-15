from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_hh_text(text: str) -> Tuple[str, str]:
    """
    HH-RLHF 的 chosen/rejected 通常是完整对话文本：
      Human: ...
      Assistant: ...

    我们按最后一次 '\n\nAssistant:' 切分：
      prompt   = 最后一个 Assistant 回复之前的上下文
      response = 最后一个 Assistant 的回复
    """
    marker = "\n\nAssistant:"

    if marker not in text:
        # 兜底：无法拆分时，把全文作为 response，prompt 为空
        return "", text.strip()

    prompt, response = text.rsplit(marker, 1)

    prompt = prompt.strip()
    response = response.strip()

    # 为了保持 prompt 语义完整，把最后的 Assistant: 标记放回 prompt 末尾
    # 这样模型输入会是：
    # ### Prompt:
    # Human: ...
    # Assistant:
    #
    # ### Response:
    # ...
    prompt = prompt + "\n\nAssistant:"

    return prompt, response


def convert_one(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    if "chosen" not in example or "rejected" not in example:
        raise KeyError(f"HH example must contain chosen/rejected. Keys: {list(example.keys())}")

    chosen_text = str(example["chosen"])
    rejected_text = str(example["rejected"])

    prompt_c, chosen_resp = split_hh_text(chosen_text)
    prompt_r, rejected_resp = split_hh_text(rejected_text)

    # 正常情况下 chosen/rejected 的 prompt 应该相同。
    # 如果不同，优先使用 chosen 的 prompt；这类样本不会直接丢弃，避免损失太多数据。
    prompt = prompt_c if prompt_c.strip() else prompt_r

    if not prompt.strip():
        return None
    if not chosen_resp.strip() or not rejected_resp.strip():
        return None
    if chosen_resp.strip() == rejected_resp.strip():
        return None

    return {
        "prompt": prompt,
        "chosen": chosen_resp,
        "rejected": rejected_resp,
    }


def load_hh_split(dataset_name: str, data_dirs: List[str], split: str):
    """
    HH-RLHF 有多个子目录：
      - helpful-base
      - helpful-online
      - helpful-rejection-sampled
      - harmless-base

    这里支持把多个 data_dir 合并。
    """
    parts = []

    for data_dir in data_dirs:
        print(f"[INFO] Loading {dataset_name}, data_dir={data_dir}, split={split}")
        ds = load_dataset(dataset_name, data_dir=data_dir, split=split)
        parts.append(ds)

    if len(parts) == 1:
        return parts[0]

    return concatenate_datasets(parts)


def convert_split(
    raw_dataset,
    max_samples: Optional[int],
) -> List[Dict[str, str]]:
    rows = []

    for ex in tqdm(raw_dataset, desc="Converting HH"):
        item = convert_one(ex)

        if item is None:
            continue

        rows.append(item)

        if max_samples is not None and len(rows) >= max_samples:
            break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        default=["helpful-base"],
        help=(
            "HH-RLHF subdirectories. Recommended for Exp1: helpful-base. "
            "Optional: helpful-base helpful-online helpful-rejection-sampled"
        ),
    )

    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--max_test_samples", type=int, default=1000)

    args = parser.parse_args()

    train_raw = load_hh_split(
        dataset_name=args.dataset_name,
        data_dirs=args.data_dirs,
        split="train",
    )

    test_raw = load_hh_split(
        dataset_name=args.dataset_name,
        data_dirs=args.data_dirs,
        split="test",
    )

    print("[INFO] train columns:", train_raw.column_names)
    print("[INFO] test columns: ", test_raw.column_names)

    train_rows = convert_split(train_raw, args.max_train_samples)
    test_rows = convert_split(test_raw, args.max_test_samples)

    train_path = os.path.join(args.output_dir, "hh_pref_train.jsonl")
    test_path = os.path.join(args.output_dir, "hh_pref_test.jsonl")

    save_jsonl(train_rows, train_path)
    save_jsonl(test_rows, test_path)

    print(f"[OK] saved HH train: {train_path}, n={len(train_rows)}")
    print(f"[OK] saved HH test:  {test_path}, n={len(test_rows)}")

    if train_rows:
        print("[INFO] first converted example:")
        print(json.dumps(train_rows[0], ensure_ascii=False, indent=2)[:1500])


if __name__ == "__main__":
    main()