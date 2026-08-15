import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()

    if not text:
        return []

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data

    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_no}")
        rows.append(obj)
    return rows


def get_field(obj: Dict[str, Any], names: Iterable[str], default: Optional[Any] = None) -> Optional[Any]:
    for name in names:
        if name in obj and obj[name] is not None:
            return obj[name]
    return default


def normalize_for_icp(obj: Dict[str, Any], response_field: str) -> Dict[str, str]:
    x = get_field(obj, ["x", "X", "prompt", "instruction", "question"])
    y_plus = get_field(obj, ["y_plus", "Y+", "chosen", "response_chosen", "output", "response"])
    y_minus = get_field(obj, ["y_minus", "Y-", "rejected", "response_rejected"])

    if x is None:
        raise ValueError(f"Missing prompt field in item keys={list(obj.keys())}")

    if response_field == "y_plus":
        response = y_plus
    elif response_field == "y_minus":
        response = y_minus
    elif response_field == "output":
        response = get_field(obj, ["output", "response", "chosen", "y_plus", "Y+"])
    else:
        raise ValueError(f"Unsupported response_field: {response_field}")

    if response is None:
        raise ValueError(f"Missing response field `{response_field}` in item keys={list(obj.keys())}")

    return {
        "instruction": str(x),
        "input": "",
        "output": str(response),
    }


def select_rows(rows: List[Dict[str, Any]], size: int, mode: str, seed: int, name: str) -> List[Dict[str, Any]]:
    if len(rows) < size:
        raise ValueError(f"{name} has only {len(rows)} rows, but {size} were requested")

    if mode == "first":
        return rows[:size]

    if mode == "random":
        rng = random.Random(seed)
        return rng.sample(rows, size)

    raise ValueError(f"Unsupported sample mode: {mode}")


def write_json(data: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_labeled_rows(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    members = []
    nonmembers = []

    for row in rows:
        membership = str(row.get("membership", "")).lower()
        label = row.get("label")

        if membership == "member" or label == 1:
            members.append(row)
        elif membership in {"nonmember", "non-member"} or label == 0:
            nonmembers.append(row)
        else:
            raise ValueError(f"Cannot infer membership for row keys={list(row.keys())}")

    return members, nonmembers


def build_icp_rows(rows: List[Dict[str, Any]], label: int, response_field: str) -> List[Dict[str, Any]]:
    membership = "member" if label == 1 else "nonmember"
    output_rows = []

    for fallback_index, row in enumerate(rows):
        item = normalize_for_icp(row, response_field)
        item.update(
            {
                "label": label,
                "membership": membership,
                "source_dataset": row.get("source_dataset"),
                "source_index": row.get("source_index", fallback_index),
                "source_id": row.get("id"),
            }
        )
        output_rows.append(item)

    return output_rows


def build_perturbation_rows(rows: List[Dict[str, Any]], response_field: str) -> List[Dict[str, Any]]:
    perturbation_rows = []

    for row in rows:
        membership = str(row.get("membership", "")).lower()
        label = 1 if membership == "member" or row.get("label") == 1 else 0
        target_example = normalize_for_icp(row, response_field)
        perturbations = []

        for candidate in row.get("candidate_responses", []):
            y_i = str(candidate.get("y_i", "")).strip()
            if y_i:
                perturbations.append(y_i)

        perturbation_rows.append(
            {
                "target_example": target_example,
                "mask_perturbations": perturbations,
                "label": label,
                "membership": "member" if label == 1 else "nonmember",
                "source_dataset": row.get("source_dataset"),
                "source_index": row.get("source_index"),
                "source_id": row.get("id"),
            }
        )

    return perturbation_rows


def convert(args: argparse.Namespace) -> None:
    source_rows = None
    if args.step2_path:
        source_rows = read_json_or_jsonl(args.step2_path)
        member_rows, nonmember_rows = split_labeled_rows(source_rows)
    else:
        if not args.member_path or not args.nonmember_path:
            raise ValueError("Either --step2_path or both --member_path/--nonmember_path are required")
        member_rows = read_json_or_jsonl(args.member_path)
        nonmember_rows = read_json_or_jsonl(args.nonmember_path)

    members = select_rows(member_rows, args.member_size, args.member_sample_mode, args.seed, "member")
    nonmembers = select_rows(
        nonmember_rows,
        args.nonmember_size,
        args.nonmember_sample_mode,
        args.seed,
        "nonmember",
    )

    train_data = build_icp_rows(members, 1, args.response_field)
    test_data = build_icp_rows(nonmembers, 0, args.response_field)
    attack_data = train_data + test_data

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, f"{args.file_prefix}_train.json")
    test_path = os.path.join(args.output_dir, f"{args.file_prefix}_test.json")
    attack_path = os.path.join(args.output_dir, f"{args.file_prefix}_attack.json")
    perturbation_path = os.path.join(args.output_dir, f"{args.file_prefix}_perturbations.json")

    write_json(train_data, train_path)
    write_json(test_data, test_path)
    write_json(attack_data, attack_path)
    if source_rows is not None:
        selected_ids = {row.get("id") for row in members + nonmembers}
        selected_rows = [row for row in source_rows if row.get("id") in selected_ids]
        if len(selected_rows) != len(members) + len(nonmembers):
            selected_rows = members + nonmembers
        write_json(build_perturbation_rows(selected_rows, args.response_field), perturbation_path)

    print("[INFO] ICP-MIA data prepared")
    if args.step2_path:
        print(f"[INFO] step2 input     : {args.step2_path}")
    else:
        print(f"[INFO] member input    : {args.member_path}")
        print(f"[INFO] nonmember input : {args.nonmember_path}")
    print(f"[INFO] train JSON      : {train_path}")
    print(f"[INFO] test JSON       : {test_path}")
    print(f"[INFO] attack JSON     : {attack_path}")
    if source_rows is not None:
        print(f"[INFO] perturbations   : {perturbation_path}")
    print(f"[INFO] members/nonmembers: {len(train_data)}/{len(test_data)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Safe-RLHF MIA jsonl files to ICP-MIA JSON files")
    parser.add_argument("--step2_path", default=None, help="3.2_candidate_response_generation.json from membership inference")
    parser.add_argument("--member_path", default=None)
    parser.add_argument("--nonmember_path", default=None)
    parser.add_argument("--output_dir", default="./data/safe_rlhf_mia")
    parser.add_argument("--file_prefix", default="safe_rlhf_mia")
    parser.add_argument("--member_size", type=int, default=512)
    parser.add_argument("--nonmember_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--member_sample_mode", choices=["first", "random"], default="first")
    parser.add_argument("--nonmember_sample_mode", choices=["first", "random"], default="random")
    parser.add_argument("--response_field", choices=["y_plus", "y_minus", "output"], default="y_plus")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
