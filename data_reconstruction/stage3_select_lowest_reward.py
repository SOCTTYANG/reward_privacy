from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = os.environ.get("SAFE_RLHF_PROJECT_ROOT") or os.environ.get("ROOT_DIR")
if not PROJECT_ROOT:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(PROJECT_ROOT)
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from safe_rlhf.dual_reward.dataset import format_prompt_response  # noqa: E402
from safe_rlhf.dual_reward.model import DualRewardModel  # noqa: E402


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict], path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: object) -> str:
    return " ".join(str(text).strip().split())


def build_triplet_index(triplet_rows: List[Dict]) -> Dict[str, Dict]:
    index = {}
    for row in triplet_rows:
        x = normalize_text(row.get("x", ""))
        if x and x not in index:
            index[x] = row
    return index


def build_triplet_sample_id_index(triplet_rows: List[Dict]) -> Dict[str, Dict]:
    index = {}
    for row in triplet_rows:
        sample_id = row.get("sample_id", None)
        if sample_id is None:
            continue
        key = str(sample_id)
        if key and key not in index:
            index[key] = row
    return index


class DualRewardScorer:
    def __init__(
        self,
        reward_model_path: str,
        device: str = "cuda:0",
        max_length: int = 512,
        score_head: str = "help",
        lambda_safe: float = 1.0,
    ) -> None:
        self.device = device
        self.max_length = max_length
        self.score_head = score_head
        self.lambda_safe = lambda_safe

        self.tokenizer = AutoTokenizer.from_pretrained(
            reward_model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = DualRewardModel.from_pretrained_reward(
            reward_model_path=reward_model_path,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> Tuple[float, float, float]:
        text = format_prompt_response(prompt, response)
        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}

        outputs = self.model(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            return_dict=True,
        )

        help_score = float(outputs.help_scores.reshape(-1)[0].detach().cpu().item())
        safe_score = float(outputs.safe_scores.reshape(-1)[0].detach().cpu().item())

        if self.score_head == "help":
            reward_score = help_score
        elif self.score_head == "safe":
            reward_score = safe_score
        elif self.score_head == "sum":
            reward_score = help_score + self.lambda_safe * safe_score
        else:
            raise ValueError(f"Unsupported score_head: {self.score_head}")

        return reward_score, help_score, safe_score


def select_lowest_reward(
    scorer: DualRewardScorer,
    x: str,
    candidate_y_minus_list: List[str],
) -> Tuple[List[float], List[float], List[float], int]:
    """Implement the paper's selection rule exactly.

    For every candidate i, compute r_hat_i = R(x, y_hat_i_minus), then
    j = argmin_{i in {1, ..., K}} r_hat_i.  The caller obtains the final
    reconstruction as y_hat_star_minus = candidate_y_minus_list[j].
    """
    candidate_reward_scores = []
    help_scores = []
    safe_scores = []

    # r_hat_i = R(x, y_hat_i_minus), i = 1, ..., K
    for y_hat_i_minus in candidate_y_minus_list:
        r_hat_i, help_score, safe_score = scorer.score(x, y_hat_i_minus)
        candidate_reward_scores.append(r_hat_i)
        help_scores.append(help_score)
        safe_scores.append(safe_score)

    # j = argmin_{i in {1, ..., K}} R(x, y_hat_i_minus)
    j = min(
        range(len(candidate_y_minus_list)),
        key=lambda i: candidate_reward_scores[i],
    )
    return candidate_reward_scores, help_scores, safe_scores, j


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the final reconstructed y_minus by the lowest reward-model score."
    )
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--triplet_file", type=str, required=True)
    parser.add_argument("--stage2_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--score_head",
        type=str,
        default="help",
        choices=("help", "safe", "sum"),
        help="Reward scalar used for argmin selection.",
    )
    parser.add_argument("--lambda_safe", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 80)
    print("[INFO] Loading dual reward model...")
    scorer = DualRewardScorer(
        reward_model_path=args.reward_model_path,
        device=args.device,
        max_length=args.max_length,
        score_head=args.score_head,
        lambda_safe=args.lambda_safe,
    )
    print("[INFO] Dual reward model loaded.")
    print(f"[INFO] Selection rule: argmin {args.score_head} reward score")
    print("=" * 80)

    triplet_rows = load_jsonl(args.triplet_file)
    stage2_rows = load_jsonl(args.stage2_file)

    if args.max_samples > 0:
        stage2_rows = stage2_rows[: args.max_samples]

    print(f"[INFO] Loaded triplet rows: {len(triplet_rows)}")
    print(f"[INFO] Loaded stage2 rows : {len(stage2_rows)}")

    triplet_index = build_triplet_index(triplet_rows)
    triplet_sample_id_index = build_triplet_sample_id_index(triplet_rows)
    output_rows = []
    skipped = 0

    for idx, row in enumerate(stage2_rows):
        x = normalize_text(row.get("x", ""))
        candidate_y_minus_list = row.get("candidate_y_minus_list", [])

        if not x or not isinstance(candidate_y_minus_list, list) or len(candidate_y_minus_list) == 0:
            skipped += 1
            continue

        gold = None
        sample_id = row.get("sample_id", None)
        if sample_id is not None:
            gold = triplet_sample_id_index.get(str(sample_id))
        if gold is None:
            gold = triplet_index.get(x)

        y_plus = row.get("y_plus", "")
        real_y_minus = row.get("real_y_minus", row.get("y_minus", ""))
        if gold is not None:
            y_plus = y_plus or gold.get("y_plus", "")
            real_y_minus = real_y_minus or gold.get("y_minus", "")

        if not real_y_minus:
            skipped += 1
            continue

        candidate_reward_scores, help_scores, safe_scores, j = select_lowest_reward(
            scorer=scorer,
            x=x,
            candidate_y_minus_list=candidate_y_minus_list,
        )

        # y_hat_star_minus = y_hat_j_minus
        y_hat_star_minus = candidate_y_minus_list[j]
        selected_reward = candidate_reward_scores[j]

        output_rows.append(
            {
                "sample_id": row.get("sample_id", idx),
                "x": x,
                "y_plus": y_plus,
                "real_y_minus": real_y_minus,
                "candidate_y_minus_list": candidate_y_minus_list,
                "candidate_reward_scores": candidate_reward_scores,
                "candidate_help_scores": help_scores,
                "candidate_safe_scores": safe_scores,
                "selection_strategy": f"lowest_{args.score_head}_reward",
                "lowest_reward_index": j,
                "lowest_reward_generated_y_minus": y_hat_star_minus,
                "lowest_reward_score": selected_reward,
                "selected_generated_y_minus": y_hat_star_minus,
                "selected_reward_score": selected_reward,
            }
        )

        if idx < 5:
            print(f"\n[Preview sample {idx}]")
            print("-" * 80)
            print("x:")
            print(x[:300])
            print("-" * 80)
            for i, cand in enumerate(candidate_y_minus_list):
                marker = " <-- selected" if i == j else ""
                print(
                    f"[candidate {i}] reward={candidate_reward_scores[i]:.6f}, "
                    f"help={help_scores[i]:.6f}, safe={safe_scores[i]:.6f}{marker}"
                )
                print(cand[:220])
                print("-" * 80)

        if (idx + 1) % 50 == 0:
            print(f"[INFO] Processed {idx + 1}/{len(stage2_rows)} samples")

    save_jsonl(output_rows, args.output_file)

    print("=" * 80)
    print(f"[INFO] Saved {len(output_rows)} rows to:")
    print(args.output_file)
    print(f"[INFO] Skipped rows: {skipped}")
    print("=" * 80)


if __name__ == "__main__":
    main()
