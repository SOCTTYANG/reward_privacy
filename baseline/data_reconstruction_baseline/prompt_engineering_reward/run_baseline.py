from __future__ import annotations

import gc
import json
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import set_seed

from baseline_components import (
    DualRewardScorer,
    PromptEngineeringGenerator,
    build_generation_prompt,
    build_summary,
    first_nonempty,
    load_jsonl,
    parse_args,
    save_json,
)


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def make_empty_state(idx: int, row: Dict[str, Any], args) -> Optional[Dict[str, Any]]:
    x = first_nonempty(row, ("x", "prompt", "question"))
    y_plus = first_nonempty(row, ("y_plus", "chosen", "preferred", "help_chosen"))
    real_y_minus = first_nonempty(row, ("y_minus", "real_y_minus", "rejected", "dispreferred", "help_rejected"))

    if not x or not y_plus or not real_y_minus:
        return None

    return {
        "sample_id": idx,
        "x": x,
        "y_plus": y_plus,
        "real_y_minus": real_y_minus,
        "score_head": args.score_head,
        "delta": args.delta,
        "max_rounds": args.max_rounds,
        "attempts": [],
        "best_attempt": None,
        "stopped_by_delta": False,
    }


def load_reward_scorer(args) -> DualRewardScorer:
    return DualRewardScorer(
        reward_model_path=args.reward_model_path,
        device=args.reward_device,
        max_length=args.reward_max_length,
        score_head=args.score_head,
        lambda_safe=args.lambda_safe,
    )


def load_generator(args) -> PromptEngineeringGenerator:
    return PromptEngineeringGenerator(
        base_model_path=args.base_model_path,
        device=args.generator_device,
        torch_dtype=args.torch_dtype,
        device_map=args.generator_device_map,
        max_input_length=args.generator_max_input_length,
    )


def precompute_known_rewards(states: List[Dict[str, Any]], args) -> None:
    print("[INFO] Loading reward model R for known scores...")
    scorer = load_reward_scorer(args)
    print("[INFO] Reward model loaded.")

    for pos, state in enumerate(states):
        y_plus_reward, y_plus_help, y_plus_safe = scorer.score(state["x"], state["y_plus"])
        target_reward, target_help, target_safe = scorer.score(state["x"], state["real_y_minus"])

        state["y_plus_reward_score"] = y_plus_reward
        state["y_plus_help_score"] = y_plus_help
        state["y_plus_safe_score"] = y_plus_safe
        state["real_y_minus_reward_score"] = target_reward
        state["real_y_minus_help_score"] = target_help
        state["real_y_minus_safe_score"] = target_safe

        if (pos + 1) % 50 == 0:
            print(f"[INFO] Precomputed known rewards: {pos + 1}/{len(states)}")

    del scorer
    release_cuda_memory()
    print("[INFO] Released reward model R after known-score precompute.")


def generate_round(states: List[Dict[str, Any]], round_id: int, args) -> None:
    pending = [state for state in states if not state["stopped_by_delta"]]
    if not pending:
        return

    print(f"[INFO] Loading generator model f for round {round_id}...")
    generator = load_generator(args)
    print("[INFO] Generator loaded.")

    for pos, state in enumerate(pending):
        generation_prompt = build_generation_prompt(
            x=state["x"],
            y_plus=state["y_plus"],
            y_plus_reward=float(state["y_plus_reward_score"]),
            target_y_minus_reward=float(state["real_y_minus_reward_score"]),
            attempts=state["attempts"],
            score_head=args.score_head,
            max_feedback_chars=args.max_feedback_chars,
        )

        generated = ""
        for retry_id in range(args.empty_retries + 1):
            generated = generator.generate_one(
                prompt=generation_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                retry_id=retry_id,
            )
            if generated:
                break
        if not generated:
            generated = "[EMPTY RESPONSE]"

        state["_pending_generation"] = generated
        if args.save_generation_prompts:
            state["_pending_generation_prompt"] = generation_prompt

        if (pos + 1) % 20 == 0:
            print(f"[INFO] Generated round {round_id}: {pos + 1}/{len(pending)} pending samples")

    del generator
    release_cuda_memory()
    print(f"[INFO] Released generator model f after round {round_id}.")


def score_round(states: List[Dict[str, Any]], round_id: int, args) -> None:
    pending = [
        state
        for state in states
        if not state["stopped_by_delta"] and "_pending_generation" in state
    ]
    if not pending:
        return

    print(f"[INFO] Loading reward model R for round {round_id} generated-score feedback...")
    scorer = load_reward_scorer(args)
    print("[INFO] Reward model loaded.")

    for pos, state in enumerate(pending):
        generated = state.pop("_pending_generation")
        generation_prompt = state.pop("_pending_generation_prompt", None)
        reward_score, help_score, safe_score = scorer.score(state["x"], generated)
        abs_diff = abs(float(state["real_y_minus_reward_score"]) - reward_score)

        attempt = {
            "round": round_id,
            "generated_y_minus": generated,
            "reward_score": reward_score,
            "help_score": help_score,
            "safe_score": safe_score,
            "abs_diff": abs_diff,
        }
        if generation_prompt is not None:
            attempt["generation_prompt"] = generation_prompt

        state["attempts"].append(attempt)

        best_attempt = state.get("best_attempt")
        if best_attempt is None or abs_diff < float(best_attempt["abs_diff"]):
            state["best_attempt"] = attempt

        if abs_diff <= args.delta:
            state["stopped_by_delta"] = True
            state["best_attempt"] = attempt

        if (pos + 1) % 20 == 0:
            print(f"[INFO] Scored round {round_id}: {pos + 1}/{len(pending)} pending samples")

    del scorer
    release_cuda_memory()
    print(f"[INFO] Released reward model R after round {round_id}.")


def state_to_output_row(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    selected = state.get("best_attempt")
    if selected is None:
        return None

    stopped_by_delta = bool(state["stopped_by_delta"])
    selection_strategy = "first_within_delta" if stopped_by_delta else "closest_after_max_rounds"

    return {
        "sample_id": state["sample_id"],
        "x": state["x"],
        "y_plus": state["y_plus"],
        "real_y_minus": state["real_y_minus"],
        "score_head": state["score_head"],
        "delta": state["delta"],
        "max_rounds": state["max_rounds"],
        "y_plus_reward_score": state["y_plus_reward_score"],
        "y_plus_help_score": state["y_plus_help_score"],
        "y_plus_safe_score": state["y_plus_safe_score"],
        "real_y_minus_reward_score": state["real_y_minus_reward_score"],
        "real_y_minus_help_score": state["real_y_minus_help_score"],
        "real_y_minus_safe_score": state["real_y_minus_safe_score"],
        "attempts": state["attempts"],
        "stopped_by_delta": stopped_by_delta,
        "stop_round": int(selected["round"]) if stopped_by_delta else None,
        "selection_strategy": selection_strategy,
        "selected_round": int(selected["round"]),
        "selected_generated_y_minus": selected["generated_y_minus"],
        "generated_y_minus": selected["generated_y_minus"],
        "pred_y_minus": selected["generated_y_minus"],
        "selected_reward_score": selected["reward_score"],
        "selected_help_score": selected["help_score"],
        "selected_safe_score": selected["safe_score"],
        "selected_abs_diff_to_real_reward": selected["abs_diff"],
    }


def save_state_snapshot(states: List[Dict[str, Any]], path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for state in states:
            f.write(json.dumps(state, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_rounds <= 0:
        raise ValueError("--max_rounds must be positive.")
    if args.empty_retries < 0:
        raise ValueError("--empty_retries must be non-negative.")
    if args.seed >= 0:
        set_seed(args.seed)

    print("=" * 80)
    print("[INFO] Running staged low-memory prompt-engineering reward baseline.")
    print("[INFO] Generator f and reward model R are never kept in memory at the same time.")
    print("=" * 80)

    rows = load_jsonl(args.triplet_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    states: List[Dict[str, Any]] = []
    skipped = 0
    for idx, row in enumerate(rows):
        state = make_empty_state(idx, row, args)
        if state is None:
            skipped += 1
            continue
        states.append(state)

    print(f"[INFO] Loaded input rows: {len(rows)}")
    print(f"[INFO] Valid states: {len(states)}")
    print(f"[INFO] Skipped rows before generation: {skipped}")
    print(f"[INFO] max_rounds={args.max_rounds}, delta={args.delta}, score_head={args.score_head}")

    precompute_known_rewards(states, args)
    save_state_snapshot(states, args.output_file + ".state_after_known_scores.jsonl")

    for round_id in range(1, args.max_rounds + 1):
        pending_count = sum(1 for state in states if not state["stopped_by_delta"])
        print("=" * 80)
        print(f"[ROUND {round_id}] Pending samples before generation: {pending_count}")
        if pending_count == 0:
            break

        generate_round(states, round_id, args)
        score_round(states, round_id, args)
        save_state_snapshot(states, args.output_file + f".state_after_round{round_id}.jsonl")

        stopped_count = sum(1 for state in states if state["stopped_by_delta"])
        print(f"[ROUND {round_id}] Stopped by delta so far: {stopped_count}/{len(states)}")

    output_rows = []
    for state in states:
        output_row = state_to_output_row(state)
        if output_row is not None:
            output_rows.append(output_row)

    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(output_rows, skipped=skipped, args=args)
    if args.summary_file:
        save_json(summary, args.summary_file)

    print("=" * 80)
    print("[DONE] Staged prompt-engineering reward baseline finished.")
    print(f"[DONE] Output rows: {len(output_rows)}")
    print(f"[DONE] Skipped rows: {skipped}")
    print(f"[DONE] Output file: {args.output_file}")
    if args.summary_file:
        print(f"[DONE] Summary file: {args.summary_file}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
