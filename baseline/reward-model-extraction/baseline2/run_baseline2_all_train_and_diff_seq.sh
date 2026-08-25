#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/path/to/code"
MODEL_DATA_ROOT="/path/to/code"

BASELINE2_SCRIPT="${MODEL_DATA_ROOT}/baseline/reward-model-extraction/baseline2/train_baseline2_miniplm_rm_difference_sampling.py"
DIFF_SCRIPT="${WORKDIR}/reward-model-extraction/scripts/eval_target_vs_substitute_diff.py"

OUT_ROOT="${MODEL_DATA_ROOT}/output"
DATA_OUT_ROOT="${MODEL_DATA_ROOT}/data"
LOG_ROOT="${MODEL_DATA_ROOT}/logs/baseline2"
RESULT_ROOT="${MODEL_DATA_ROOT}/results/baseline2"

mkdir -p "$OUT_ROOT" "$DATA_OUT_ROOT" "$LOG_ROOT" "$RESULT_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/path/to/tmp
export HF_HOME=/path/to/cache/huggingface
export TRANSFORMERS_CACHE=/path/to/cache/huggingface/transformers
export TORCH_HOME=/path/to/cache/torch

mkdir -p "$TMPDIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

cd "$WORKDIR"

RESULT_TSV="${RESULT_ROOT}/baseline2_diff_results.tsv"

# FORCE=1 时会强制重跑；默认 FORCE=0，已经有结果就跳过，避免重复浪费时间
FORCE="${FORCE:-0}"

init_result_file() {
  if [[ ! -f "$RESULT_TSV" ]]; then
    echo -e "TargetRM\tTag\tDiff_avg\tDiff_var\tDiff_std\tN_items\tSimple_Defender Evaluation_acc\tSelected_N\tSelected_diff_avg\tSelected_diff_var\tTrain_metric_path\tDiff_summary_path" > "$RESULT_TSV"
  fi
}

append_result() {
  local target_name="$1"
  local tag="$2"
  local metric_path="$3"
  local diff_summary="$4"

  python - "$target_name" "$tag" "$metric_path" "$diff_summary" "$RESULT_TSV" <<'PY'
import json
import sys
from pathlib import Path

target_name, tag, metric_path, diff_summary, result_tsv = sys.argv[1:]

diff = json.load(open(diff_summary, "r", encoding="utf-8"))
s = diff["all"]["item_level"]

metric = json.load(open(metric_path, "r", encoding="utf-8"))
eval_metrics = metric.get("eval_metrics", {})
sampling = metric.get("sampling_metrics", {})

row = [
    target_name,
    tag,
    f"{s['Diff_avg']:.6f}",
    f"{s['Diff_var']:.6f}",
    f"{s['Diff_std']:.6f}",
    str(s["count"]),
    f"{eval_metrics.get('gold_acc', float('nan')):.6f}" if eval_metrics.get("gold_acc") is not None else "NA",
    str(sampling.get("num_selected_examples", "NA")),
    f"{sampling.get('selected_diff_avg', float('nan')):.6f}" if sampling.get("selected_diff_avg") is not None else "NA",
    f"{sampling.get('selected_diff_var', float('nan')):.6f}" if sampling.get("selected_diff_var") is not None else "NA",
    metric_path,
    diff_summary,
]

path = Path(result_tsv)
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
header = "TargetRM\tTag\tDiff_avg\tDiff_var\tDiff_std\tN_items\tSimple_Defender Evaluation_acc\tSelected_N\tSelected_diff_avg\tSelected_diff_var\tTrain_metric_path\tDiff_summary_path"

# 去掉旧的同 tag 行，防止重复追加
new_lines = []
if not lines:
    new_lines.append(header)
else:
    new_lines.append(lines[0])
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) > 1 and parts[1] == tag:
            continue
        new_lines.append(line)

new_lines.append("\t".join(row))
path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
print("[OK] saved result:", "\t".join(row[:6]))
PY
}

run_one() {
  local target_name="$1"
  local tag="$2"
  local target_base="$3"
  local target_adapter="$4"

  local ref_out="${OUT_ROOT}/baseline2_ref_${tag}"
  local selected_path="${DATA_OUT_ROOT}/baseline2_miniplm_rm_${tag}_alpha05_train.jsonl"
  local all_scored_path="${DATA_OUT_ROOT}/baseline2_miniplm_rm_${tag}_all_scored.jsonl"
  local student_out="${OUT_ROOT}/baseline2_miniplm_rm_${tag}_to_roberta"
  local diff_out="${OUT_ROOT}/diff_eval_baseline2_miniplm_rm_${tag}_to_roberta"

  local metric_path="${student_out}/baseline2_miniplm_rm_metrics.json"
  local diff_summary="${diff_out}/diff_summary.json"

  local train_log="${LOG_ROOT}/${tag}_train.log"
  local diff_log="${LOG_ROOT}/${tag}_diff.log"

  echo "================================================================================"
  echo "[START] ${target_name} | ${tag}"
  echo "================================================================================"

  if [[ "$FORCE" == "1" ]]; then
    echo "[INFO] FORCE=1, removing old outputs for ${tag}"
    rm -rf "$ref_out" "$student_out" "$diff_out"
    rm -f "$selected_path" "$all_scored_path"
  fi

  if [[ ! -f "$metric_path" ]]; then
    echo "[TRAIN] baseline2 for ${target_name}"
    CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v python "$BASELINE2_SCRIPT" \
      --target_base_model "$target_base" \
      --target_adapter_path "$target_adapter" \
      --reference_model_path models/roberta-base \
      --student_model_path models/roberta-base \
      --train_path data/train.jsonl \
      --eval_path data/test.jsonl \
      --reference_output_dir "$ref_out" \
      --selected_output_path "$selected_path" \
      --all_scored_output_path "$all_scored_path" \
      --student_output_dir "$student_out" \
      --max_reference_samples 1000 \
      --max_score_samples 30000 \
      --sampling_ratio 0.5 \
      --reference_epochs 1 \
      --student_epochs 1 \
      --reference_batch_size 8 \
      --student_batch_size 8 \
      --eval_batch_size 16 \
      --gradient_accumulation_steps 1 \
      --max_length 512 \
      --target_batch_size 1 \
      --reference_score_batch_size 16 \
      --reference_lr 2e-5 \
      --student_lr 2e-5 \
      --bf16 2>&1 | tee "$train_log"
  else
    echo "[SKIP TRAIN] Found: $metric_path"
  fi

  if [[ ! -f "$diff_summary" ]]; then
    echo "[DIFF] baseline2 Diff eval for ${target_name}"
    CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v python "$DIFF_SCRIPT" \
      --target_base_model "$target_base" \
      --target_adapter_path "$target_adapter" \
      --substitute_model_path "$student_out" \
      --train_path data/train.jsonl \
      --test_path data/test.jsonl \
      --sample_train 500 \
      --sample_test 500 \
      --output_dir "$diff_out" \
      --max_length 512 \
      --target_batch_size 1 \
      --substitute_batch_size 16 \
      --bf16 2>&1 | tee "$diff_log"
  else
    echo "[SKIP DIFF] Found: $diff_summary"
  fi

  append_result "$target_name" "$tag" "$metric_path" "$diff_summary"

  echo "================================================================================"
  echo "[DONE] ${target_name} | ${tag}"
  echo "Train metrics: $metric_path"
  echo "Diff summary : $diff_summary"
  echo "Train log    : $train_log"
  echo "Diff log     : $diff_log"
  echo "================================================================================"
}

init_result_file

# 从小到大依次运行
run_one "LLaMA3.2-3B" "exp8_llama32_3b" \
  "/path/to/models/llama32-3b" \
  "output/target_rm_llama32_3b_full_margin5_e1"

run_one "LLaMA2-7B" "exp3_llama2_7b" \
  "/path/to/target-model/llama2-7b" \
  "output/target_rm_llama2_7b_lora_full_margin5_e2"

run_one "Mistral-7B" "exp12_mistral_7b" \
  "/path/to/models/mistral-7b-v0.1" \
  "output/target_rm_mistral_7b_full_margin5_e1"

run_one "Qwen3-8B" "exp10_qwen3_8b" \
  "/path/to/models/qwen3-8b" \
  "output/target_rm_qwen3_8b_full_margin5_e1"

run_one "LLaMA2-13B" "exp6_llama2_13b" \
  "/path/to/models/llama2-13b-hf" \
  "output/target_rm_llama2_13b_full_margin5_e1"

echo
echo "================ Final baseline2 results ================"
cat "$RESULT_TSV"
echo "========================================================="
