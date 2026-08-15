# Prompt-Engineering Reward Baseline

This baseline does not fine-tune the reconstruction model. It provides the generator with `(x, y_plus)`, the reward of `y_plus`, and the target reward of the real `y_minus`. The generator proposes a response, the reward model scores it, and the score is fed back into the next prompt.

The process stops when the generated reward is within `delta` of the target reward or when `max_rounds` is reached. If no attempt reaches the threshold, the closest-reward attempt is selected.

## Files

- `baseline_components.py`: generator, reward scorer, prompts, arguments, and shared I/O.
- `run_baseline.py`: low-memory iterative baseline implementation; generator and reward model are loaded in separate stages.
- `run_llama2_7b.sh`: end-to-end Llama2-7B launcher and BLEU/cosine evaluation.
- `analysis/collect_comparison_samples.py`: collect matched main-method/baseline examples.
- `analysis/format_comparison_samples.py`: format selected examples as clean JSONL and Markdown.

## Run

```bash
cd /path/to/reward-model-privacy
MAX_SAMPLES=20 bash baseline/data_reconstruction_baseline/prompt_engineering_reward/run_llama2_7b.sh
```

Important controls:

```bash
DELTA=0.05 MAX_ROUNDS=3 SCORE_HEAD=help \
bash baseline/data_reconstruction_baseline/prompt_engineering_reward/run_llama2_7b.sh
```
