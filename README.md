# Reward Model Privacy

This repository collects privacy and security experiments for reward models (RMs) across three main areas: **reward model extraction**, **membership inference**, and **training-data reconstruction**. It also includes corresponding baselines and a differential privacy defense.

The repository is organized primarily as experimental scripts. Datasets, pretrained models, checkpoints, logs, and generated outputs are not included. Before running an experiment, replace placeholder paths in the scripts or supply valid local paths through command-line arguments.

## Repository Structure

```text
reward_privacy-main/
├── reward-model-extraction/       # Main RM extraction methods, evaluation, and ablations
├── membership_inference/          # Four-stage membership inference attack
├── data_reconstruction/           # Main training-data reconstruction method and evaluation
├── defense/                       # Differential privacy defense experiments
├── baseline/                      # Baselines for the three privacy attacks
├── README.md                      # Repository overview
└── requirements.txt               # Shared Python dependencies
```

## Directories and Files

### `reward-model-extraction/`

This directory implements the complete reward model extraction workflow: train a target RM, construct attacker data, query target scores, train a substitute RM, and compare the target and substitute models.

- `README.md`: Module overview, data format, and path conventions.
- `requirements.txt`: Minimal dependencies for this module; the root requirements file is a repository-wide superset.
- `run_all_agreement_diff.sh`: Batch computation of ranking agreement and differences across experiments.

Python implementations in `src/`:

- `__init__.py`: Marks `src` as a Python package.
- `train_target_rm.py`: Trains a sequence-classification target RM such as RoBERTa.
- `train_target_rm_llama_lora.py`: Fine-tunes a causal-language-model target RM with LoRA.
- `train_extracted_rm_two_stage.py`: Trains an encoder-based substitute RM in two stages: preference learning followed by target-score regression.
- `train_extracted_rm_llama_lora_two_stage.py`: Performs two-stage training for a LoRA causal-LM substitute RM.
- `score_aux_with_target.py`: Scores attacker auxiliary samples with a target sequence-classification RM.
- `score_aux_with_llama_lora.py`: Scores auxiliary samples with a LoRA causal-LM RM.
- `score_aux_with_public_reward_model.py`: Scores auxiliary data with a public reward model.
- `prepare_attacker_auxiliary_dataset.py`: Builds the auxiliary dataset used to query the target model.
- `prepare_attacker_auxiliary_dataset_disjoint.py`: Builds an auxiliary dataset disjoint from other training data.
- `prepare_attacker_preference_dataset.py`: Prepares preference data for the first stage of substitute-model training.
- `prepare_preference_aux.py`: Converts auxiliary records into a preference-learning format.
- `prepare_defender_evaluation_dataset.py`: Prepares data for defender-side evaluation.
- `prepare_defender_evaluation_dataset_builder.py`: Provides reusable defender evaluation dataset-building logic.
- `eval_public_reward_model_extraction.py`: Evaluates substitute models distilled or extracted from public RMs.
- `eval_rm_misclassified.py`: Analyzes samples misclassified by target or substitute RMs.
- `extraction_metrics.py`: Shared extraction metrics, including positive-pair ranking agreement.

Experiment launchers and utilities in `scripts/`:

- `01_train_target_rm_roberta.sh`: Trains a RoBERTa target RM.
- `02_score_attacker_auxiliary_with_target.sh`: Queries the target RM on auxiliary data.
- `03_train_extracted_rm_two_stage_exp1.sh`: Runs two-stage substitute-RM training for Experiment 1.
- `04_train_extracted_rm_two_stage_exp2_distilroberta.sh`: Runs Experiment 2 with DistilRoBERTa.
- `05_train_target_rm_llama2_7b_lora.sh`: Trains a Llama-2-7B LoRA target RM.
- `06_score_attacker_auxiliary_with_adapter.sh`: Scores auxiliary data with a LoRA target adapter.
- `07_run_exp3_exp4_llama_target.sh`: Runs Experiments 3 and 4 with a Llama target.
- `08_run_ablation_llama2_7b_roberta.sh`: Runs the Llama/RoBERTa architecture ablation.
- `08_train_exp5_llama2_7b_to_llama2_7b.sh`: Runs same-architecture Llama-2-7B-to-Llama-2-7B extraction.
- `09_run_exp3_exp4_margin5_target.sh`: Runs Experiments 3 and 4 with the configured margin setting.
- `10_train_target_rm_any_causal_lm_lora.sh`: Generic LoRA target-RM launcher for compatible causal language models.
- `run_one_target_roberta_33.sh`: Runs one RoBERTa target experiment with the 33% data setting.
- `run_exp8_exp10_exp12_exp6_roberta_all.sh`: Runs multiple RoBERTa experiments in a batch.
- `run_ours_student_single_a100.sh`: Runs the proposed student-model method on one A100 GPU.
- `run_ours_student_scaling_a100.sh`: Runs student-model scaling experiments on A100 GPUs.
- `run_ours_student_scaling_remaining_single_a100.sh`: Resumes remaining single-A100 scaling experiments.
- `run_ablation_attacker_preference_diff.sh`: Ablates differences in attacker preference data.
- `run_ablation_attacker_preference_teacher_scaling.sh`: Studies teacher scaling with attacker preference data.
- `run_defender_evaluation_all_teachers.sh`: Evaluates all teacher models.
- `run_defender_evaluation_all_teachers_diff.sh`: Compares evaluation differences across teacher models.
- `run_defender_evaluation_comparison_and_diff.sh`: Runs defender-side comparisons and difference analysis.
- `run_defender_evaluation_target_experiments.sh`: Runs defender evaluation for target-model experiments.
- `batch_compute_agreement.py`: Computes prediction or ranking agreement between target and substitute RMs in batches.
- `convert_defender_evaluation_to_preference.py`: Converts defender evaluation records into preference pairs.
- `eval_ours_vs_baseline_metrics.py`: Summarizes and compares metrics for the proposed method and baselines.
- `eval_target_vs_substitute_diff.py`: Analyzes differences between target and substitute model outputs.
- `evaluate_teacher_accuracies.py`: Computes the accuracy of multiple teacher RMs.
- `download_target_models.sh`: Downloads target models required by the experiments.

Query-budget ablations in `scripts/ablation/query_budget/`:

- `run_roberta_query_budget_all_teachers.sh`: Runs query-budget ablations for all RoBERTa teachers.
- `run_attacker_auxiliary_query_budget_ablation.sh`: Repeats extraction under different auxiliary-query budgets.

Public reward model experiments in `scripts/open_source_reward_models/`:

- `run_public_reward_model_extraction.sh`: Runs standard extraction from a public RM.
- `run_public_reward_model_disjoint_extraction.sh`: Runs public-RM extraction with disjoint auxiliary data.
- `run_public_reward_model_joint_distill.sh`: Runs the joint-distillation variant.
- `run_public_reward_model_pair_aux_distill.sh`: Distills a public RM using paired auxiliary data.

### `membership_inference/`

The attack is divided into four sequential stages. Intermediate results are passed between stages as JSON or JSONL artifacts with metadata.

- `method/3_1_target_data_decomposition.py`: Loads target preference data, normalizes membership labels, and creates Stage 1 records.
- `method/3_2_candidate_response_generation.py`: Generates candidate responses and scores them with a reward model.
- `method/3_3_llm_update_ppo_full.py`: Performs an independent PPO probe for each record and extracts update signals such as gradient norms.
- `method/3_4_membership_inference.py`: Combines reward margins and gradient norms, applies a calibrated threshold, and reports AUC, F1, TPR/FPR, and related metrics.
- `tool/__init__.py`: Marks the utility directory as a Python package.
- `tool/paper_mia.py`: Shared I/O, field normalization, reward scoring, token encoding, PPO loss, and gradient utilities.

### `data_reconstruction/`

- `method/step1_finetune_seq2seq.py`: Fine-tunes a Seq2Seq or causal language model on `(x, y_plus) -> y_minus`, with LoRA and sequence-length budgeting support.
- `method/stage2_generate_candidates_k3.py`: Generates three reconstruction candidates for each input and records intermediate results.
- `method/stage3_select_lowest_reward.py`: Selects the lowest-reward candidate as the final reconstruction.
- `evaluation/eval_bleu_cosine.py`: Computes BLEU-style lexical overlap and cosine similarity between Transformer representations of reconstructed and reference text.

### `defense/`

`defense/reward-model-extraction/` trains LoRA reward models with record-level DP-SGD and reuses scoring and evaluation code from the main extraction directory.

- `README.md`: Defense module overview.
- `train_target_rm_llama_lora_dp.py`: Self-contained DP-SGD LoRA RM trainer that clips per-record preference-pair gradients, adds Gaussian noise, and tracks privacy loss with an RDP accountant.
- `run_one_target_roberta_33_dp.sh`: Runs a single-target RoBERTa DP experiment with the 33% data setting.
- `run_llama2_7b_dp_eps8_33.sh`: Runs the Llama-2-7B DP experiment with an approximate target privacy budget of epsilon 8 and 33% of the data.
- `run_all_remaining_dp_eps8_33.sh`: Runs the remaining epsilon-8, 33%-data DP configurations.

### `baseline/`

#### Reward Model Extraction Baselines

- `reward-model-extraction/README.md`: Overview of the two extraction baselines.
- `reward-model-extraction/baseline1/train_minillm_rm_reverse_kl.py`: MiniLLM-style reverse-KL distillation that converts target and substitute RM scores into soft preference distributions.
- `reward-model-extraction/baseline2/train_baseline2_miniplm_rm_difference_sampling.py`: MiniPLM-style reward-difference sampling and substitute-RM training.
- `reward-model-extraction/baseline2/run_baseline2_all_train_and_diff_seq.sh`: Sequentially runs baseline 2 training and difference evaluation.

#### Data Reconstruction Baseline

- `data_reconstruction_baseline/README.md`: Directory overview and usage example.
- `data_reconstruction_baseline/prompt_engineering_reward/README.md`: Description of the iterative reward-feedback prompting method.
- `data_reconstruction_baseline/prompt_engineering_reward/baseline_components.py`: Shared generator, reward scorer, prompt templates, arguments, and I/O components.
- `data_reconstruction_baseline/prompt_engineering_reward/run_baseline.py`: Low-memory iterative reconstruction entry point that loads the generator and reward model in separate stages.
- `data_reconstruction_baseline/prompt_engineering_reward/run_llama2_7b.sh`: Runs the Llama-2-7B prompting baseline and BLEU/cosine evaluation.

#### Membership Inference Baseline: SPV-MIA

- `ANeurIPS2024_SPV-MIA-main/scripts/spv_mia_safe_rlhf.py`: Implements SPV-MIA for Safe-RLHF models and computes ROC/AUC metrics.
- `ANeurIPS2024_SPV-MIA-main/scripts/run_spv_mia_5models_2gpu.sh`: Runs five model configurations across two GPUs.
- `ANeurIPS2024_SPV-MIA-main/scripts/tail_spv_mia_5models_logs.sh`: Follows logs for the five SPV-MIA runs.
- `ANeurIPS2024_SPV-MIA-main/scripts/refresh_spv_mia_low_fpr_summary.py`: Recomputes and summarizes SPV-MIA metrics in the low-FPR region.

#### Membership Inference Baseline: ICP-MIA

- `ICP-MIA-main/prepare_safe_rlhf_data.py`: Converts Safe-RLHF data into the format required by ICP-MIA.
- `ICP-MIA-main/icp_mia_attack.py`: Implements similarity-prefix and self-perturbation ICP attacks, configuration loading, evaluation, and plotting.
- `ICP-MIA-main/scripts/run_icp_mia_5models.sh`: Runs ICP-MIA for five model configurations.

## Suggested Workflow

1. Prepare preference JSONL files, attacker auxiliary data, and local model paths.
2. Validate the fields and model compatibility of one Python entry point with a small sample and short sequence length.
3. Replace placeholder paths, GPU IDs, and experiment sizes in the relevant shell launcher.
4. After validating the main workflow, run the batch, ablation, and comparison scripts under `scripts/`.

Run `python path/to/script.py --help` before using a Python entry point to review its actual command-line arguments.
