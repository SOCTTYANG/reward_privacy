# Reward Model Privacy

本仓库汇集了奖励模型（Reward Model, RM）的三类隐私与安全实验：**奖励模型窃取**、**成员推断**和**训练数据重构**，并提供对应的基线方法与基于差分隐私的防御实现。代码以实验脚本为主；数据集、预训练模型、检查点和实验输出不随仓库提供，运行前需要把脚本中的占位路径或命令行参数替换为本地路径。

## 目录总览

reward_privacy-main/
├── reward-model-extraction/       # 奖励模型窃取主方法、评估与消融实验
├── membership_inference/          # 四阶段成员推断攻击
├── data_reconstruction/           # 训练数据重构主方法与评估
├── defense/                       # 差分隐私防御实验
├── baseline/                      # 三类任务的对比基线
├── README.md                      # 本说明文件
└── requirements.txt               # 仓库通用 Python 依赖


## 子目录与文件说明

### `reward-model-extraction/`：奖励模型窃取

该目录实现“训练目标 RM → 构造攻击者数据 → 查询目标得分 → 训练替代 RM → 比较目标与替代模型”的完整流程。

- `README.md`：该模块的简要流程、数据格式和路径说明。
- `requirements.txt`：该模块原有的最小依赖列表；根目录 requirements 是整个仓库的超集。
- `run_all_agreement_diff.sh`：批量计算不同实验结果之间的排序一致性/差异。

`src/` 中的 Python 实现：

- `__init__.py`：将 `src` 标记为 Python 包。
- `train_target_rm.py`：训练基于序列分类模型（如 RoBERTa）的目标奖励模型。
- `train_target_rm_llama_lora.py`：用 LoRA 微调因果语言模型形式的目标奖励模型。
- `train_extracted_rm_two_stage.py`：两阶段训练编码器式替代奖励模型，先学习偏好、再拟合目标分数。
- `train_extracted_rm_llama_lora_two_stage.py`：两阶段训练 LoRA 因果语言模型替代 RM。
- `score_aux_with_target.py`：用目标序列分类 RM 为攻击者辅助样本打分。
- `score_aux_with_llama_lora.py`：用带 LoRA 适配器的因果语言模型 RM 打分。
- `score_aux_with_public_reward_model.py`：调用公开奖励模型给辅助数据打分。
- `prepare_attacker_auxiliary_dataset.py`：构造攻击者用于查询目标模型的辅助数据集。
- `prepare_attacker_auxiliary_dataset_disjoint.py`：构造与其他训练数据互斥的辅助数据集。
- `prepare_attacker_preference_dataset.py`：整理替代模型第一阶段所需的攻击者偏好数据。
- `prepare_preference_aux.py`：把辅助记录整理为偏好学习可用的格式。
- `prepare_defender_evaluation_dataset.py`：准备防御方评估数据。
- `prepare_defender_evaluation_dataset_builder.py`：提供防御评估集构建逻辑与可复用处理函数。
- `eval_public_reward_model_extraction.py`：评估从公开 RM 蒸馏/窃取得到的替代模型。
- `eval_rm_misclassified.py`：分析目标或替代 RM 的误分类样本。
- `extraction_metrics.py`：共享的窃取评估指标，例如正样本对排序一致率。

`scripts/` 中的入口与实验编排：

- `01_train_target_rm_roberta.sh`：训练 RoBERTa 目标 RM。
- `02_score_attacker_auxiliary_with_target.sh`：用目标 RM 查询辅助数据。
- `03_train_extracted_rm_two_stage_exp1.sh`：运行实验 1 的两阶段替代 RM 训练。
- `04_train_extracted_rm_two_stage_exp2_distilroberta.sh`：以 DistilRoBERTa 运行实验 2。
- `05_train_target_rm_llama2_7b_lora.sh`：训练 Llama-2-7B LoRA 目标 RM。
- `06_score_attacker_auxiliary_with_adapter.sh`：用 LoRA 目标模型为辅助集打分。
- `07_run_exp3_exp4_llama_target.sh`：运行以 Llama 为目标模型的实验 3/4。
- `08_run_ablation_llama2_7b_roberta.sh`：运行 Llama 与 RoBERTa 组合的消融实验。
- `08_train_exp5_llama2_7b_to_llama2_7b.sh`：运行 Llama-2-7B 到 Llama-2-7B 的同架构窃取。
- `09_run_exp3_exp4_margin5_target.sh`：在指定 margin 设置下运行实验 3/4。
- `10_train_target_rm_any_causal_lm_lora.sh`：面向任意兼容因果语言模型的 LoRA 目标 RM 入口。
- `run_one_target_roberta_33.sh`：运行单个 33% 数据设置的 RoBERTa 目标实验。
- `run_exp8_exp10_exp12_exp6_roberta_all.sh`：批量执行多组 RoBERTa 实验。
- `run_ours_student_single_a100.sh`：面向单张 A100 的主方法学生模型实验。
- `run_ours_student_scaling_a100.sh`：在 A100 上运行学生模型规模扩展实验。
- `run_ours_student_scaling_remaining_single_a100.sh`：补跑尚未完成的单 A100 规模实验。
- `run_ablation_attacker_preference_diff.sh`：攻击者偏好数据差异消融。
- `run_ablation_attacker_preference_teacher_scaling.sh`：教师规模与攻击者偏好数据消融。
- `run_defender_evaluation_all_teachers.sh`：批量评估全部教师模型。
- `run_defender_evaluation_all_teachers_diff.sh`：批量比较教师模型评估结果差异。
- `run_defender_evaluation_comparison_and_diff.sh`：运行防御评估对比及差异统计。
- `run_defender_evaluation_target_experiments.sh`：运行目标模型侧的防御评估实验。
- `batch_compute_agreement.py`：批量计算目标/替代 RM 的预测或排序一致性。
- `convert_defender_evaluation_to_preference.py`：把防御评估记录转换为偏好对格式。
- `eval_ours_vs_baseline_metrics.py`：汇总并比较主方法与基线指标。
- `eval_target_vs_substitute_diff.py`：分析目标模型与替代模型输出差异。
- `evaluate_teacher_accuracies.py`：统计多个教师 RM 的准确率。
- `download_target_models.sh`：下载实验所需的目标模型。

`scripts/ablation/query_budget/`：

- `run_roberta_query_budget_all_teachers.sh`：对所有 RoBERTa 教师运行查询预算消融。
- `run_attacker_auxiliary_query_budget_ablation.sh`：在不同攻击者辅助查询预算下重复窃取流程。

`scripts/open_source_reward_models/`：

- `run_public_reward_model_extraction.sh`：从公开奖励模型执行标准窃取流程。
- `run_public_reward_model_disjoint_extraction.sh`：使用互斥辅助数据执行公开 RM 窃取。
- `run_public_reward_model_joint_distill.sh`：运行联合蒸馏变体。
- `run_public_reward_model_pair_aux_distill.sh`：使用成对辅助数据进行蒸馏。

### `membership_inference/`：成员推断

该方法将攻击拆为四个顺序阶段，阶段间通过带元数据的 JSON/JSONL artifact 传递结果。

- `method/3_1_target_data_decomposition.py`：读取目标偏好数据，规范化成员标签并生成第一阶段记录。
- `method/3_2_candidate_response_generation.py`：为每条目标记录生成候选回答，并用奖励模型打分。
- `method/3_3_llm_update_ppo_full.py`：对每条记录独立执行 PPO 探测，提取梯度范数等更新信号。
- `method/3_4_membership_inference.py`：融合奖励间隔与梯度范数，按校准阈值预测成员身份并计算 AUC、F1、TPR/FPR 等指标。
- `tool/__init__.py`：将工具目录标记为 Python 包。
- `tool/paper_mia.py`：四阶段共享的 I/O、字段规范化、奖励打分、token 编码、PPO 损失及梯度工具。

### `data_reconstruction/`：训练数据重构

- `method/step1_finetune_seq2seq.py`：用 `(x, y_plus) → y_minus` 任务微调 Seq2Seq/因果语言模型，并支持 LoRA 与长度预算控制。
- `method/stage2_generate_candidates_k3.py`：对每个输入生成 3 个候选重构回答并组织中间结果。
- `method/stage3_select_lowest_reward.py`：用奖励模型选择奖励最低的候选，作为最终重构结果。
- `evaluation/eval_bleu_cosine.py`：计算重构文本与真实文本的 BLEU 风格词面指标及 Transformer 表征余弦相似度。

### `defense/`：差分隐私防御

`defense/reward-model-extraction/` 使用记录级 DP-SGD 训练 LoRA 奖励模型，并复用主窃取目录中的打分与评估代码。

- `README.md`：防御模块简介。
- `train_target_rm_llama_lora_dp.py`：自包含的 DP-SGD LoRA RM 训练器；按偏好对裁剪单记录梯度、注入高斯噪声并用 RDP accountant 记录隐私损失。
- `run_one_target_roberta_33_dp.sh`：运行单目标 RoBERTa 的 33% 数据 DP 实验。
- `run_llama2_7b_dp_eps8_33.sh`：运行 Llama-2-7B、目标隐私预算约为 epsilon=8 的 33% 数据实验。
- `run_all_remaining_dp_eps8_33.sh`：批量补跑其余 epsilon=8、33% 数据设置。

### `baseline/`：对比基线

#### 奖励模型窃取基线

- `reward-model-extraction/README.md`：两类窃取基线说明。
- `reward-model-extraction/baseline1/train_minillm_rm_reverse_kl.py`：MiniLLM 风格的反向 KL 蒸馏基线，把目标/替代 RM 分数转成软偏好分布。
- `reward-model-extraction/baseline2/train_baseline2_miniplm_rm_difference_sampling.py`：MiniPLM 风格的奖励差异采样与替代 RM 训练。
- `reward-model-extraction/baseline2/run_baseline2_all_train_and_diff_seq.sh`：顺序运行 baseline2 的训练与差异评估。

#### 数据重构基线

- `data_reconstruction_baseline/README.md`：重构基线目录说明与运行示例。
- `data_reconstruction_baseline/prompt_engineering_reward/README.md`：基于奖励反馈的迭代提示方法说明。
- `data_reconstruction_baseline/prompt_engineering_reward/baseline_components.py`：生成器、奖励打分器、提示模板、参数与 I/O 公共组件。
- `data_reconstruction_baseline/prompt_engineering_reward/run_baseline.py`：低显存迭代重构入口，分阶段加载生成模型与奖励模型。
- `data_reconstruction_baseline/prompt_engineering_reward/run_llama2_7b.sh`：运行 Llama-2-7B 提示基线并调用 BLEU/余弦评估。

#### 成员推断基线：SPV-MIA

- `ANeurIPS2024_SPV-MIA-main/scripts/spv_mia_safe_rlhf.py`：在 Safe-RLHF 双奖励模型上实现 SPV-MIA，并计算 ROC/AUC 等指标。
- `ANeurIPS2024_SPV-MIA-main/scripts/run_spv_mia_5models_2gpu.sh`：用两张 GPU 批量运行五个模型。
- `ANeurIPS2024_SPV-MIA-main/scripts/tail_spv_mia_5models_logs.sh`：集中跟踪五个模型的运行日志。
- `ANeurIPS2024_SPV-MIA-main/scripts/refresh_spv_mia_low_fpr_summary.py`：重新汇总低 FPR 区域的 SPV-MIA 指标。

#### 成员推断基线：ICP-MIA

- `ICP-MIA-main/prepare_safe_rlhf_data.py`：把 Safe-RLHF 数据准备成 ICP-MIA 所需格式。
- `ICP-MIA-main/icp_mia_attack.py`：实现基于相似前缀或自扰动的 ICP 成员推断、配置加载、评估与绘图。
- `ICP-MIA-main/scripts/run_icp_mia_5models.sh`：批量对五个模型运行 ICP-MIA。

## 推荐运行顺序

1. 准备偏好 JSONL、攻击者辅助数据和本地模型路径。
2. 先用小样本及较短序列验证单个 Python 入口的字段和模型兼容性。
3. 再修改对应 `.sh` 中的占位路径、GPU 编号和实验规模。
4. 主流程成功后运行 `scripts/` 中的批处理、消融和对比脚本。

所有 Python 入口均建议先执行 `python path/to/script.py --help` 查看实际参数。
