# 文件简要索引

回答：每个文件做什么、优先看哪一组。下列代码路径相对于 `code/`；运行时工作目录必须是 `code/`。

## 正式训练与测试

| 文件 | 简要说明 |
|---|---|
| [late_subject_domain.py](../code/late_subject_domain.py) | 后期无对抗／高低组／受试者身份三分支，正式43起点实验 |
| [repeat_group_domain42.py](../code/repeat_group_domain42.py) | 预训练42／后期42配对训练 |
| [repeat_group_domain44.py](../code/repeat_group_domain44.py) | 预训练43／后期44配对训练，并非独立44预训练 |
| [evaluate_subject_domain.py](../code/evaluate_subject_domain.py) | 测试三种后期域定义 |
| [evaluate_group_domain42.py](../code/evaluate_group_domain42.py) | 测试42／42配对模型 |
| [evaluate_group_domain44.py](../code/evaluate_group_domain44.py) | 测试43／44配对模型 |
| [compare_pooled_csp_lda.py](../code/compare_pooled_csp_lda.py) | 联合CSP+LDA传统参考；与分组用CSP+LDA区分 |

## 历史后期与消融

| 文件 | 简要说明 |
|---|---|
| [warm_dann_pair.py](../code/warm_dann_pair.py) | 初步后期DANN配对，不与最终批次方案混算 |
| [repeat_warm_dann_seed43.py](../code/repeat_warm_dann_seed43.py) | 初步后期DANN的43起点重复 |
| [late_classwise_dann.py](../code/late_classwise_dann.py) | 后期类条件DANN训练 |
| [evaluate_late_classwise.py](../code/evaluate_late_classwise.py) | 后期类条件模型测试 |
| [evaluate_original_pair.py](../code/evaluate_original_pair.py) | 早期原基线与plan_a_domain003的固定session2测试；不是warm配对测试 |

## 历史算法探索

| 文件 | 简要说明 |
|---|---|
| [repeat_dann_pair.py](../code/repeat_dann_pair.py) | 从头训练的43／44基线与方案A |
| [train_alignment.py](../code/train_alignment.py) | 早期global／classwise对齐，历史运行曾中止 |
| [train_classwise_original.py](../code/train_classwise_original.py) | 原框架下类条件对齐 |
| [train_cross_subject_contrast.py](../code/train_cross_subject_contrast.py) | 跨受试者对比学习探索 |
| [train_subject_weighted.py](../code/train_subject_weighted.py) | 受试者分类损失加权探索 |
| [quick_screen.py](../code/quick_screen.py) | 小规模快速筛选，对照未充分学习时不能判模块无效 |

## 诊断与分析

| 文件 | 简要说明 |
|---|---|
| [audit_branches.py](../code/audit_branches.py) | 冻结时序／CSP分支的信息分析 |
| [audit_representation.py](../code/audit_representation.py) | 冻结特征线性探针与泛化差距 |
| [compare_thresholds.py](../code/compare_thresholds.py) | 70%阈值跨越统计，核对输入属于验证还是测试 |
| [diagnose_seed42.py](../code/diagnose_seed42.py) | seed42模型逐人诊断；也被其他诊断脚本导入 |
| [inspect_training_steps.py](../code/inspect_training_steps.py) | 优化器计数、AMP跳步和非有限值检查 |
| [probe_augmentation.py](../code/probe_augmentation.py) | 预测对增强的敏感性，不进行重新训练 |
| [probe_centering.py](../code/probe_centering.py) | 中心化预测诊断 |
| [probe_gradients.py](../code/probe_gradients.py) | 损失梯度大小与方向诊断 |
| [probe_running_center.py](../code/probe_running_center.py) | 运行式中心化诊断 |
| [profile_session1_eeg.py](../code/profile_session1_eeg.py) | session1 EEG统计特征分析 |

## 核心依赖与批量实验

| 路径 | 简要说明 |
|---|---|
| [parallel_experiments_v1/README_使用说明.md](../code/parallel_experiments_v1/README_使用说明.md) | 队列入口及准备数据依赖；历史候选，不是正式三次配对入口 |
| [parallel_experiments_v1/compare_thresholds.py](../code/parallel_experiments_v1/compare_thresholds.py) | 队列入口及准备数据依赖；历史候选，不是正式三次配对入口 |
| [parallel_experiments_v1/launch_experiments.py](../code/parallel_experiments_v1/launch_experiments.py) | 队列入口及准备数据依赖；历史候选，不是正式三次配对入口 |
| [parallel_experiments_v1/shared_data.py](../code/parallel_experiments_v1/shared_data.py) | 队列入口及准备数据依赖；历史候选，不是正式三次配对入口 |
| [parallel_experiments_v1/train_recipe.py](../code/parallel_experiments_v1/train_recipe.py) | 队列入口及准备数据依赖；历史候选，不是正式三次配对入口 |
| [parallel_experiments_v1/train_subject_weighted.py](../code/parallel_experiments_v1/train_subject_weighted.py) | 队列入口及准备数据依赖；历史候选，不是正式三次配对入口 |
| [parallel_experiments_v1/候选清单.json](../code/parallel_experiments_v1/候选清单.json) | 队列入口及准备数据依赖；历史候选，不是正式三次配对入口 |
| [src/plan_a/__init__.py](../code/src/plan_a/__init__.py) | 包标记 |
| [src/plan_a/data.py](../code/src/plan_a/data.py) | 数据读取、滤波、缓存与11通道协议 |
| [src/plan_a/evaluate.py](../code/src/plan_a/evaluate.py) | 固定模型评估 |
| [src/plan_a/inference.py](../code/src/plan_a/inference.py) | 推理接口 |
| [src/plan_a/model.py](../code/src/plan_a/model.py) | 模型结构；方案A含学生、教师及梯度反转 |
| [src/plan_a/requirements.txt](../code/src/plan_a/requirements.txt) | 该模块依赖约束 |
| [src/plan_a/train.py](../code/src/plan_a/train.py) | 训练与分组、蒸馏、验证逻辑 |
| [src/plan_b/__init__.py](../code/src/plan_b/__init__.py) | 包标记 |
| [src/plan_b/augment.py](../code/src/plan_b/augment.py) | 方案B增强 |
| [src/plan_b/evaluate.py](../code/src/plan_b/evaluate.py) | 固定模型评估 |
| [src/plan_b/inference.py](../code/src/plan_b/inference.py) | 推理接口 |
| [src/plan_b/model.py](../code/src/plan_b/model.py) | 模型结构；方案A含学生、教师及梯度反转 |
| [src/plan_b/requirements.txt](../code/src/plan_b/requirements.txt) | 该模块依赖约束 |
| [src/plan_b/train.py](../code/src/plan_b/train.py) | 训练与分组、蒸馏、验证逻辑 |
| [src/tools/compare_plan_ab.py](../code/src/tools/compare_plan_ab.py) | A/B结果比较工具 |

## 路径和历史文档更正

- `evaluate_original_pair.py` 实际测试 `formal_a_001` 原基线和 `plan_a_domain003_001`，不是warm配对模型；以源码为准。
- `docs/` 三份已有文档保留提交时内容，里面的旧仓库根目录路径在本包对应 `code/`；旧文档中的“暂不移动”描述属于之前阶段。
- 服务器绝对路径、检查点与结果目录以实际参数和配置为准，不保证当前服务器仍存在。
- 逐文件来源及嵌套候选副本差异见 `manifest.json`；旧资源去向见 `reports/repository_inventory.csv`，删除状态见 `cleanup_action`。

## 保留的OpenBMI历史分组脚本

`code/evaluate/bci_illiteracy_evaluate.py`：用户的MOABB/OpenBMI逐人CSP+LDA分析，保留溯源。该脚本有绝对路径和顶层数据操作，不直接导入；它不等于正式session1-only分组或联合CSP+LDA测试。
