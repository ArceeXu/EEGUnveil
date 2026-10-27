# SDAD实验简要索引

代码路径相对于 `../code/`；历史服务器目录相对于 `/mne_data/mne_data/bci_runs/`。

## 正式入口

| 实验 | 训练 | 测试 | 测试结果目录 |
|---|---|---|---|
| CSP+LDA参考 | `compare_pooled_csp_lda.py` | 同脚本 | `pooled_csp_lda_test_001/` |
| 起点42／后期42 | `repeat_group_domain42.py` | `evaluate_group_domain42.py` | `group_domain_repeat42_001_test/` |
| 起点43／后期43 | `late_subject_domain.py` | `evaluate_subject_domain.py` | `late_subject_domain_43_001_test/` |
| 起点43／后期44 | `repeat_group_domain44.py` | `evaluate_group_domain44.py` | `group_domain_source43_seed44_001_test/` |

control保留蒸馏而不加对抗；group加入后期高低组DANN；43起点还含subject身份对抗消融。三次运行只有两个预训练起点，不能称为三个独立预训练种子。

## 历史实验与诊断

- 后期普通／类条件对抗：`warm_dann_pair.py`、`repeat_warm_dann_seed43.py`、`late_classwise_dann.py`及对应评估。
- 原A/B、从头DANN、类条件对齐、加权、跨个体对比：保留实现和依赖，见[逐文件索引](file_index.md)。
- 批量候选：`parallel_experiments_v1/`；嵌套副本保留，优先上层入口。
- 诊断：增强、梯度、优化器计数、特征探针、中心化和EEG统计，不单独作为因果证明。
- 历史分组：`evaluate/bci_illiteracy_evaluate.py`，其协议与正式session1-only分组分开，不默认可替换固定名单。

## 结果边界

session1训练／验证，session2测试；54人中训练定义的25名低表现者固定，阈值准确率≥70%。完整方法相对传统算法的收益，与DANN相对匹配蒸馏对照的增益分别说明。

不将验证值写成测试值、不拼接不同运行的最优指标、不将多次查看的session2称为独立确认性验证。本包没有重新训练或补算结果；原小车专用模型和日志已移除，不混入SDAD实验。
