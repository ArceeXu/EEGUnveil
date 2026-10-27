# 运行目录

从此目录运行训练、测试及诊断脚本。脚本分组和逐文件说明见 [文件索引](../docs/file_index.md)。

- `src/plan_a/`：蒸馏及DANN历史核心，保留字节以兼容指纹。
- `src/plan_b/`：历史对比学习方案，不是最终SDAD-Net。
- `parallel_experiments_v1/`：历史候选队列及准备数据依赖。
- 根目录正式入口：`late_subject_domain.py`、`repeat_group_domain42.py`、`repeat_group_domain44.py`及对应评估。
- 其余根目录脚本：历史探索、消融或诊断；查看索引后使用。

嵌套的 `parallel_experiments_v1/parallel_experiments_v1/` 是历史副本，未删除；推荐使用上层队列入口。数据和权重必须外部提供，且输出使用新目录。
