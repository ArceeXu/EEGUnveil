> 最新GUI状态：已撤销自定义模型适配，src/online恢复Xu原提交。GUI使用Xu原模型路径；SDAD和硬件链路独立运行。此前有关适配器的描述属于历史记录，见gui_original.md。

> 最新恢复状态：原v50pro权重已恢复，TCP收发与串口小车控制已保留到 `hardware_legacy/`。此前有关旧权重删除或禁止使用的描述是上一阶段记录；其只是不适用于OpenBMI选择器，硬件专用入口可使用。

# 整理与团队整合修改总表

日期：2026-10-27。基于当前本地Git工作树核对。

两处目录分别管理：`sdad-cleanup`是用户仓库整理副本，`EEGUnveil-integration`是以Xu dev_arcee为基础的团队整合副本。

**状态变化提醒：**团队副本当前有4个旧v50pro文件删除，及旧四分类加载函数的额外清理。这与先前“团队没有删除”的状态不同；本表按当前实际文件列出，不把所有差异都归因于本轮清单操作。本轮只生成说明，没有再次删除或恢复文件。

## 一、按类别汇总

| 副本 | 操作 | 类别 | 内容 |
|---|---|---|---|
| 用户整理副本 | 删除 | 原小车专用内容 | 共73个跟踪文件：旧模型、GUI/通信、训练、日志、时序、演示及打包资源 |
| 用户整理副本 | 修改 | 首页、依赖 | README改为SDAD说明，requirements改为核心依赖引用 |
| 用户整理副本 | 新增 | SDAD整合包 | 正式入口、历史实验、诊断、文档、文件去向和完整性检查 |
| 团队整合副本 | 新增 | 方法入口 | project.py统一三代方法与GUI调用 |
| 团队整合副本 | 新增 | SDAD实现收录 | sdad_integration/；原训练源码保持字节 |
| 团队整合副本 | 新增/修改 | GUI适配 | 模型注册器、SDAD适配器和Xu GUI的加载分流；布局和控件保留 |
| 团队整合副本 | 修改 | 校验和测试 | 数据边界校验；更新过时测试并增加适配测试 |
| 团队整合副本 | 修改 | 文档 | 三代说明、协议对照、来源、结果边界及已有文档更正 |
| 团队整合副本 | 删除（当前检测） | 旧四分类权重 | models/online/v50pro/中4个文件；不包括Xu 62通道混合模型 |
| 团队整合副本 | 修改（当前检测） | 旧四分类加载 | 删除旧GDF路径及3个未使用的旧加载函数 |
| 团队整合副本 | 保留 | Xu研究资源 | 62通道模型、OpenBMI训练主体、分组、结果、GUI和5个sh脚本 |

## 二、团队整合副本逐文件表

以下路径相对于 `EEGUnveil-integration/`。新增包含“收录已有文件”和“新写的适配代码”，两者在说明中区分。

| 操作 | 文件 | 说明 |
|---|---|---|
| 修改 | `README.md` | 项目首页改为三代方法导航；Xu原首页另存保留。 |
| 修改 | `docs/openbmi_mixed_training.md` | 修正文档对旧四分类评估入口的说明。 |
| 修改 | `docs/openbmi_split.md` | 修正当前OpenBMI入口与旧四分类资源清理状态的说明。 |
| 删除 | `models/online/v50pro/best_csp.pkl` | 当前工作树已删除的旧模型资源；Git历史仍保留。 |
| 删除 | `models/online/v50pro/best_model.pth` | 当前工作树已删除的旧模型资源；Git历史仍保留。 |
| 删除 | `models/online/v50pro/best_scalers.pkl` | 当前工作树已删除的旧模型资源；Git历史仍保留。 |
| 删除 | `models/online/v50pro/model_config.json` | 当前工作树已删除的旧模型资源；Git历史仍保留。 |
| 修改 | `src/offline/train_v65_distill.py` | 当前额外变更：删除旧DATA_DIR及find_event_map、load_bci_iv2a、load_all_subjects；OpenBMI训练主体保留。 |
| 修改 | `src/online/main_app.py` | 保留Xu界面；修改模型创建、MAT加载分流，增加SDAD网络模式检查。 |
| 修改 | `src/tools/split_openbmi.py` | 补充试次边界、空事件及标签数量检查。 |
| 修改 | `tests/test_group_openbmi_csp.py` | 对齐现有process_subject双session接口，保留训练折隔离和不读取test的检查。 |
| 新增／收录 | `.gitignore` | 新增运行缓存、数据及生成文件忽略规则。 |
| 新增／收录 | `docs/integration/README.md` | 团队整合、协议、来源、审查或资源说明。 |
| 新增／收录 | `docs/integration/Xu_README_original.md` | 团队整合、协议、来源、审查或资源说明。 |
| 新增／收录 | `docs/integration/checks/check_1.log` | 检查日志或结构化检查结果。 |
| 新增／收录 | `docs/integration/checks/check_2.log` | 检查日志或结构化检查结果。 |
| 新增／收录 | `docs/integration/checks/check_3.log` | 检查日志或结构化检查结果。 |
| 新增／收录 | `docs/integration/checks/check_4.log` | 检查日志或结构化检查结果。 |
| 新增／收录 | `docs/integration/checks/check_5.log` | 检查日志或结构化检查结果。 |
| 新增／收录 | `docs/integration/checks/check_6.log` | 检查日志或结构化检查结果。 |
| 新增／收录 | `docs/integration/checks/final_integrity.json` | 检查日志或结构化检查结果。 |
| 新增／收录 | `docs/integration/checks/gui_artifact.log` | 检查日志或结构化检查结果。 |
| 新增／收录 | `docs/integration/checks/results.json` | 检查日志或结构化检查结果。 |
| 新增／收录 | `docs/integration/cleanup.json` | 当前清理记录，包含本次核对发现的后续清理信息。 |
| 新增／收录 | `docs/integration/protocols.md` | 团队整合、协议、来源、审查或资源说明。 |
| 新增／收录 | `docs/integration/review.md` | 团队整合、协议、来源、审查或资源说明。 |
| 新增／收录 | `docs/integration/shell_scripts.md` | 5个Xu批处理启动脚本的用途说明；脚本本身保留。 |
| 新增／收录 | `docs/integration/sources.json` | 团队整合、协议、来源、审查或资源说明。 |
| 新增／收录 | `docs/integration/validation.md` | 团队整合、协议、来源、审查或资源说明。 |
| 新增／收录 | `integration/__init__.py` | 整合适配包标记。 |
| 新增／收录 | `integration/sdad_backend.py` | 新增SDAD学生模型、原始因果预处理和MAT回放适配。 |
| 新增／收录 | `project.py` | 新增统一方法入口，按各自工作目录启动原实现。 |
| 新增／收录 | `requirements.txt` | 新增团队功能依赖清单；不是正式复现环境锁定文件。 |
| 新增／收录 | `sdad_integration/.gitignore` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/README.md` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/code/README.md` | 运行目录简要说明。 |
| 新增／收录 | `sdad_integration/code/audit_branches.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/audit_representation.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/compare_pooled_csp_lda.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/compare_thresholds.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/diagnose_seed42.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/evaluate/bci_illiteracy_evaluate.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/evaluate_group_domain42.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/evaluate_group_domain44.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/evaluate_late_classwise.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/evaluate_original_pair.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/evaluate_subject_domain.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/inspect_training_steps.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/late_classwise_dann.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/late_subject_domain.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/README_使用说明.md` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/compare_thresholds.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/launch_experiments.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/parallel_experiments_v1/README_使用说明.md` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/parallel_experiments_v1/compare_thresholds.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/parallel_experiments_v1/launch_experiments.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/parallel_experiments_v1/shared_data.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/parallel_experiments_v1/train_recipe.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/parallel_experiments_v1/train_subject_weighted.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/parallel_experiments_v1/候选清单.json` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/shared_data.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/train_recipe.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/train_subject_weighted.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/parallel_experiments_v1/候选清单.json` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/probe_augmentation.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/probe_centering.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/probe_gradients.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/probe_running_center.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/profile_session1_eeg.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/quick_screen.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/repeat_dann_pair.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/repeat_group_domain42.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/repeat_group_domain44.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/repeat_warm_dann_seed43.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_a/__init__.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_a/data.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_a/evaluate.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_a/inference.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_a/model.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_a/requirements.txt` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_a/train.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_b/__init__.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_b/augment.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_b/evaluate.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_b/inference.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_b/model.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_b/requirements.txt` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/plan_b/train.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/src/tools/compare_plan_ab.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/train_alignment.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/train_classwise_original.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/train_cross_subject_contrast.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/train_subject_weighted.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/code/warm_dann_pair.py` | 从用户研究副本收录的原代码／配置，不是本次新发明的算法。 |
| 新增／收录 | `sdad_integration/docs/README.md` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/docs/artifacts.md` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/docs/experiment_index.md` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/docs/file_index.md` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/docs/merge_guide.md` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/manifest.json` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/reports/README.md` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/reports/cleanup_log.md` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/reports/import_checks.json` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/reports/repository_inventory.csv` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/reports/validation.md` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `sdad_integration/verify_package.py` | SDAD整合包中的文档、来源清单或检查材料。 |
| 新增／收录 | `src/online/model_registry.py` | 新增Xu/SDAD配置识别、模型分流及旧格式拒绝。 |
| 新增／收录 | `tests/test_integration_adapter.py` | 新增推理一致性、MAT预处理一致性和接口边界测试。 |

## 三、用户整理副本的原文件变更

新增整合包与团队副本中的SDAD收录内容对应；此处列原有文件的修改／删除。

| 操作 | 文件 |
|---|---|
| 修改 | `README.md` |
| 删除 | `archive/cue_timeline_backup/cue_timeline0tt.txt` |
| 删除 | `archive/cue_timeline_backup/cue_timeline1.txt` |
| 删除 | `archive/cue_timeline_backup/cue_timeline2.txt` |
| 删除 | `archive/cue_timeline_backup/cue_timeline3.txt` |
| 删除 | `archive/cue_timeline_backup/cue_timeline3train.txt` |
| 删除 | `archive/cue_timeline_backup/cue_timeline4.txt` |
| 删除 | `archive/cue_timeline_backup/cue_timeline5.txt` |
| 删除 | `archive/cue_timeline_backup/cue_timeline6.txt` |
| 删除 | `archive/cue_timeline_backup/cue_timeline7.txt` |
| 删除 | `archive/cue_timeline_backup/cue_timeline8.txt` |
| 删除 | `archive/cue_timeline_backup/cue_timeline9.txt` |
| 删除 | `archive/cue_timeline_backup/txt notebook.py` |
| 删除 | `assets/pyinstaller/main_app.spec` |
| 删除 | `assets/slides/acc_data.pptx` |
| 删除 | `models/offline/v42/best_csp.pkl` |
| 删除 | `models/offline/v42/best_model.pth` |
| 删除 | `models/offline/v42/best_scalers.pkl` |
| 删除 | `models/offline/v42/fold_1_curves.png` |
| 删除 | `models/offline/v42/meta.json` |
| 删除 | `models/offline/v42/training_metrics.json` |
| 删除 | `models/offline/v65/best_csp.pkl` |
| 删除 | `models/offline/v65/best_model.pth` |
| 删除 | `models/offline/v65/best_scalers.pkl` |
| 删除 | `models/offline/v65/model_config.json` |
| 删除 | `models/online/v50pro/best_csp.pkl` |
| 删除 | `models/online/v50pro/best_model.pth` |
| 删除 | `models/online/v50pro/best_scalers.pkl` |
| 删除 | `models/online/v50pro/model_config.json` |
| 删除 | `outputs/logs/online_accuracy/Ee1.txt` |
| 删除 | `outputs/logs/online_accuracy/Ee2.txt` |
| 删除 | `outputs/logs/online_accuracy/Ee3.txt` |
| 删除 | `outputs/logs/online_accuracy/Ee4.txt` |
| 删除 | `outputs/logs/online_accuracy/Ee5.txt` |
| 删除 | `outputs/logs/online_accuracy/Ee6.txt` |
| 删除 | `outputs/logs/online_accuracy/Ee7.txt` |
| 删除 | `outputs/logs/online_accuracy/Ee8.txt` |
| 删除 | `outputs/logs/online_accuracy/Ee9.txt` |
| 删除 | `outputs/logs/online_accuracy/e1.txt` |
| 删除 | `outputs/logs/online_accuracy/e2.txt` |
| 删除 | `outputs/logs/online_accuracy/e3.txt` |
| 删除 | `outputs/logs/online_accuracy/e4.txt` |
| 删除 | `outputs/logs/online_accuracy/e5.txt` |
| 删除 | `outputs/logs/online_accuracy/e6.txt` |
| 删除 | `outputs/logs/online_accuracy/e7.txt` |
| 删除 | `outputs/logs/online_accuracy/e8.txt` |
| 删除 | `outputs/logs/online_accuracy/e9.txt` |
| 删除 | `outputs/logs/runtime/onlinev50prodata.txt` |
| 删除 | `outputs/logs/runtime/sequence.txt` |
| 删除 | `outputs/logs/runtime/v65data.txt` |
| 删除 | `outputs/timelines/cue_timeline0tt.txt` |
| 删除 | `outputs/timelines/cue_timeline1.txt` |
| 删除 | `outputs/timelines/cue_timeline2.txt` |
| 删除 | `outputs/timelines/cue_timeline3.txt` |
| 删除 | `outputs/timelines/cue_timeline3train.txt` |
| 删除 | `outputs/timelines/cue_timeline4.txt` |
| 删除 | `outputs/timelines/cue_timeline5.txt` |
| 删除 | `outputs/timelines/cue_timeline6.txt` |
| 删除 | `outputs/timelines/cue_timeline7.txt` |
| 删除 | `outputs/timelines/cue_timeline8.txt` |
| 删除 | `outputs/timelines/cue_timeline9.txt` |
| 修改 | `requirements.txt` |
| 删除 | `src/offline/evaluate_v65_offline.py` |
| 删除 | `src/offline/train_v42.py` |
| 删除 | `src/offline/train_v65_distill.py` |
| 删除 | `src/online/debug_dll_load.py` |
| 删除 | `src/online/evaluate_v50pro_stream.py` |
| 删除 | `src/online/main_app.py` |
| 删除 | `src/online/receive_control_utf8.py` |
| 删除 | `src/online/receiver.py` |
| 删除 | `src/online/sender.py` |
| 删除 | `src/online/torch_bootstrap.py` |
| 删除 | `src/online/train_v50pro_online.py` |
| 删除 | `src/tools/compare_logs.py` |
| 删除 | `src/tools/generate_cue_timeline.py` |

## 四、未进行的操作

- 未向GitHub推送、未创建PR或合并远程分支。
- 未对真实EEG重新训练、重选检查点或重新测量准确率。
- 未删除用户其他目录中的EEG、服务器SDAD权重或本地诊断记录。
- 未将Xu权重转成SDAD权重；两者通道与协议不同。
- 未做真实SDAD回放验收；等待best.pt、config.json及preprocessing.joblib。

本报告自身是新增说明文件。两处仓库的Git历史均未被重写。
