> 最新状态：用户要求保留原在线模型和小车通信。相关13个文件已恢复；团队入口见仓库根目录 `hardware_legacy/README.md`。下文删除描述属于首次清理历史，以manifest的最新cleanup_action为准。

# SDAD-Net 团队整合包

**先看本页，再看 [文件索引](docs/file_index.md)。** 本文件夹集中你的SDAD-Net正式实验、历史探索、诊断工具和必要依赖，供后续接入EEGUnveil。

## 文件夹里有什么

| 位置 | 简要说明 |
|---|---|
| [code/](code/) | 完整运行目录，正式入口、历史实验及其依赖；从这里运行 |
| [docs/file_index.md](docs/file_index.md) | 按职责列出的代码清单，每个文件一句话说明 |
| [docs/experiment_index.md](docs/experiment_index.md) | 已提交的实验索引快照：结果、协议与实验定位 |
| [docs/README.md](docs/README.md) | 已提交的方法说明快照 |
| [docs/artifacts.md](docs/artifacts.md) | 已提交的外部数据和结果资源说明快照 |
| [docs/merge_guide.md](docs/merge_guide.md) | 与Xu整合时的边界、步骤和接口核对 |
| [reports/](reports/) | 检查报告与原始资源清单；没有伪造或补算实验结果 |
| [manifest.json](manifest.json) | 原路径、新路径、来源版本及SHA256校验值 |
| [verify_package.py](verify_package.py) | 只检查代码完整性和语法，不启动训练 |

## 正式运行入口

| 运行 | 训练 | 测试 |
|---|---|---|
| 预训练42／后期42 | `repeat_group_domain42.py` | `evaluate_group_domain42.py` |
| 预训练43／后期43 | `late_subject_domain.py` | `evaluate_subject_domain.py` |
| 预训练43／后期44 | `repeat_group_domain44.py` | `evaluate_group_domain44.py` |
| 传统CSP+LDA参考 | `compare_pooled_csp_lda.py` | 同一脚本 |

43／44共享43的预训练起点。`control`是已有蒸馏的无对抗对照，不是团队第一代无蒸馏模型。完整方法相对CSP+LDA的提升与DANN自身的增益分别评价。

## 如何运行

先进入本文件夹下的 `code/`，再执行原脚本。例如在仓库根目录：

```bash
cd sdad_integration/code
python late_subject_domain.py --help
python -m src.plan_a.train --help
```

在已有GPU环境中，先确认PyTorch可用，再按实际需要安装 `code/src/plan_a/requirements.txt` 中的依赖。不要直接用旧工程的全量冻结依赖覆盖服务器环境。

历史脚本仍保留原来的服务器默认路径。换服务器时先查看`--help`，核对数据、初始化检查点、准备数据及输出位置；部分诊断脚本存在硬编码路径，见文件索引。已有结果加载还会校验源码指纹，失败时应核对原环境／原文件字节，不应绕过校验。

## 整理方式

- 这是独立整合副本，用户的SDAD研究代码和历史实验保留；经用户要求，原小车专用内容已从本地工作树移除。
- 为保留互相导入及文件指纹，原脚本和核心依赖按原相对结构集中存放，未改算法和源码字节。
- 正式、历史、诊断通过索引分类，避免移动到不同子目录后破坏加载。
- 原数据、SDAD权重、服务器实验结果不在此次代码克隆中；资源清单不代表已打包。
- 原小车模型、GUI、通信、旧训练代码、日志、提示时序和演示素材已移除，见 [删除记录](reports/cleanup_log.md)。Git历史保留。
- `evaluate/bci_illiteracy_evaluate.py` 属于用户OpenBMI研究，已保留到 `code/evaluate/`；它依赖MOABB等额外库且执行时会读数据，不纳入帮助入口检查。
- 整合完成后，可在团队分支选择此文件夹作为SDAD部分，原副本作为迁移前对照，不长期维护两份活动代码。

## 快速检查

```bash
python sdad_integration/verify_package.py
```

该检查不依赖PyTorch、不读取EEG、不训练模型。更完整的导入检查结果见 `reports/validation.md`。
