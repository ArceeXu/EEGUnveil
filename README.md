# OpenBMI 混合训练与知识蒸馏

基于 OpenBMI 运动想象脑电数据的左右手识别项目，包含受试者分组、离线训练、知识蒸馏、混合采样比例搜索、留出受试者评估及 GUI 数据回放。

项目重点研究：在训练阶段加入部分 BCI 盲组受试者的数据，是否能够改善模型对**未参与训练的 BCI 盲组受试者**的识别表现。

## 功能

- 基于训练阶段 CSP 分类表现划分正常组与 BCI 盲组。
- 按受试者划分训练、验证和测试集，避免同一受试者跨集合。
- 对比仅正常组训练与正常组／BCI 盲组混合训练。
- 支持有／无知识蒸馏对照和多个训练随机种子。
- 使用验证集选择混合采样比例 λ。
- 输出总体、分组及逐受试者测试指标。
- 提供中文 GUI，支持本地 MAT 回放、网络回放、字体调整和全屏显示。

“BCI 盲组”是本项目按既定分类表现阈值定义的实验分组，不是医学诊断。

## 目录

```text
.
├── README.md
├── src/
│   ├── offline/
│   │   ├── train_v65_distill.py       # 模型、采样和训练核心
│   │   ├── run_openbmi_holdout.py    # 留受试者训练／测试
│   │   ├── search_openbmi_lambda.py  # λ 搜索与汇总
│   │   ├── takeover_openbmi_lambda.py
│   │   └── upgrade_lambda_workers.py
│   ├── online/
│   │   ├── main_app.py              # 中文 GUI
│   │   ├── receiver.py              # 模型加载、预处理与推理
│   │   ├── sender.py                # MAT 试次网络发送
│   │   ├── torch_bootstrap.py
│   │   ├── evaluate_v50pro_stream.py # OpenBMI 单记录回放入口
│   │   └── train_v50pro_online.py    # 转到 OpenBMI 离线训练入口
│   └── tools/
│       ├── download_openbmi.py
│       ├── split_openbmi.py
│       ├── group_openbmi_csp.py
│       ├── plot_openbmi_holdout.py
│       └── plot_blind_mixing.py
├── models/offline/lambda_analysis/
│   └── lambda0.5625_seed42/          # 保留的一套模型与配套文件
├── outputs/
│   ├── splits/                     # 固定受试者划分
│   └── export/                     # 环境和分组记录
├── docs/
│   ├── openbmi_split.md
│   ├── openbmi_csp_groups.md
│   ├── openbmi_mixed_training.md
│   └── openbmi_gui.md
└── run_*.sh                        # 批量实验启动脚本
```

仓库不包含原始数据、实验日志及逐试次 CSV。运行训练、测试和绘图时会在本地生成相应文件。

## 环境

主要依赖：

```text
torch
numpy
scipy
scikit-learn
mne
joblib
pandas
matplotlib
tkinter（GUI）
```

使用与训练环境兼容的 PyTorch、MNE 和 scikit-learn 版本。若保留了
`outputs/export/requirements_server.txt`，可用它核对原服务器依赖；
该文件是服务器环境快照，不代表所有软件包都适合直接安装到 Windows。

已有可用 Python 环境时，可补充主要依赖：

```bash
python -m pip install numpy scipy scikit-learn mne joblib pandas matplotlib
```

PyTorch 需与操作系统及 CUDA 环境匹配。Tkinter 通常随 Windows Python／Conda 安装提供。
GUI 支持 CPU 推理；CUDA 可用时自动使用 GPU。

## 快速启动 GUI

以下命令均在仓库根目录执行：

```bash
python src/online/main_app.py
```

1. 选择模型文件夹，默认使用 `lambda0.5625_seed42`。
2. 选择原始 OpenBMI `.mat` 文件。
3. 选择“本地文件回放”，点击“开始回放”。
4. 使用“字体＋／－”调整字号，F11 全屏，Esc 退出全屏。

无需重新训练，也无需启动发送端。首次读取及整段滤波需要时间。

MAT 文件内部的 `train`／`test` 阶段，与受试者属于训练集／测试集是不同概念。
展示未见受试者时，应选择固定划分中 test 名单对应的数据。

网络回放、模型选择及界面说明见 [GUI 使用说明](docs/openbmi_gui.md)。

## 数据准备

自行准备原始 OpenBMI 数据，推荐目录：

```text
Data/OpenBMI/raw/
Data/OpenBMI/splits/official_250hz_full/
Data/OpenBMI/groups/csp70_20/
```

数据导出示例：

```bash
python src/tools/split_openbmi.py --data-dir Data/OpenBMI/raw --output-dir Data/OpenBMI/splits/official_250hz_full --mode official --tmin 0 --tmax 4 --sfreq 250
```

受试者分组示例：

```bash
python src/tools/group_openbmi_csp.py --split-dir Data/OpenBMI/splits/official_250hz_full --output-dir Data/OpenBMI/groups/csp70_20 --threshold 0.70
```

分组使用训练阶段数据的交叉验证表现。目录名中的 `csp70_20` 是本项目约定，
详细通道、时间窗、会话聚合和阈值定义见 [分组说明](docs/openbmi_csp_groups.md)。

已有固定划分时，请保留对应的分组报告与数据处理配置，不要在不同实验间重新生成不同划分。

## 实验设计

四组主对照均使用同一套受试者划分，并保留方差正则：

| 方法 | 参数训练所用人群 | 蒸馏 |
|---|---|---|
| N0 | 仅正常组 | 无 |
| N1 | 仅正常组 | 有 |
| M0 | 正常组＋BCI 盲组，λ 控制采样 | 无 |
| M1 | 正常组＋BCI 盲组，λ 控制采样 | 有 |

主要比较：

- **M0 − N0**：无蒸馏条件下，混合训练的收益。
- **M1 − N1**：有蒸馏条件下，混合训练的收益。
- **M1 − M0**：混合训练条件下，蒸馏配置的影响。
- **N1 − N0**：仅正常组训练条件下，蒸馏配置的影响。

λ 表示混合批次中正常组样本的占比，不是真实世界人群比例。
例如 batch size 为 32、λ=0.5625 时，每批为 18 个正常组样本和 14 个 BCI 盲组样本。

本轮候选 λ 为 0.5625、0.625、0.75、0.875、0.9375；
按两组等权的验证指标选择 λ=0.5625。它是**当前候选集合中的验证最优值**，
不代表所有可能比例中的全局最优值。

训练随机种子为 42、2024、3407，受试者划分种子为 2026。
关闭蒸馏时，硬标签损失的系数与启用蒸馏时不同；损失定义见
[训练说明](docs/openbmi_mixed_training.md)。

## 训练与测试

### 训练完整方法 M1

```bash
python src/offline/run_openbmi_holdout.py --stage train --groups-dir Data/OpenBMI/groups/csp70_20 --data-root Data --split-file outputs/splits/openbmi_subject_holdout.json --split-seed 2026 --train-population all --sampling mixed --mix-lambda 0.5625 --seed 42 --batch-size 32 --epochs 120 --num-workers 0 --output-dir models/offline/reproduce/M1_seed42
```

示例输出到新的 `reproduce` 目录，避免覆盖仓库附带模型。

在上述命令基础上调整实验参数：

| 方法 | 替换的训练参数 |
|---|---|
| N0 | `--train-population normal --sampling natural --no-distillation` |
| N1 | `--train-population normal --sampling natural` |
| M0 | `--train-population all --sampling mixed --mix-lambda 0.5625 --no-distillation` |
| M1 | `--train-population all --sampling mixed --mix-lambda 0.5625` |

每组使用独立输出目录。重复实验时分别设置 `--seed 42`、`2024`、`3407`。
仅正常组的 `natural` 表示遍历筛选后的正常组样本，不表示混入 BCI 盲组。

### 测试训练完成的模型

```bash
python src/offline/run_openbmi_holdout.py --stage test --groups-dir Data/OpenBMI/groups/csp70_20 --data-root Data --split-file outputs/splits/openbmi_subject_holdout.json --split-seed 2026 --batch-size 32 --num-workers 0 --output-dir models/offline/reproduce/M1_seed42
```

测试读取输出目录中的模型及配置，生成：

- `test_metrics.json`：总体和分组指标。
- `subject_metrics.csv`：逐受试者指标。
- `test_predictions.csv`：逐试次预测。

验证集用于选参数和模型，最终测试集用于评估。不要依据最终测试结果反复调整 λ。

根目录 `run_*.sh` 为服务器批量调度脚本，包含原服务器路径；
迁移后应先修改工作目录和数据路径。λ 搜索脚本同样包含历史默认路径及基线复用逻辑，
使用前查看其帮助和配置，不要直接重复启动多个调度器。

## 当前实验结果

在本轮固定划分的 7 名留出 BCI 盲组受试者上，先对受试者等权平均，
再汇总三个训练随机种子，得到：

| 训练方案 | BCI 盲组平均准确率 |
|---|---:|
| 仅正常组，无蒸馏 N0 | 63.07% |
| 仅正常组，有蒸馏 N1 | 57.26% |
| 混合训练，无蒸馏 M0 | 70.88% |
| 混合训练，有蒸馏 M1 | 70.52% |

混合训练＋蒸馏相对仅正常组＋蒸馏提高 **13.26 个百分点**；
相对仅正常组无蒸馏基线提高 **7.45 个百分点**。
无蒸馏条件下，混合训练相对对应基线提高 **7.81 个百分点**。

这些结果支持当前划分下混合训练对未见 BCI 盲组受试者的改善。
上述差值是描述性结果，不等于统计显著性结论。
三个训练种子反映训练随机性，不等同于三个独立受试者划分。

## 保留的模型

仓库仅保留完整方法 λ=0.5625、seed42 的一套模型。
该种子用于固定演示，不宣称它是三个种子中性能最好的模型。

```text
models/offline/lambda_analysis/lambda0.5625_seed42/
├── best_model.pth
├── best_csp.pkl
├── best_scalers.pkl
├── model_config.json
├── subject_split.json
├── validation_metrics.json
└── test_metrics.json
```

权重、CSP 和标准化器必须配套使用。GUI 直接读取这些文件，不在回放阶段重新拟合它们。
单套模型不能直接复现四种方法、三个种子的全部汇总结果；完整复现需要重新训练其余实验。

## 结果绘图

绘图代码保留在 `src/tools/`，主要图包括：

- BCI 盲组混合训练主对比。
- BCI 盲组逐受试者准确率。
- BCI 盲组配对收益。
- 四组方法总体及分组对比。
- λ 验证集参数分析。

`plot_blind_mixing.py` 读取合并后的逐受试者 CSV，字段包括
`subject_id, group, accuracy, method, seed`，准确率取值为 0–1，
方法标记为 N0／N1／M0／M1。当前脚本针对本轮固定受试者和三个种子绘制。

仓库不附带这些 CSV；需从完整实验结果重新生成或自行保留本地副本。
中文绘图需通过脚本的字体参数指定本机宋体文件；字体文件不随仓库分发。

## 功能边界

- 离线实验使用连续记录滤波后截取试次，再重采样至 250 Hz。
- GUI 沿用这套预处理，用于录制数据回放；不能据此宣称已实现实时设备推理。
- GUI 不连接串口小车，也不发送车辆控制命令。
- 在线目录中保留旧版本号的两个入口已经转为 OpenBMI 用途。
- 当前 GUI 改动尚未完成运行验证；不影响此前已经完成的离线实验结果。

## 相关说明

- [数据划分](docs/openbmi_split.md)
- [CSP 受试者分组](docs/openbmi_csp_groups.md)
- [混合训练与蒸馏](docs/openbmi_mixed_training.md)
- [GUI 使用方法](docs/openbmi_gui.md)

