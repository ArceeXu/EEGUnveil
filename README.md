# EEGUnveil：面向BCI盲的脑控康复训练辅助系统

EEGUnveil基于OpenBMI运动想象脑电数据，研究如何改善低表现者的左右手意图识别，并使用脑控小车的在线推理、信号收发与串口控制链路，为交互式辅助训练提供工程基础。

项目按三代方法组织：**正常组基础训练 → 混合训练与知识蒸馏 → 后期域对抗蒸馏SDAD-Net**。仓库同时提供GUI、实验评估工具、历史消融与诊断代码。

> “BCI盲组／低表现组”是按规定解码表现和70%阈值定义的实验分组，不是医学诊断。当前结果来自公开数据实验，不代表已验证康复疗效。

<!-- README_NAV_START -->
## 内容导航

- [1. 项目组成与三代方法](#project-methods)
- [2. 项目目录结构](#project-structure)
- [3. 环境与快速启动](#setup-and-start)
  - [启动GUI](#start-gui)
  - [OpenBMI网络回放](#network-replay)
- [4. 数据准备与评估协议](#data-and-protocols)
- [5. Gen1 / Gen2训练、测试与对照](#gen1-gen2-training)
- [6. SDAD-Net：后期配对实验](#sdad-experiments)
- [7. 已有实验结果](#experiment-results)
- [8. 小车在线模型与通信](#car-communication)
- [9. 批量实验、结果与历史探索](#batch-and-history)
<!-- README_NAV_END -->

<a id="project-methods"></a>

## 1. 项目组成与三代方法

| 部分 | 核心思路 | 实现位置 |
|---|---|---|
| **Gen1：基础模型** | 仅正常组参与参数训练，不使用蒸馏，建立基础对照 | `src/offline/`，对应N0 |
| **Gen2：混合蒸馏** | 正常组与低表现组按固定比例混合，结合时空特征与同步教师—学生蒸馏 | `src/offline/`，对应M1 |
| **Gen3：SDAD-Net** | 保留时空融合与蒸馏，在基础分类能力形成后加入高低组域对抗 | `sdad_integration/code/` |
| **GUI展示与回放** | 中文界面、本地MAT回放、网络试次回放、脑电波形与分类结果展示 | `src/online/` |
| **小车硬件链路** | v50在线模型、TCP分块收发、串口指令与控制逻辑 | `hardware_legacy/` |

SDAD-Net全称为 **Stage-wise Domain-Adversarial Distillation Network（分阶段域对抗蒸馏网络）**。训练时保留教师与辅助判别器，学生模型承担推理。

<a id="project-structure"></a>

## 2. 目录导航

```text
.
├── project.py                         # 统一命令入口，自动切换各模块工作目录
├── requirements.txt                   # 核心功能依赖
├── src/
│   ├── offline/                       # 模型、混合蒸馏、留受试者训练与λ搜索
│   ├── online/                        # GUI、OpenBMI推理和MAT网络回放
│   └── tools/                         # 数据划分、CSP分组和结果绘图
├── sdad_integration/
│   ├── code/                          # 后期DANN、A/B探索、消融与诊断
│   │   ├── src/plan_a/                # SDAD历史模型、蒸馏与预处理依赖
│   │   ├── src/plan_b/                # 对比学习方案
│   │   └── parallel_experiments_v1/   # 候选实验队列
├── hardware_legacy/                   # 在线模型、小车通信与参数化启动入口
├── models/offline/lambda_analysis/     # λ=0.5625混合蒸馏模型
├── models/online/v50pro/               # 在线模型
├── outputs/
│   ├── splits/                        # 固定受试者划分
│   ├── export/                        # 分组报告与原服务器依赖快照
│   └── figures/                       # 已归档的指标、图表和部分结果记录
├── tests/                             # 数据隔离、训练逻辑、原版GUI及通信检查
└── run_*.sh                           # 服务器批量实验脚本
```

<a id="setup-and-start"></a>

## 3. 环境与快速启动

主要依赖：PyTorch、NumPy、SciPy、scikit-learn、MNE、joblib、pandas、matplotlib；GUI还需要Python/Conda提供的Tkinter。

已有环境时先确认PyTorch及GPU状态：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

按需补充常用库：

```bash
python -m pip install numpy scipy scikit-learn mne joblib pandas matplotlib
```

<a id="start-gui"></a>

### 启动GUI

以下命令在仓库根目录执行：

```bash
python project.py run gui
```

也可直接执行 `python src/online/main_app.py`。操作步骤：

1. 选择配套模型目录，默认是`models/offline/lambda_analysis/lambda0.5625_seed42/`。
2. 选择原始OpenBMI `.mat` 文件，以及文件中的`train`或`test`阶段。
3. 选择“本地文件回放”，点击“开始回放”；首次整段滤波需要等待。
4. 使用“字体＋／－”调整字号，F11全屏，Esc退出全屏。

<a id="network-replay"></a>

### OpenBMI网络回放

在另一终端运行发送端，再在GUI中选择网络回放和相同地址、端口：

```bash
python src/online/sender.py --data /absolute/path/to/sess02_subj02_EEG_MI.mat --phase test --host 127.0.0.1 --port 65432
```

<a id="data-and-protocols"></a>

## 4. 数据准备与评估协议

### Gen1 / Gen2：固定留受试者实验

建议数据目录：

```text
Data/OpenBMI/raw/                       # MAT
Data/OpenBMI/splits/official_250hz_full/ # 导出后的官方train/test阶段
Data/OpenBMI/groups/csp70_20/            # CSP分组导出与report.json
```

导出试次并进行分组：

```bash
python src/tools/split_openbmi.py --data-dir Data/OpenBMI/raw --output-dir Data/OpenBMI/splits/official_250hz_full --mode official --tmin 0 --tmax 4 --sfreq 250
python src/tools/group_openbmi_csp.py --split-dir Data/OpenBMI/splits/official_250hz_full --output-dir Data/OpenBMI/groups/csp70_20 --threshold 0.70
```

当前混合训练实验的分组依据是两个session的**训练阶段**各自进行CSP+LDA交叉验证，再对准确率取均值；评分使用20个运动区通道。导出数据供网络使用的通道与分组评分通道不同，保存模型输入为62通道。受试者划分为32训练、11验证、11测试，固定划分种子2026。

复现实验需使用对应`report.json`及`outputs/splits/openbmi_subject_holdout.json`，不能任意重新生成名单后仍视为同一实验。
### SDAD-Net：固定受试者跨会话实验

- 54名受试者；依据session1训练数据固定25名低表现者。
- session1的`EEG_MI_train`用于训练，`EEG_MI_test`用于验证。
- session2的`EEG_MI_test`用于测试；达到70%的人数与均值共同报告。
- 使用11通道、4秒／1000点输入及原因果预处理。

<a id="gen1-gen2-training"></a>

## 5. Gen1 / Gen2训练、测试与对照

混合训练的四组对照使用相同留受试者划分：

| 方法 | 参数训练人群 | 采样 | 蒸馏 | 用途 |
|---|---|---|---|---|
| N0 | 仅正常组 | natural | 无 | 第一代基础模型 |
| N1 | 仅正常组 | natural | 有 | 正常组蒸馏消融 |
| M0 | 正常组＋低表现组 | mixed | 无 | 混合训练消融 |
| M1 | 正常组＋低表现组 | mixed | 有 | 第二代混合蒸馏 |

λ表示混合批次中正常组样本的占比。当前候选为0.5625、0.625、0.75、0.875、0.9375，按验证集两组等权指标选中0.5625；batch size=32时对应18个正常组样本、14个低表现组样本。它不是全局最优比例或真实人群占比。

### 训练M1示例

```bash
python project.py run gen2 -- --stage train --groups-dir Data/OpenBMI/groups/csp70_20 --data-root Data --split-file outputs/splits/openbmi_subject_holdout.json --split-seed 2026 --seed 42 --batch-size 32 --epochs 120 --num-workers 0 --output-dir models/offline/reproduce/M1_seed42
```

第一代将`gen2`改为`gen1`并使用独立的`N0_seed42`输出目录即可；统一入口已固定对应的采样、人群与蒸馏开关。N1、M0消融使用原`src/offline/run_openbmi_holdout.py`
### 测试同一输出目录

```bash
python src/offline/run_openbmi_holdout.py --stage test --groups-dir Data/OpenBMI/groups/csp70_20 --data-root Data --split-file outputs/splits/openbmi_subject_holdout.json --split-seed 2026 --batch-size 32 --num-workers 0 --output-dir models/offline/reproduce/M1_seed42
```

输出包括`test_metrics.json`、`subject_metrics.csv`和`test_predictions.csv`。CSP与标准化只在训练数据拟合；验证集选择参数和检查点，测试阶段不重新拟合。关闭蒸馏也会改变原实现中硬标签损失的系数，不能把差异全部解释为教师知识的单独作用。

<a id="sdad-experiments"></a>

## 6. SDAD-Net：后期配对实验

后期实验从同一学生、教师检查点构建`control`和`group`两个分支：前者继续蒸馏，后者在相同分类训练规则上加入高低组DANN。保持初始化、批次构造和训练预算匹配，依据验证集选定检查点后再测试。

| 运行 | 训练入口 | 测试入口 | 预训练起点 |
|---|---|---|---|
| 42／42 | `project.py run sdad42` | `project.py run test-sdad42` | seed42原蒸馏基线 |
| 43／43 | `project.py run sdad43` | `project.py run test-sdad43` | seed43原蒸馏基线 |
| 43／44 | `project.py run sdad44` | `project.py run test-sdad44` | 同一seed43基线，后期种子44 |

先查看参数：

```bash
python project.py run sdad43 -- --help
python project.py run test-sdad43 -- --help
```

服务器预检查示例（路径须对应实际保存的资源）：

```bash
python project.py run sdad43 -- --preflight --recipes control group --reference /mne_data/mne_data/bci_runs/dann_pair_43_44_001/online/baseline/seed_43 --prepared /mne_data/mne_data/bci_runs/parallel_remaining_001/_prepared --device cuda
```

确认预检查通过后，去掉`--preflight`并指定新的`--output-root`进行训练；测试用同次训练目录作为`--model-root`，另指定新的`--output-root`。`sdad43`默认还支持`subject`身份对抗分支，复现高低组配对时明确传入`--recipes control group`。

初始化目录需要配套的`last.pt`、配置和预处理，准备数据目录需要`train.pt`、`validation.pt`及`preprocessing.joblib`。

<a id="experiment-results"></a>

## 7. 已有实验结果

### 7.1 混合训练留受试者实验

在固定测试名单中的7名低表现者上，先按受试者等权平均，再对42、2024、3407三个训练种子平均：

| 方案 | 低表现组平均准确率 |
|---|---:|
| N0：正常组、无蒸馏 | 63.07% |
| N1：正常组、有蒸馏 | 57.26% |
| M0：混合、无蒸馏 | 70.88% |
| M1：混合、有蒸馏 | 70.52% |

### 7.2 SDAD跨会话实验记录

| 方法 | 全体均值 | 全体达标人数均值 | 固定25人低表现组均值 | 低表现组达标人数均值 |
|---|---:|---:|---:|---:|
| 联合CSP+LDA固定参考 | 57.11% | 7/54 | 53.92% | 1/25 |
| 后期DANN／SDAD-Net | **74.51%** | **36.00/54** | **72.77%** | **15.00/25** |

<a id="car-communication"></a>

## 8. 小车在线模型与通信

`hardware_legacy/`独立保留原在线链路，包含v50pro模型、CSP、标准化器、TCP分块收发、串口指令及刹车／转向控制。

先检查资源，不连接设备：

```bash
python project.py run hardware -- check
python project.py run hardware -- --help
```

| 入口 | 作用 | 运行前需要 |
|---|---|---|
| `hardware -- sender` | 从原GDF文件模拟发流 | `--data-dir`、受试者与监听地址 |
| `hardware -- receiver` | TCP接收、在线滤波、推理与波形显示 | 匹配的`--timeline`、服务端地址 |
| `hardware -- car` | 接收推理后通过串口控制小车 | 上述参数及实际`--serial-port` |

<a id="batch-and-history"></a>

## 9. 批量实验、结果与历史探索

- 根目录5个`run_*.sh`完整保留：覆盖四组对照、正常组消融、比例0.5625无蒸馏消融及已完成模型测试。
- SDAD包保留原方案A/B、类条件对齐、受试者加权、对比学习及其诊断工具；历史探索与正式配对结果分开
- `src/tools/`提供混合收益、逐人表现和λ验证绘图。
- 小规模筛选、增强敏感性和梯度分析只提供排查线索
