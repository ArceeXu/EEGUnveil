# OpenBMI 训练集的 CSP 70% 分组

本流程先保留每位受试者、每个 session 原有的训练与测试阶段，再仅依据训练阶段的 CSP+LDA 交叉验证准确率，对该 subject/session 的全部训练 trial 分组。这里的“BCI 盲”是给定算法与实验设置下的低表现标签，不是医学诊断或永久属性。

## 1. 导出原始训练与测试阶段

依赖：`numpy scipy mne scikit-learn`。本机已有 `bci-openbmi` conda 环境；先执行 `conda activate bci-openbmi`，然后在仓库根目录执行：

```powershell
python src/tools/split_openbmi.py --mode official --sfreq 250 --output-dir Data/OpenBMI/splits/official_250hz_full
```

`EEG_MI_train → train.npz`，`EEG_MI_test → test.npz`，不混合两个阶段。标准文件每阶段 100 trial，左右手各 50 个。默认提取 `[0,4)` 秒，每个 trial 独立抗混叠重采样。输出为 `(N,62,1000)`，单位伏，`y=0` 左手、`y=1` 右手。每个人、每个 session 分开保存，详细参数见 `openbmi_split.md`。

脚本处理实际下载的数据，不要求一定有全部 54 人；但默认要求每位已发现受试者的两个 session 齐全。目录已有输出时拒绝覆盖；部分数据可用 `--subjects 1 --sessions 1`。本机已有 `official_250hz` 目录，所以示例使用新目录，避免覆盖已有结果。

## 2. 仅对训练阶段分组

```powershell
python src/tools/group_openbmi_csp.py --split-dir Data/OpenBMI/splits/official_250hz_full --output-dir Data/OpenBMI/groups/csp70_full
```

固定的基线设置如下，它是本项目的可复现分组协议，不声称逐项复现 OpenBMI 论文：

- 使用导出训练集的全部通道、全部时间点；默认即 62 通道、0–4 秒。
- 每个 trial 独立应用 8–30 Hz 四阶 Butterworth 双向滤波，不跨 trial 拼接。该零相位处理只用于离线分组，不能当作在线因果滤波。
- 每个 subject/session 内做 5 折分层交叉验证，打乱顺序，固定 seed=42。
- 每折仅在 4/5 trial 上拟合 CSP（6 分量、协方差收缩 0.1、完整秩），提取对数平均功率，再拟合 LDA（lsqr，自动收缩）；预测剩余 1/5 trial。
- 汇总全部 trial 的折外预测正确率。它不是训练集回代准确率，也不是 CSP 自身的特征值或解释方差。
- 准确率 **<0.70** 为 `bci_blind`；**≥0.70** 为 `normal`，恰好 70% 属于 normal。以未四舍五入的准确率比较。

CSP 和 LDA 均在每折重新拟合。固定带通不估计跨 trial 参数，因此可先逐 trial 滤波。最终 `test.npz` 不被打开，代码也会拒绝随机混合两个阶段的输入。左右手标签 `y` 与人群标签是两种不同含义，不能互相替换。

输出：

```text
Data/OpenBMI/groups/csp70_full/
  report.json
  normal.csv
  bci_blind.csv
  normal/sub-XX/ses-YY/train.npz
  bci_blind/sub-XX/ses-YY/train.npz
```

每个 subject/session 只出现在一个组中，同一人在两个 session 中可以属于不同组。每组 CSV 列出成员、折外准确率、trial 数、来源和输出路径；空组也输出表头。`report.json` 记录各折准确率、参数、软件版本、输入文件 SHA256 和完成状态。只有 `status=complete` 才表示整批处理成功；中断后改用新输出目录重跑。

分组 NPZ 保留原始训练 EEG（不保存分组基线滤波后的信号）、所有原字段和左右手标签，并增加：

- `group_label`：每 trial 一个值，0=normal，1=bci_blind。
- `group_name`、`csp_cv_accuracy`：当前 subject/session 的组别与整体准确率。
- `csp_oof_prediction`、`csp_fold_id`：原 trial 顺序下的折外预测与验证折编号（0–4）。

```python
import csv
from pathlib import Path
import numpy as np

root = Path("Data/OpenBMI/groups/csp70_full")
with (root / "bci_blind.csv").open(encoding="utf-8-sig", newline="") as f:
    members = list(csv.DictReader(f))
for member in members:
    with np.load(root / member["grouped_train"], allow_pickle=False) as data:
        X, y = data["X"], data["y"]
        group_label = data["group_label"]
        # 将 X、y 加入该组的数据加载器；不要将 group_label 用作左右手目标。
```

参数可用 `--threshold 0.70 --folds 5 --seed 42 --components 6 --low 8 --high 30` 显式固定。修改基线会改变分组，应先确定协议再比较后续模型，不能反复查看测试结果选择分组参数。小样本下 70% 附近的归组可能不稳定，应同时报告准确率与分组协议。

## 3. 后续混合训练的含义

“正常组：BCI 盲组比例”应控制训练时两组 trial 的采样数量或概率，不是本脚本的 70% 阈值，也不是训练／测试划分比例。本次交付止于划分与分组，尚未修改深度学习训练器或加入比例采样参数。

测试阶段按对应 subject/session 的训练分组标签汇总，不能重新用测试准确率决定组别。模型选型、早停和比例搜索需要训练内部的验证方案；若在内部验证中使用组别进行采样调参，分组也应在相应外层训练部分重新估计，避免内部验证标签经分组间接泄漏。

当前按 session 独立校准，训练和测试包含同一批受试者，不能据此宣称对完全未见过的人泛化。若研究要求每人一个固定组别，应预先确定 session 的聚合或独立校准规则；本代码不以任意规则合并两次测量。现有四分类模型仍需适配 OpenBMI 的 62 通道与左右手二分类。

## 验证与参考

```powershell
python -m unittest discover -s tests -p "test_*openbmi*.py"
```

测试包含原始阶段隔离、阈值边界、每折拟合／验证 trial 不重叠、每 trial 恰好一次折外预测、拒绝混合阶段输入、输出保留原数据及真实 CSP 管线的合成数据测试。

本机验证：8 项测试通过；使用已有 `official_250hz/sub-01/ses-01/train.npz` 真实试跑，得到折外准确率 53%，归入 `bci_blind`。结果位于 `Data/OpenBMI/groups/csp70_smoke/report.json`，仅覆盖一个 subject/session，不代表全数据集统计。

- [OpenBMI 原始论文](https://doi.org/10.1093/gigascience/giz002)：数据阶段结构与 MI 70% 阈值的研究背景。
- [MNE CSP](https://mne.tools/stable/generated/mne.decoding.CSP.html)：CSP 参数与特征定义。
- [scikit-learn StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)：分层交叉验证。
