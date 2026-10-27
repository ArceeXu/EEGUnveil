# OpenBMI 按受试者和 session 划分

脚本 `src/tools/split_openbmi.py` 递归查找下载目录中的 `sessXX_subjYY_EEG_MI.mat`，每个人、每个 session 单独输出训练集和测试集。依赖 `numpy`、`scipy`，无需 MOABB、MNE 或 PyTorch。以下命令在仓库根目录执行。

## 默认方案：保留实验原有训练／测试阶段

```powershell
python src/tools/split_openbmi.py
```

`EEG_MI_train` 导出为训练集，`EEG_MI_test` 导出为测试集。标准文件每个阶段各 100 个 trial，各含 50 个左手和 50 个右手 trial，因此为 1:1。脚本实际检查标签和 trial 数量，不硬编码数量、不丢弃异常 trial。这种方式保留阶段先后关系，更适合后续评价从训练阶段到测试阶段的表现。

默认读取本地已下载的全部受试者，并要求每个人的两个 session 齐全；不会自动下载缺失数据。指定 `--subjects 1 2 --sessions 1` 可只处理指定部分。如果发现同一人同一 session 有多个副本或缺少指定文件，会报错。

## 可选方案：每个 session 内按类别随机划分 8:2

```powershell
python src/tools/split_openbmi.py --mode random --test-size 0.2 --seed 42
```

此模式先合并同一个人的同一个 session 内两个阶段的 trial，再分别对左右手分层抽样。标准文件得到 160 个训练 trial、40 个测试 trial；每类各 80/20。每类测试数四舍五入，非标准 trial 数时实际比例可能略有不同；任一类别被分空时拒绝划分。随机种子由全局 seed、subject 和 session 共同确定，增加其他受试者不会改变已有受试者的划分。

随机划分混合了离线训练和在线反馈阶段，不能把结果解释为“只用离线数据训练，预测未来在线阶段”。两个方案应分别报告，不能混用测试结果。

## 时窗与采样率

默认取提示后 `[0, 4)` 秒，保留原始采样率。使用连续信号 `x` 和事件偏移 `t`，不直接依赖预分段的 `smt`。事件偏移直接作为数组下标，与 MOABB 实现一致；已在本地受试者 1/session 1 核对 `x[t:t+4000]` 和 `smt` 完全一致。

需要较小的输出可按 trial 独立抗混叠重采样至 250 Hz：

```powershell
python src/tools/split_openbmi.py --sfreq 250 --output-dir Data/OpenBMI/splits/official_250hz
```

可选 `--tmin 1 --tmax 3.5`。时窗必须在 0～4 秒内。不做带通、陷波、基线校正或标准化；仅提取 trial、将微伏转换为伏，以及可选的独立重采样。原始 MAT 不会修改。

## 输出及加载

```text
Data/OpenBMI/splits/official/
  sub-01/
    ses-01/
      train.npz
      test.npz
      metadata.json
    ses-02/...
  sub-02/...
```

`random` 模式默认放在 `splits/random`。默认原始采样率输出 `X.shape=(trial数, 62, 4000)`；250 Hz 时为 `(trial数, 62, 1000)`。磁盘占用取决于实际数据和压缩率，处理完整数据集前应预留空间。

```python
import numpy as np

with np.load("Data/OpenBMI/splits/official/sub-01/ses-01/train.npz",
             allow_pickle=False) as data:
    X_train = data["X"]   # float32，(N, C, T)，单位 V
    y_train = data["y"]   # int64，0=左手，1=右手
    subject_ids = data["subject_id"]
    session_ids = data["session_id"]
```

NPZ 同时保存 `channel_names`、`sfreq`、`original_sfreq`、`class_names`、`unit`、`source_run`、`trial_id`、`onset_sample` 和 `metadata_json`。`trial_id` 是原始阶段内从 0 开始的 trial 编号，`onset_sample` 是原始采样率下的事件位置。唯一 trial 标识应使用 `(subject_id, session_id, source_run, trial_id)`，不要只用 `trial_id`。

每个 session 的 `metadata.json` 记录源文件、划分配置、输出形状和类别计数。已有输出禁止覆盖；更改比例、时窗或采样率时请指定新的 `--output-dir`。若中断留下部分输出，也应改用新目录，防止无意改写已冻结测试集。

## 后续 BCI 盲研究的边界

已新增 `src/tools/group_openbmi_csp.py`，在官方训练阶段内部用 CSP+LDA 交叉验证准确率按 70% 阈值分组。完整两步命令、输出格式与实验边界见 [openbmi_csp_groups.md](openbmi_csp_groups.md)。以下“本步骤”指划分脚本本身。

- 本步骤只做 trial 划分，不自动标注 BCI 盲人群。`y` 是运动想象类别，不是人群标签；需要另建受试者或 subject/session 到人群的映射，并明确分组标准。
- “正常人：BCI 盲”的混合比例与 `--test-size` 是两回事。前者控制训练样本的组别组成，应仅作用于训练数据；测试集应固定，并分别报告两组表现。
- 如果根据解码准确率确定分组，应在训练数据内用交叉验证或独立校准数据完成，不能使用最终测试集。比例选择、模型选择、早停也需要从训练集内另划验证集。
- 标准化、CSP、特征选择等有拟合过程的操作仅在训练集拟合，再应用到测试集。滑窗扩增必须在 trial 划分后进行，同一个 trial 的所有窗口属于同一集合。
- 当前方案是受试者内、session 内评估。训练和测试都包含同一批人，不能证明对未见受试者或新 session 的泛化能力；此类结论需要额外的留人或留 session 实验。
- 现有仓库训练脚本读取 BCICIV2a、输出四类；本脚本输出 OpenBMI 左右手二分类数据。后续需修改数据加载、通道数及分类头，不能直接套用已有四分类模型。
- 当前下载脚本虽然 `--all` 的帮助文字写“54人”，实际循环为 `range(1, 11)`，最多下载前 10 人。本划分脚本处理实际存在的数据，不将其视为完整 54 人数据集。

格式参考：[MOABB Lee2019 源码](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/Lee2019.py)、[OpenBMI 原始论文](https://doi.org/10.1093/gigascience/giz002)。部分 MOABB 文档提示 MI 测试阶段无标签，本地已检查的文件实际包含 `y_dec`；脚本直接检查 MAT 内容，缺失或异常标签时报错。

测试命令：

```powershell
python -m unittest discover -s tests -p test_split_openbmi.py
```
