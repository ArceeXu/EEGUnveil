# OpenBMI 正常组主导的混合蒸馏训练

入口仍为 `src/offline/train_v65_distill.py`。输入是分组脚本输出目录中的 `report.json`，沿用其中固定的受试者组别，从 `source_train` 读取原始训练 NPZ。不读取最终 `test.npz`，不使用 CSP 分组阶段的四维特征。

```bash
python src/offline/train_v65_distill.py --groups-dir Data/OpenBMI/groups/csp70_motor20 --mix-lambda 0.75 --output-dir models/offline/v65_openbmi_lambda075
```

将 groups-dir 换成实际生成的分组目录。若需要新版 20 通道分组，请先运行新版分组脚本，不要把旧目录的名称当作已经更新的评分协议。

## 基线与消融命令

下面使用已经存在的 `csp70_20` 分组目录。它的分组报告应保持固定，所有实验使用同一输入和同一 seed。

```bash
# 自然组别比例 + 蒸馏 + 方差正则（输入混合策略基线）
python src/offline/train_v65_distill.py --groups-dir Data/OpenBMI/groups/csp70_20 --sampling natural --seed 42 --output-dir models/offline/natural_kd_seed42

# lambda 混合 + 无蒸馏 + 方差正则（去除蒸馏的消融）
python src/offline/train_v65_distill.py --groups-dir Data/OpenBMI/groups/csp70_20 --sampling mixed --mix-lambda 0.75 --no-distillation --seed 42 --output-dir models/offline/mixed075_no_kd_seed42

# lambda 混合 + 蒸馏 + 方差正则（完整方法）
python src/offline/train_v65_distill.py --groups-dir Data/OpenBMI/groups/csp70_20 --sampling mixed --mix-lambda 0.75 --seed 42 --output-dir models/offline/mixed075_kd_seed42

# 自然组别比例 + 无蒸馏 + 方差正则（补齐 2x2 对照）
python src/offline/train_v65_distill.py --groups-dir Data/OpenBMI/groups/csp70_20 --sampling natural --no-distillation --seed 42 --output-dir models/offline/natural_no_kd_seed42
```

natural 模式不使用 lambda：保留原有每折左右手类别均衡步骤，然后 shuffle=True，每轮每个保留的 trial 使用一次。不会按正常/BCI 盲组重新采样。自然组别比例指当前折类别均衡后的实际比例；报告会记录该数值。mixed 模式有放回、按固定批次组别比例采样，两种模式的采样重复次数不同，应在实验中说明。

--no-distillation 不创建教师，只保留学生的 CE + 方差正则，仍使用相同学习率、增强与早停。其硬标签损失按原实现直接使用，完整蒸馏模式则以 (1-alpha) 乘硬标签项，这是继承的损失定义，报告消融时应写清。--seed 同时设置数据划分、混合采样和训练随机种子，不重新计算已经固定的分组。

## λ 的定义

λ 是正常组的批次采样占比，正常组∶BCI 盲组 = λ∶(1−λ)。默认 0.75。要求 0.5 < λ < 1，不能只使用某一组。每个批次固定两组数量，并在每组内部均衡左右手；128 的批次在默认值下包含正常组 96、BCI 盲组 32 个 trial。按偶数取整后的实际比例会打印并保存，若取整导致某组为空或正常组不占多数则停止。

在每折原有类别均衡下采样后，训练加载器按组及左右手有放回采样，所以同一 trial 可以重复出现，一轮不保证遍历所有 trial。每轮批次数为 ceil(均衡后的训练 trial 数/批大小)。验证集合及顺序不随 λ 改变，不进行比例采样。λ 不乘到损失上，不替代原有 LAMBDA_VAR=1e-5 或 DISTILL_ALPHA=0.5。

CSP 和 StandardScaler 仍在每折类别均衡后的训练 trial 上拟合；λ 只作用于神经网络的批次输入，不给 CSP/标准化额外加权。

## 保留和适配

保留教师 ShallowConvNet、学生 EEGNetLight 时空双分支、6 维 CSP 空间特征、192 维拼接、教师独立 CE 更新与 detach、学生 CE+方差惩罚+温度 KL、AdamW、十折标签分层验证及原有软/硬/全局停止逻辑。

按论文文字修正原 v65 的三处不同参数：Transformer dropout 0.3→0.1，学生 weight_decay 1e-2→1e-4，增强噪声 sigma 0.025→0.005。最佳权重使用克隆快照，避免后续训练把“最佳状态”覆盖。

OpenBMI 左右手二分类要求教师和学生输出为 2 类；类别权重沿用原脚本的左右手项 [0.98,1.01]。通道数和时间长度由原始 NPZ 决定；现有输入为 62 通道、250 Hz、[0,4) 秒的 1000 点，不补造原论文的第 1001 点。分组用 20 通道和 1–3.5 秒，不等于训练也采用该输入。

离线预处理通过 NPZ 的 metadata_json.source_file 找到原始 MAT，只读取 EEG_MI_train；沿用原 GDF 加载器的 MNE FIR 0.5–100 Hz 和 50 Hz 陷波参数，先处理连续信号，再按 NPZ 的 onset_sample 提取 [0,4) 秒并抗混叠重采样至 250 Hz。原始 MAT 必须仍在记录的位置。没有改写原始文件或分组 NPZ。

KL 沿用仓库 F.kl_div(log P_student, P_teacher) 实现，即 KL(teacher || student)；论文文字的 KL 符号方向与代码不同，本次未翻转已有蒸馏逻辑。

## 输出及评估边界

输出为 best_model.pth、best_csp.pkl、best_scalers.pkl、model_config.json 和 validation_metrics.json。额外打印并保存最佳 epoch 的正常组/BCI 盲组验证准确率。教师参与训练，但按原逻辑只保存用于推理的学生，不额外导出教师 checkpoint。

保留论文的最高验证准确率跨折选优和 86.6% 全局停止；因此可能未跑完十折，保存的验证成绩不是独立测试成绩。固定组别来自整个原始训练阶段，包括内部验证 trial 的标签，故内部验证不应当作无偏的分组算法评估；最终结论应在完全隔离的官方测试阶段确认。试验不同 λ 时，不应依据最终测试集反复挑选 λ。

本次没有改动 `evaluate_v65_offline.py`（它仍为 BCICIV2a 四分类读取），不能直接用它评价新模型。没有修改在线控制或硬件相关文件。

批大小、轮数、加载进程数可通过 --batch-size、--epochs、--num-workers 指定，默认分别为 128、120、0。原始 Transformer 在 1000 点长序列上显存需求较大，必要时降低批大小；实际比例受批大小取整约束。
