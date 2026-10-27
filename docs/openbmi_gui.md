# OpenBMI 左右手识别 GUI

## 启动GUI

打开终端在根目录输入

```bash
python src/online/main_app.py
```

界面依赖 Tkinter、PyTorch、NumPy、SciPy、
scikit-learn、MNE、joblib、Matplotlib。
模型 pickle 文件应来自自己的训练，依赖版本宜与训练环境一致。

## 本地回放步骤

1. 模型文件夹默认为混合训练＋蒸馏、λ=0.5625、seed42。
   可以选择其他已训练的 OpenBMI 模型文件夹。
2. 选择原始 OpenBMI `.mat` 文件，而非 groups 下的 npz。
3. 运行方式选择“本地文件回放”，阶段默认 test，点击“开始回放”。
4. 首次读取和滤波可能需要时间；界面状态栏会显示正在预处理。
5. 字体＋／－调整字号；窗口可拖动放大；F11 全屏、Esc 退出全屏。
6. 点击“停止回放”停止；整段滤波期间会在预处理完成后响应停止。

模型目录需包含：
- model_config.json
- best_model.pth
- best_csp.pkl
- best_scalers.pkl

图中 C3、Cz、C4 为当前试次预处理后的波形，单位 µV。
显示的推理耗时包括通道对齐、CSP、标准化和模型计算，不含 MAT 加载与滤波。
置信度是模型输出概率，不是识别正确的保证。
回放倍速控制试次间等待；推理和绘图也需要时间，不代表严格实时速度。

## 可选：网络回放

另外再开启一个终端，输入

```bash
python src/online/sender.py \
  --data "你的数据目录/实际文件.mat" \
  --phase test \
  --host 127.0.0.1 \
  --port 65432 \
  --interval 1
```

将文件路径替换为实际 MAT 文件。随后在 GUI 选择“网络接收”，地址 127.0.0.1，端口 65432，再点击开始。
此模式由发送端选择 MAT 文件和 train/test 阶段；GUI 的文件、阶段和倍速选项不参与网络回放。
发送端处理完一个记录后退出；下一次连接需重新启动发送端。
跨机器时发送端绑定自己的可访问地址，GUI 填发送端地址，并放通相应端口。
新协议传送完整预处理试次，不能与旧版 GDF 发送端混用。

## 数据与模型对应方式

- 读取 EEG_MI_train 或 EEG_MI_test，只加载选定阶段的一份记录。
- 原始数据由 µV 转为 V。
- 连续记录 FIR 0.5–100 Hz 滤波及 50 Hz 陷波。
- 按原有离线代码的 onset 索引截取 4 秒并重采样到 250 Hz。
- 按 model_config.json 通道名重排，调用保存的 CSP 和各通道标准化器。
- 复用离线 EEGNetLight，输出二分类；原始标签 1 映射右手，另一类映射左手，
  与已有 OpenBMI 数据划分脚本一致。
- 不一次性加载全体受试者，也不在 GUI 中重新拟合 CSP、标准化器或模型。

MAT 中的 test 是数据文件内部阶段，不能自动保证受试者未参与训练。
要展示未见受试者，请选择 subject_split.json 中 test 名单对应的文件。
当前回放符合率仅描述所选记录；正式论文结果仍使用之前的留出评估输出。