# OpenBMI 左右手识别 GUI

本次界面使用已有离线模型，对原始 OpenBMI MAT 文件进行逐试次回放。
不需要重新训练，不需要旧版 TXT 时间线、GDF/XDF 文件或小车硬件。

## 在 Windows 本机启动

在 Git Bash 输入：

```bash
cd /d/TJFiles/EEGNet/Final-Version-BCI-Control-Car-1
D:/anaconda3/envs/bci-openbmi/python.exe src/online/main_app.py
```

使用已有 bci-openbmi 环境。界面依赖 Tkinter、PyTorch、NumPy、SciPy、
scikit-learn、MNE、joblib、Matplotlib；不再需要 pyxdf 或串口库。
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

通常直接用本地回放即可。需要演示网络通信时，另开终端先启动发送端，
将下方文件路径替换为实际 MAT 文件：

```bash
cd /d/TJFiles/EEGNet/Final-Version-BCI-Control-Car-1
D:/anaconda3/envs/bci-openbmi/python.exe src/online/sender.py --data "D:/你的数据目录/实际文件.mat" --phase test --host 127.0.0.1 --port 65432 --interval 1
```

随后在 GUI 选择“网络接收”，地址 127.0.0.1，端口 65432，再点击开始。
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

## 旧入口

receiver.py、sender.py、main_app.py 已采用 OpenBMI 回放链路。
evaluate_v50pro_stream.py 保留旧文件名，但已改为 OpenBMI 单记录命令行回放。
train_v50pro_online.py 保留旧文件名，转到现有 OpenBMI 留受试者训练入口；
GUI 不调用它，也不要求重新训练。

receive_control_utf8.py 是独立的旧四分类串口小车程序，未接入本 GUI，
本次没有改动它的车辆动作协议；不要用它运行 OpenBMI 模型。
本次没有修改离线训练代码、已有模型和实验图表。

本功能使用整段记录的离线滤波，属于已录制数据回放演示。
它不是实时 EEG 设备采集链路，也不会给小车发送指令。
代码本次仅作静态修改，按要求未启动 GUI、训练、测试或回放。

