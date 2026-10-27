# 原在线模型与小车通信链路

本模块按用户要求恢复原v50pro和通信程序，用于填补硬件演示部分。Xu的OpenBMI GUI、MAT网络回放和三代算法代码保持独立。

| 文件 | 用途 |
|---|---|
| `models/online/v50pro/` | 原25通道、1001点、四分类模型与CSP／标准化器 |
| `src/online/sender.py` | 原GDF数据分块发送端；是模拟数据发流，不是采集设备驱动 |
| `src/online/receiver.py` | 原TCP接收、在线滤波、滑窗推理与波形显示 |
| `src/online/receive_control_utf8.py` | 原接收推理＋串口小车控制，含转向、前进、刹车逻辑 |
| `src/online/main_app.py` | 原旧界面源文件参考；项目主GUI仍为Xu版本 |
| `src/online/torch_bootstrap.py`、`debug_dll_load.py` | 原Windows DLL支持／排查工具 |
| `src/tools/generate_cue_timeline.py` | 从原GDF生成提示时间线 |
| `outputs/timelines/` | 两份原提示时间线示例，不默认匹配当前数据 |
| `launch.py` | 新增参数化入口，避免使用原文件写死的设备地址 |
| `manifest.json` | 恢复来源及逐文件SHA256 |

## 先查看，不连接设备

从项目根目录运行：

```bash
python project.py run hardware -- check
python project.py run hardware -- --help
```

`check`只检查文件与配置，不连接网络、不打开串口。旧模型和原通信源文件保留原始字节；改地址等通过launch参数覆盖。

## 使用链路

1. 准备原模型对应的GDF或兼容采集流，并生成匹配的提示时间线。
2. 发流示例：`python project.py run hardware -- sender --data-dir /absolute/path/to/GDF --subject 3 --host 0.0.0.0 --port 65432`。
3. 仅接收推理：`python project.py run hardware -- receiver --timeline /absolute/path/to/timeline.txt --host 127.0.0.1`。
4. 控制小车：将receiver改为car，并显式提供实际 `--serial-port`；波特率默认115200，可通过 `--baud` 指定。

用户需要核对TCP地址、端口、实际串口、车体协议及转向参数，提示时间线必须与任务一致。现有测试只用合成信号和内存串口，不会启动真实小车。

## 协议边界

- 原链路使用通道数头部＋D/M分块消息；Xu发送端使用带JSON头的试次消息，两者不能交叉连接。
- 原v50模型为Left/Right/Foot/Tongue四分类；OpenBMI/SDAD为左右手二分类。不能把v50演示结果当作SDAD准确率或补出前进／停止类别。
- 原串口指令与控制逻辑已保留；将SDAD二分类接入同一车体是后续单独的控制映射工作，本次没有自动接通。
- 原模型文件不包含完整训练期通道名称。接真实采集设备前必须确认25通道顺序、单位与采样率。
- 原旧模型可能包含原数据集EOG通道，保留工程演示不代表其符合当前EEG算法比赛的数据协议。

## 可选依赖

除项目核心依赖外，串口模块需要`pyserial`；原旧界面需要`pyxdf`。需要哪部分再安装，不覆盖GPU版PyTorch。
