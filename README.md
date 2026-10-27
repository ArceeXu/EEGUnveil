# EEGUnveil：面向BCI盲的脑控康复训练辅助系统

整合版本：三代算法独立入口、Xu原版GUI，以及保留的原在线模型和小车通信模块。当前代码与外部权重的就绪情况见下方说明。

| 部分 | 实现 |
|---|---|
| Gen1 | Xu正常组训练，无蒸馏 |
| Gen2 | Xu固定比例混合、时空融合、同步蒸馏 |
| Gen3 / SDAD-Net | 用户后期高低组DANN及配套实验 |
| GUI | 完整使用Xu原版，不添加模型适配 |
| 硬件链路 | 独立保留原v50pro、TCP收发及串口小车控制 |

## 使用入口

在项目根目录：

```bash
python project.py list
python project.py run gui
python project.py run gen1 -- --help
python project.py run gen2 -- --help
python project.py run sdad43 -- --help
python project.py run hardware -- check
```

GUI仍直接运行Xu的 `src/online/main_app.py`，使用其原始receiver、sender、模型加载和预处理。不自动识别或加载SDAD模型。SDAD保持独立命令行训练与测试；权重后续补齐。

## 文档

- [Xu GUI恢复说明与校验](docs/integration/gui_original.md)
- [整合说明](docs/integration/README.md)
- [双方协议差异](docs/integration/protocols.md)
- [源码来源](docs/integration/sources.json)
- [检查结果](docs/integration/validation.md)
- [SDAD逐文件用途](sdad_integration/docs/file_index.md)
- [原在线模型与硬件通信](hardware_legacy/README.md)
- [Xu原始首页说明](docs/integration/Xu_README_original.md)（相对路径按项目根目录解释）

## 运行边界

Gen1/Gen2和SDAD的通道、数据划分、分组与滤波不同，分别保留配置和结果，不直接互换权重或拼成受控比较。Gen2现有预设λ=0.5625，来源于Xu已保存验证选择。

Xu原版GUI用于OpenBMI记录回放；真实小车链路在hardware_legacy中单独运行。v50pro为25通道四分类，不能代替SDAD左右手二分类成果。原EEG数据和真实SDAD权重需外部提供。

已有GPU环境先确认PyTorch，再按需安装依赖；历史复现应匹配原库版本，不直接覆盖环境。保留原LICENSE、双方来源和历史记录。
