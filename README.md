# EEGUnveil：面向BCI盲的脑控康复训练辅助系统

整合版本：三代算法独立入口、Xu原版GUI，以及保留的原在线模型和小车通信模块。当前代码与外部权重的就绪情况见下方说明。

| 部分 | 实现 |
|---|---|
| Gen1 | 正常组训练，无蒸馏 |
| Gen2 | 固定比例混合、时空融合、同步蒸馏 |
| Gen3 / SDAD-Net | 用户后期高低组DANN及配套实验 |
| GUI | 实时展示 |
| 硬件链路 | v50pro、TCP收发及串口小车控制 |

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

GUI直接运行 `src/online/main_app.py`，使用receiver、sender、模型加载和预处理。
## 文档

- [Xu GUI恢复说明与校验](docs/integration/gui_original.md)
- [整合说明](docs/integration/README.md)
- [双方协议差异](docs/integration/protocols.md)
- [源码来源](docs/integration/sources.json)
- [检查结果](docs/integration/validation.md)
- [SDAD逐文件用途](sdad_integration/docs/file_index.md)
- [原在线模型与硬件通信](hardware_legacy/README.md)
- [Xu原始首页说明](docs/integration/Xu_README_original.md)

## 运行边界

Gen1/Gen2和SDAD的通道、数据划分、分组与滤波不同，分别保留配置和结果，不直接互换权重或拼成受控比较。Gen2现有预设λ=0.5625。

GUI展示；真实小车链路在hardware_legacy中单独运行。
