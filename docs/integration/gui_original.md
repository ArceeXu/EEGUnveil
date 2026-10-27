# Xu原版GUI恢复记录

日期：2026-10-27。按用户要求使用Xu原版GUI，不增加SDAD模型适配。

## 已恢复

以Xu `dev_arcee` 的来源提交 `ff684d83c3f3cd341d2e0669c8ee30ef96e2b79e` 为准，`src/online/`下8个原始文件全部恢复并逐文件验证Git原始内容SHA256。GUI、receiver、sender及原在线入口均无差异。

恢复记录与校验值：`gui_restore.json`。

## 已移除的自定义内容

- `src/online/model_registry.py`：自动模型识别与分流。
- `integration/sdad_backend.py`：额外SDAD GUI适配器。
- `integration/__init__.py`：该适配包标记。
- `tests/test_integration_adapter.py`：上述自定义适配器测试。

新增 `tests/test_gui_original.py`，防止原版GUI被意外改动，并检查GUI直接使用原receiver。

## 保留的内容

- Xu所有模型、实验、训练主体及5个sh启动脚本。
- 用户SDAD训练、测试、历史探索、诊断与原推理代码。
- hardware_legacy的原在线模型、TCP收发与串口小车控制。
- project.py仅作为命令调用入口，不替换Xu的模型加载器。

现在运行 `python project.py run gui` 即为Xu原版界面。没有额外再建一套GUI副本，因为当前主GUI本身已经恢复为原版。真实SDAD权重后续补齐，用独立入口加载，不自动接入该GUI。

本次仅撤销GUI适配，没有撤销其他已记录的边界检查和旧四分类辅助函数清理。未修改远程仓库。
