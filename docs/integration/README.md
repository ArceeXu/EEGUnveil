# 团队整合说明

| 模块 | 当前处理 |
|---|---|
| Xu训练、划分、比例搜索和权重 | 保留原实现与资源，其他小范围变更见历史差异 |
| Xu GUI和在线目录 | 已全部恢复原提交，无模型适配 |
| 用户SDAD | 独立放入sdad_integration，不覆盖Xu src |
| 小车通信 | hardware_legacy保留v50、TCP和串口程序 |
| 统一入口 | project.py以各自工作目录调用原脚本 |

不共享GUI模型加载器：Xu GUI保留其62通道OpenBMI路径；SDAD独立训练、测试和推理；旧四分类小车链路单独使用。三者协议不混用。

原有数据边界检查已补充，旧分组测试已更新至process_subject接口；这些不是GUI适配，不在本次撤销范围内。旧四分类加载辅助函数的历史清理也不属于GUI代码。

来源见sources.json，GUI恢复见gui_original.md，硬件恢复见online_restore.md。提交前先补齐SDAD权重和配套预处理，分别验证真实数据；本次不启动真实训练或硬件。

用户已明确选择直接更新main。发布基于main现有历史，不重写Xu分支；具体发布范围见publication.md。先前检查中的“未推送”是对应检查时的历史状态。
