> 最新状态：用户要求保留原在线模型和小车通信。相关13个文件已恢复；团队入口见仓库根目录 `hardware_legacy/README.md`。下文删除描述属于首次清理历史，以manifest的最新cleanup_action为准。

# 数据、权重与结果资源

本页回答：哪些资源在包内，哪些需要外部提供。

| 资源 | 当前状态 |
|---|---|
| SDAD训练、测试、诊断代码 | 位于 `../code/` |
| SDAD初始化及最终学生权重 | 外部提供，本包不含 |
| 原始OpenBMI EEG、缓存 | 外部提供，本包不含 |
| SDAD逐人结果与服务器汇总 | 尚未从用户其他下载目录收录，不代表已丢失 |
| 旧小车v42、v65、v50pro模型 | 经用户要求已从本地整理副本删除 |

原服务器资源根目录为 `/mne_data/mne_data/`，实验位于其 `bci_runs/` 下。固定分组CSV、CSP、标准化器及准备数据与检查点配套，不可跨版本混用。

继续训练应提供学生和教师状态、配置与预处理资源；仅有best.pt不一定包含教师。推理需要学生权重、结构配置、CSP及标准化器。

本地曾使用的下载目录为 `D:\Final-Version-BCI-Control-Car\outputs\seed42_diagnostics\`，本次没有操作此目录。

去向与删除状态见 `../reports/repository_inventory.csv`；详情见 `../reports/cleanup_log.md`。清单保留历史SHA256，不代表所有原文件仍存在。当前工作树删除没有清除GitHub历史。
