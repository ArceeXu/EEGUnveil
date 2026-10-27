> 最新状态：用户要求保留原在线模型和小车通信。相关13个文件已恢复；团队入口见仓库根目录 `hardware_legacy/README.md`。下文删除描述属于首次清理历史，以manifest的最新cleanup_action为准。

# 原小车专用文件删除记录

日期：2026-10-27。范围：本地 `sdad-cleanup` 副本及其整合包。尚未提交或推送。

## 删除范围

- 原v42、v65、v50pro模型及配套资源。
- 原小车专用离线／在线训练、GUI、通信和打包配置。
- 原提示时序、运行日志、旧时序备份及演示PPT。
- 整合包内的旧工程参考副本 `legacy_reference/`。

## 保留范围

- `src/plan_a`、`src/plan_b`、后期DANN、消融、诊断和候选队列。
- `src/tools/compare_plan_ab.py` 以及OpenBMI分组分析脚本。
- 用户的SDAD文档与整合说明。
- `.git`历史：删除只影响当前工作树，没有重写提交历史。

根README改为SDAD项目说明，根requirements改为引用OpenBMI核心依赖。原始EEG、服务器DANN权重和本地其他目录未被操作。

## 删除的原仓库文件（73个）

- `archive/cue_timeline_backup/cue_timeline0tt.txt`
- `archive/cue_timeline_backup/cue_timeline1.txt`
- `archive/cue_timeline_backup/cue_timeline2.txt`
- `archive/cue_timeline_backup/cue_timeline3.txt`
- `archive/cue_timeline_backup/cue_timeline3train.txt`
- `archive/cue_timeline_backup/cue_timeline4.txt`
- `archive/cue_timeline_backup/cue_timeline5.txt`
- `archive/cue_timeline_backup/cue_timeline6.txt`
- `archive/cue_timeline_backup/cue_timeline7.txt`
- `archive/cue_timeline_backup/cue_timeline8.txt`
- `archive/cue_timeline_backup/cue_timeline9.txt`
- `archive/cue_timeline_backup/txt notebook.py`
- `assets/pyinstaller/main_app.spec`
- `assets/slides/acc_data.pptx`
- `models/offline/v42/best_csp.pkl`
- `models/offline/v42/best_model.pth`
- `models/offline/v42/best_scalers.pkl`
- `models/offline/v42/fold_1_curves.png`
- `models/offline/v42/meta.json`
- `models/offline/v42/training_metrics.json`
- `models/offline/v65/best_csp.pkl`
- `models/offline/v65/best_model.pth`
- `models/offline/v65/best_scalers.pkl`
- `models/offline/v65/model_config.json`
- `models/online/v50pro/best_csp.pkl`
- `models/online/v50pro/best_model.pth`
- `models/online/v50pro/best_scalers.pkl`
- `models/online/v50pro/model_config.json`
- `outputs/logs/online_accuracy/Ee1.txt`
- `outputs/logs/online_accuracy/Ee2.txt`
- `outputs/logs/online_accuracy/Ee3.txt`
- `outputs/logs/online_accuracy/Ee4.txt`
- `outputs/logs/online_accuracy/Ee5.txt`
- `outputs/logs/online_accuracy/Ee6.txt`
- `outputs/logs/online_accuracy/Ee7.txt`
- `outputs/logs/online_accuracy/Ee8.txt`
- `outputs/logs/online_accuracy/Ee9.txt`
- `outputs/logs/online_accuracy/e1.txt`
- `outputs/logs/online_accuracy/e2.txt`
- `outputs/logs/online_accuracy/e3.txt`
- `outputs/logs/online_accuracy/e4.txt`
- `outputs/logs/online_accuracy/e5.txt`
- `outputs/logs/online_accuracy/e6.txt`
- `outputs/logs/online_accuracy/e7.txt`
- `outputs/logs/online_accuracy/e8.txt`
- `outputs/logs/online_accuracy/e9.txt`
- `outputs/logs/runtime/onlinev50prodata.txt`
- `outputs/logs/runtime/sequence.txt`
- `outputs/logs/runtime/v65data.txt`
- `outputs/timelines/cue_timeline0tt.txt`
- `outputs/timelines/cue_timeline1.txt`
- `outputs/timelines/cue_timeline2.txt`
- `outputs/timelines/cue_timeline3.txt`
- `outputs/timelines/cue_timeline3train.txt`
- `outputs/timelines/cue_timeline4.txt`
- `outputs/timelines/cue_timeline5.txt`
- `outputs/timelines/cue_timeline6.txt`
- `outputs/timelines/cue_timeline7.txt`
- `outputs/timelines/cue_timeline8.txt`
- `outputs/timelines/cue_timeline9.txt`
- `src/offline/evaluate_v65_offline.py`
- `src/offline/train_v42.py`
- `src/offline/train_v65_distill.py`
- `src/online/debug_dll_load.py`
- `src/online/evaluate_v50pro_stream.py`
- `src/online/main_app.py`
- `src/online/receive_control_utf8.py`
- `src/online/receiver.py`
- `src/online/sender.py`
- `src/online/torch_bootstrap.py`
- `src/online/train_v50pro_online.py`
- `src/tools/compare_logs.py`
- `src/tools/generate_cue_timeline.py`
