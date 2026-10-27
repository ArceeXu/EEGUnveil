> 最新GUI状态：已撤销自定义模型适配，src/online恢复Xu原提交。GUI使用Xu原模型路径；SDAD和硬件链路独立运行。此前有关适配器的描述属于历史记录，见gui_original.md。

> 最新恢复状态：原v50pro权重已恢复，TCP收发与串口小车控制已保留到 `hardware_legacy/`。此前有关旧权重删除或禁止使用的描述是上一阶段记录；其只是不适用于OpenBMI选择器，硬件专用入口可使用。

# 协议与接口对照

以下依据实际源码和已保存model_config读取，不依据方法名称推断。

| 项目 | Xu Gen1 / Gen2 | 现有SDAD Gen3 |
|---|---|---|
| 任务 | OpenBMI左右手二分类 | 同任务 |
| 网络输入 | 62通道、1000点 | 11通道、1000点 |
| 划分 | 固定32训练／11验证／11测试受试者 | 54人session1训练／验证，session2测试 |
| 分组 | session1/train、session2/train分别CV后取均值，20运动区通道评分 | session1训练数据分组，固定25名低表现者 |
| 连续预处理 | FIR 0.5–100Hz、50Hz陷波、重采样 | 因果IIR 0.5–100Hz、50Hz陷波、降采样 |
| 输入单位 | MAT数值乘1e-6 | 保持原MAT数值单位 |
| 标准化 | 保存的逐通道scaler | 保存的mean/scale与CSP scaler |
| 权重 | best_model.pth＋model_config.json | best.pt＋config.json |
| 预处理资源 | best_csp.pkl＋best_scalers.pkl | preprocessing.joblib |
| 当前评估对象 | 未参与训练的留出受试者 | 已见受试者的跨会话表现 |

注意：Xu的“train-only”指每个session内的EEG_MI_train，不等于“只使用session1”。这与用户SDAD分组规则不同，不默认共享25人名单。

## 可统一与不可直接统一

可统一：项目导航、方法命名、GUI、左右手输出格式、结果文件管理。

暂不直接统一：权重、通道、滤波器、标准化器、受试者划分、分组名单和结果平均。若要做严格的三代性能对比，需要另行固定相同协议并重新评估；本次整合不启动该实验。

## 资源就绪情况

- Gen2：已有λ0.5625的42、2024、3407模型目录；CPU加载42模型成功。
- Gen1：仓库有实现与结果记录，未找到独立正常组模型权重目录；需要外部提供。
- SDAD：代码齐全，真实best.pt及配套preprocessing.joblib未加入；需补齐。
- 数据：原始EEG不在克隆中。

已额外检查本地 `D:\Final-Version-BCI-Control-Car\outputs\seed42_diagnostics`：找到旧方案B `shared_baseline_b_001` 的best.pt、last.pt及预处理资源，但未找到正式后期SDAD三组实验的best.pt。方案B权重不能代替SDAD模型。

只加载本人或合作者可信的torch/joblib资源。跨scikit-learn版本加载虽可能成功，不构成数值复现保证。
