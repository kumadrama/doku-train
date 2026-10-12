# Spec 005：行为标签训练数据接入

何时读取：接入 like/comment/share/follow/subscribe 的训练数据或任务；复现旧 v4 实验时无需读取。

Source: 2026-10-12 用户要求沿 immersive_click 相同链路，把五类行为接入当前训练数据流。

- 消费 doku-offline `fyp-rank-features/v5` 的不可变 Manifest 和五个 nullable UInt8 标签。
- 五个标签来自上游 behavior_flags 的成员投影，不在训练侧重算归因或查询原始日志。
- 已知正负例均参与任务；NULL 通过 `not is_null(label)` 排除，不填零。
- share 表示打开分享面板，subscribe 表示剧集追更；标签记录窗口内发生过正向行为。
- 保持配置驱动，三个模型共享既有数据读取链路；标签不进入模型输入 features。
- 不改变现有 v4 数据集、实验配置、权重或结果；新实验固定新 Manifest 的 URI、版本和摘要。
