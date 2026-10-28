# Epoch 学习率调度
Source: 2026-10-28 用户确认配置驱动方案并授权实现，首期 PyTorch。
- 可选 schedule；缺省固定学习率，既有训练算法不变。
- 支持 piecewise_linear / piecewise_constant；epoch 从 1 开始，每轮开始设置。
- 恢复训练恢复 optimizer/RNG，并按目标 epoch 重算学习率。
- history 和 TensorBoard 记录实际学习率；不改变 TensorFlow，不提交云任务。
