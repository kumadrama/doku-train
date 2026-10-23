# Spec 016：Finder 结构的公共 loss

状态：需求与设计已确认，实施及完整验证完成。

Source: 2026-10-23 用户要求公共 loss 放到 comm/nn_utils.py，训练与评估共用，不维护两套 logloss；实现方案按 Finder。

- 新增 recommend/train/comm/nn_utils.py，沿用 Finder 的 get_loss 公共入口及按任务 loss_type 分派结构。
- 神经网络训练、验证、测试与 logloss 指标使用同一底层 loss 实现，删除 task_metrics.py 的独立 NumPy logloss 公式。
- 保持 PyTorch CPU；不复制 Finder TensorFlow、私有依赖或源码。
- 保持当前标签、mask、权重、归约、不可计算指标和选模语义；不引入 Finder 未被本仓配置声明的功能。
- 树模型 loss 留在 rerank_tree/ 内独立实现；NN 不依赖树实现，后续可移除树目录。XGBoost 原生训练目标不变。
- 不启动云训练，不发布模型，不改已有实验配置。
