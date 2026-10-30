# 设计

Source: requirements.md；Spec 023 调度语义；Spec 024 TF CPU 图运行时。

当前状态：TF 新增 comm/learning_rate.py，不依赖 Torch。model.learning_rate_schedule 和可选 model.sparse_learning_rate_schedule 包含 type（piecewise_constant / piecewise_linear）、points（epoch、learning_rate）。独立 sparse 调度要求显式 sparse_optimizer；未声明的独立 sparse LR 保持原值。共用优化器时 dense LR 同时作用于稠密和稀疏梯度。

首节点必须是 epoch 1，LR 等于初始值；节点严格递增且不超过 max_epochs。LR 必须有限且正；拒绝未知类型、额外字段和空节点。用例在准备数据前校验，直接 runtime 入口同样校验。模型参数解析校验结构。

piecewise_constant 在节点指定轮开始切换；piecewise_linear 在相邻节点间按 epoch 插值。最后节点之后保持不变。每轮建图前设置 params 中实际 LR；原 Saver 恢复全部权重和优化器 slots，LR 为建图常量，不由 checkpoint 覆盖。原始配置不变，调度进入既有 identity，修改调度的恢复请求被拒绝。

每轮指标写 learning_rate，独立 sparse_optimizer 时同时写 sparse_learning_rate；日志和 epoch-metrics/metrics.json 同步包含它们。现有本地 TensorBoard 转换器将数值字段映射为 train/learning_rate 和 train/sparse_learning_rate。WD 语义不变，不自动随 LR 缩放。

为保持对照，A/B 使用同一镜像、冻结数据、随机种子、线程与机器规格；对旧 LN 配置仅增加 model.learning_rate_schedule。镜像包含当前 main 的已验证性能优化和本变更，必须记录 source/lock/config/image 摘要，不把镜像变更冒充配置差异。
