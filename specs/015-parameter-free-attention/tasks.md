# Spec 015：实施任务

Source: 2026-10-21 用户确认真正拆分字段组合与历史聚合，取代仅对齐启用字段的先前实现。

- [x] 核对 Finder embedded_features/attention_layer 完整调用链及 avg_pool 轴。
- [x] 将 combiner 限定为 concat/sum/avg，新增独立 history_pooling 和互斥条件；迁移 fixture。
- [x] 实现先组合再聚合、多字段 query/history、mask/维度校验与 none 的显式保留语义。
- [x] 同步多字段历史长度验证和 tail 截断；拒绝多字段 top_weight。
- [x] 数值、梯度、schema、字段对齐与编码专项回归。
- [x] 同步 README、architecture、Spec004/015 的有效配置语义。
- [x] 多字段 DNN/DeepFM train/test、恢复和完整 make check 验证。
