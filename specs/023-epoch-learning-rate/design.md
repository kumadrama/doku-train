# 设计（已确认）
Source: 2026-10-28 会话方案。
optimizer.learning_rate_schedule 包含 type 和 points；每个节点为 epoch、learning_rate。
首节点必须为 epoch=1 且与 optimizer.learning_rate 相同；节点严格递增、不超过 max_epochs，
学习率必须为有限正数。节点之间线性插值或保持左节点常数；最后节点后保持不变。
示例：points=[{epoch: 1, learning_rate: 0.004}, {epoch: 10, learning_rate: 0.004},
{epoch: 30, learning_rate: 0.001}]；第11轮0.00385，第20轮0.0025，第30轮及以后0.001。
公共 comm/learning_rate.py 为无状态纯函数；模型 runner 每轮开始统一更新参数组 lr，
不修改各组 weight_decay。恢复后的 epoch 是唯一调度进度来源，无独立 scheduler 状态。
既有 resume identity 校验继续有效。每轮 history 增加 learning_rate。
metric_events 发出 train learning_rate 事件，保留真实 global step；SageMaker EventSink
将该事件写为 train/learning_rate，TensorBoard 横轴使用 epoch；其他指标保留原 step。
仅在已完成 epoch 的 history 保留记录；中途失败的学习率事件可能已发出。

## YAML 示例

```yaml
optimizer:
  name: adam
  learning_rate: 0.004
  weight_decay: 0.0001
  learning_rate_schedule:
    type: piecewise_linear
    points:
      - epoch: 1
        learning_rate: 0.004
      - epoch: 10
        learning_rate: 0.004
      - epoch: 30
        learning_rate: 0.001
```

阶梯调度将 type 改为 piecewise_constant，节点在指定 epoch 开始生效。
未配置 learning_rate_schedule 的旧 YAML 无需修改。未知字段和类型 fail closed。
