# 设计与使用

Source: requirements.md；Spec 024 图运行时；Spec 025 epoch LR 调度。需求及配置方向已由当前用户请求授权。

## 当前实现

`early_stop` 是与 `model`、`max_epochs` 同级的可选字段。没有该字段时不实例化早停控制器，训练跑满 max_epochs，既有 selection 不变。显式 null、空对象或拼错字段被拒绝。

```yaml
max_epochs: 40
early_stop:
  patient: 8
  metric: auc
  weight:
    immersive_click: 0.5
    fast_swipe: 0.25
    completion: 0.25
  # min_delta: 0.0001  # 可选，默认 0
```

metric 也可以是 logloss；auc 自动最大化、logloss 自动最小化。字段名保留用户约定的 patient（不是 patience）。weight 必须引用 model.tasks 中真实名称；不引入 click/complete 到业务任务的隐式映射。当前实验的真实名是 immersive_click/completion。

patient 必须为正整数；weight 非空、权重有限非负且和有限为正，运行时归一化为加权平均。零权重项不参与计算；正权重任务必须声明同名同 func 的指标。min_delta 有限非负。全部结构在准备输入前校验，直接 prepared runtime 入口也校验。

## 选模与停止

指标来自 validation_metrics。每轮严格优于历史原始最佳分数就更新 selection.json，分数相等保留更早 checkpoint。另设 reference_score 判断改善量是否严格大于 min_delta，满足则更新 reference_score 并把 wait 清零，否则 wait 加一；wait 达到 patient 时在该轮全部保存后退出。不按相邻两轮比较。

例如第 10 轮是最佳，其后无改善，patient=8，则完成第 18 轮后停止并选第 10 轮。启用 early_stop 后其标准取代原 selection.task_losses，避免早停目标和实际测试 checkpoint 不一致；未启用则原选模路径不变。

任何正权重指标缺失、null 或非有限时，综合 score=null，记录每项原因，保留 wait/reference_score/best 原值继续训练；此时不移除该任务后重新归一化。若整次都不可评估，跑满 max_epochs，selection 明确 unavailable，experiment 不运行 test。数据/执行异常仍然失败。

## 持久化、恢复与 LR 调度

checkpoint.json 保存 early_stop 状态（best、reference_score、wait、stopped）；原配置参与既有 identity，因此修改 patient/metric/weight/min_delta 后恢复会被拒绝。恢复已停止 checkpoint 不再训练；恢复文件保留原最佳 checkpoint 引用与完整权重、优化器状态。

每轮 epoch JSON 记录综合分数、不可用原因、wait、最佳轮及是否停止；TensorBoard 写 validation/early_stop_score 和 train/early_stop_wait。最终 early-stop.json 与 experiment-status.json 记录实际完成轮数、是否触发早停、最终最佳轮和配置。允许最大轮数之前正常完成训练，不伪装成失败，也不伪造跑满轮数。

早停与 LR 调度独立：不隐式延长 patient，不自动等待未来 LR 节点。部署衰减实验前应显式评估二者时序。本次未加入 min_epochs、LR 后重置计数等额外策略。
