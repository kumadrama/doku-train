# Spec 026：TF 可选加权指标早停

Source: 2026-10-30 用户要求按 early_stop.patient、metric、weight 配置启用早停，未声明则跑满规定 epoch，并授权实施。

- 仅显式 early_stop 对象启用；缺省维持 max_epochs 和既有 selection 行为。
- 使用验证集的 auc / logloss，按配置任务权重聚合；不以训练或测试指标决定早停。
- 连续 patient 次有效评估未充分改善时结束训练；保留真实最佳 checkpoint，不将停止轮误作最佳轮。
- 早停同时定义选模标准，启用后取代原 selection.task_losses；指标不可计算记录 null/原因并继续训练，不静默改权重或降级到其他指标。
- checkpoint 保存计数与最佳状态，恢复后语义与连续训练一致；仍保留已完成训练产物。
- 本次仅实现并验证 TF CPU 路径，不启动新云训练、不变更运行中的实验。
