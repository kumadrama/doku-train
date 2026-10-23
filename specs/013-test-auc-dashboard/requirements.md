# Spec 013：每日 test AUC 看板

Source: 2026-10-17 用户确认：仅 test AUC、四个 task 趋势图、按 model_name 分组；先补 Spec 再实施 train/monitoring，本次不部署。

1. 每个模型一组四图，任务与指标名来自显式训练配置；多个模型互不混合。
2. 横轴为完成 test 评估的自然日（Asia/Shanghai），纵轴为 AUC。只展示 task 与 AUC，不展示样本数、run/version。
3. 同模型、任务、指标、同日取最后一次完成评估的结果，不平均；最新结果为 null 时保留缺失，不回退旧值，不填 0。
4. 发布门禁拒绝、导出/发布失败不能抹掉已完成 test 的指标。未训练、test 失败不生成指标。
5. 不新增告警，不改变训练、选模、发布门禁，不触发云任务或部署。

当前状态：test-metrics.json 已存在；训练任务有 awslogs，尚无 AUC 看板出口。
目标状态：结构化日志 → 既有 CloudWatch Logs → monitoring 生成的 Grafana 看板 artifact。
