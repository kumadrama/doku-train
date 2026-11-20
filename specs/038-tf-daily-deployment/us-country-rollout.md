# FYP 美国/未知国家模型发布

Source: 2026-11-19 用户要求将已完成的美国/空国家实验上线，加入 FYP 任务组并启动日更；此前确认 watch_seconds 使用美国子集 MSE，其余二分类任务使用美国子集 AUC。

## 发布需求与设计

- 复用现有发布流程，ECS 任务族为 `doku-prod-tf-rerank-daily`。
- 模型名为 `tf_rerank_ln_lr002_3feat_us_or_unknown`，独立 Endpoint 为 `doku-prod-fyp-tf-rerank-ln-lr002-3feat-us-or-unknown`。
- 导出 `completion`、`fast_swipe`、`immersive_click`、`watch_seconds`，单位沿用 base。
- 复用 base 加权验证 loss 最小值选模。首次发布消费成功密封训练产物所选 epoch 3；日更每次重新选模。
- 配方保留 30 天训练、1 天验证、30 epoch、batch 2048 和学习率 0.002。
- item_filter 为 `(video_finish_seen > 0) and ((ctx_ip_country == 'US') or (ctx_ip_country == ''))`。默认值先填充，旧日期缺列及 NULL 按显式配置填为空字符串。
- 六个二分类任务保留原指标并增加 `auc_us`；watch_seconds 保留原指标并增加 `mse_us`，子集条件均为 `ctx_ip_country == 'US'`。
- 上游成功及既有恢复调度触发该模型独立 ready adapter / ECS revision，复用任务族失败报警与 FYP 指标归组；模型 Endpoint 告警独立创建。
- 本次采用既有独立实验模型 Endpoint 发布路径。线上 FYP/Discovery 的业务模型别名发布属于 Spec 040，不给该新模型绑定共享业务 profiles。
- IaC 沿用 FYP 模型资源登记；配置与制品使用不可变路径，原始实验输出保持成功密封。

## 验证

当前状态：首次发布已完成，Endpoint InService；日更及恢复规则已启用。

训练已完成 30 轮；6 个输入过滤用例和美国子集指标用例通过，37 项训练相关测试通过。首次产物选模 epoch 3。两组 TensorBoard 曲线 30 点完整，35 份本地已取回元数据/指标与成功密封摘要一致。

2026-11-19 生产验收：

- Terraform 完整计划为 14 新建、2 策略更新、0 删除；分首次发布权限与日更触发两个阶段 apply，均成功。随后 `terraform plan -detailed-exitcode` 返回 0，配置与实际状态一致。
- 任务定义 `doku-prod-tf-rerank-daily:20`；首次发布 ECS task `d1228c21eb6e4ee9829afe8bd527f760` 的 pipeline 退出码 0，发布日志记录 `deployed`。
- Endpoint `InService`，当前模型版本 `7e6623392c8b63a07b770743f80a684bf45cdbde5ff38d9f78dedb6ffe20dcf6`，发布状态与实际 EndpointConfig 一致。成功密封训练产物的 `selection.json` 确认为 epoch 3。
- `verify_live.py` 调用部署协议的 `verify_probe`，US、空国家及 CA 三条合成输入均返回四个任务的有限数值，行数及输出任务合同通过。
- ready Lambda 代码 SHA256 与打包制品一致；模型 worker rule、既有 upstream success 和 recovery 规则均 ENABLED，目标指向独立 ready Lambda 和任务定义 revision 20。
- 训练配方的 30/1 天窗口由 ready release 中的不可变配置解析；日更指标归属 `doku-prod-tf-rerank-daily`，保留六个 `auc_us` 和时长 `mse_us`。

- 首轮日更已通过新模型 EventBridge rule 启动，cutoff `2026-11-18`，训练日期 `2026-10-19` 至 `2026-11-17`，验证日期 `2026-11-18`。SageMaker job `doku-daily-9472383918b6e0f358a26101eae7e103752a78be` 实测为 `InProgress / Training`，其 `DOKU_MONITORING_CONTEXT` 确认模型名和 FYP 任务族一致。此次验收仅确认启动，不声明这轮训练已经完成。

实际生产回执及执行脚本保留在本地 `runs/fyp-us-release-20261119`，包括 `live-verification.json`、`release-state.json`、`applied-outputs.json` 和 `post-apply-plan.log`；不提交权重、实验配置或运行数据。

## 2026-11-20 指标上报修复

Source: 用户确认看板使用普通 `auc` / `mse`，授权修复日更配置并补报已完成训练。

新增 `auc_us` / `mse_us` 后，每个导出任务有两个同类型候选指标；未配置 `monitoring.metric_mapping` 导致 `ambiguous display metric: completion`。11/18 日更训练和发布成功，但状态记录 `monitoring: failed`，密封产物未包含标准监控事件，目录仅有发布记录。

本次在该模型独立 binding 中显式声明 completion、fast_swipe、immersive_click → `auc`，watch_seconds → `mse`。训练配方、选模、美国子集指标、训练及推理镜像不变。复用既有上报协议，从成功密封的 epoch 4 指标补报；不修改原始成功产物、不重训。ready 制品仅更新 release binding 引用，其余 ZIP 条目保持一致。

验证目标：真实产物摘要校验、显式映射事件生成、日更 revision/ready binding 一致、补报 ACK、目录任务和 CloudWatch AUC 查询、Grafana 模型查询接线。生产验收完成：

- `pytest recommend_tf/train/tests/test_monitoring_reports.py -q`：7 passed；`git diff --check` 通过。
- 新 worker 为 `doku-prod-tf-rerank-daily:21`，ready ZIP digest 为 `e9b05356dc38047245c9a6c1e8508c746bcc5b1171ef991c6082d1b32d5a192d`；已核验 EventBridge 目标、实际 task binding SHA256 和 Lambda CodeSha256 一致。
- 已保存 Terraform 计划仅 2 项原地更新；apply 结果 0 added / 2 changed / 0 destroyed，后续 `plan -detailed-exitcode` 返回 0，无遗留变更。本次独立 stack 的输出资源身份未改变。
- 补报事件 `c3477c5f9d35601099749d3bfbf377c0c2b15621c1c4e16f7b442983fb87896c` 已取得 delivered ACK；目录包含四项任务，归属 FYP 任务族。六份指标源元数据均与原 COMMITTED 摘要一致，原密封产物未修改。
- 既有监控 Lambda 同步 Grafana 成功，dashboard version 63。通过 Grafana API 读取实际四个 panel target 后执行其 CloudWatch 查询，均 Complete 且各返回 1 点（数据日 11/18）：completion 0.7933、fast_swipe 0.7318、immersive_click 0.7855、watch_seconds MSE 741.8798。
- 回执与核验脚本在本地 `runs/fyp-us-metrics-fix-20261120/`；未启动新训练或更换线上模型。

## 2026-11-20 日更启动参数恢复

指标映射使 inline 监控上下文超过 SageMaker 单值 512 字符限制。已按 [Spec 042](../042-tf-monitoring-context-file/verification.md) 部署文件传递修复、升级该模型 worker 至 revision 22，并修复原 journal 恢复 cutoff 2026-11-19 的同名训练。16:05:55 UTC+8 已启动，训练入口摘要校验通过。未重新生成样本；当前只验收恢复启动，不声称本轮训练或新模型发布完成。
