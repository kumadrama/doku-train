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

实际生产回执及执行脚本保留在本地 `runs/fyp-us-release-20261119`，包括 `live-verification.json`、`release-state.json`、`applied-outputs.json` 和 `post-apply-plan.log`；不提交权重、实验配置或运行数据。
