# 生产部署与接管记录

Source: 2026-11-26 用户明确授权“部署并接管线上”；AWS STS、EventBridge、ECS、SageMaker、S3 只读快照。

## 当前状态

**未部署、未接管。** 本次没有修改线上 Endpoint、权重、业务 AppConfig、日更规则或任务。原日更继续运行。

身份核验：`doku-prod-developer`，AWS account `703844615824`，region `us-east-2`。
本地审计快照目录：`/private/tmp/model-management-rollout-20261126/`，包含规则/target、Task Definition、冻结配置、binding、发布 journal。快照不作为可变线上配置的长期权威。

| 模型 | 旧 TD revision | max_epochs | 快照 high_water |
| --- | --- | --- | --- |
| inner_watch_lhuc_m512_e16_t64 | tf-rerank-daily:40 | 15 | 2026-11-24 |
| inner_watch_lr003 | tf-rerank-daily:32 | 30 | 2026-11-24 |
| inner_watch_w02_lr003 | tf-rerank-daily:33 | 30 | 2026-11-24 |
| tf_rerank_ln_lr002_3feat_us_or_unknown | tf-rerank-daily:30 | 30 | 2026-11-24 |
| tf_rerank_ln_lr002_3feat | tf-rerank-daily:34 | 30 | 2026-11-24 |
| idea_mapped_rm32_drop10 | idea-mapped-daily:9 | 20 | 2026-11-25 |

快照时 FYP 任务族有 5 个运行中的 ECS task；不能停止或将它们记为已排空。Idea 无运行中 task，但同时存在 scheduled 和 published 两个入口，必须共同排空。

## 真实资源兼容修正

- FYP 各存量模型使用不同 inference role，official Endpoint 名称也不属于新 `doku-prod-fyp-` 前缀。新增 `adoption_services` 精确登记 Endpoint→role；只接管已核验旧资源，不扩大新服务命名空间。
- Idea 旧 binding 没有顶层 `model_name`，已有 `monitoring.model_name` 与训练 config 一致。接管接受这个既有身份字段，仍拒绝身份不一致或完全缺失。
- Idea 两个旧 target 没有显式 RetryPolicy。IaC 声明 AWS 现有默认值：86400 秒、185 次；不缩短排空时间。Source: [AWS EventBridge retry policy](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-rule-retry-policy.html)。
- FYP 五个 worker target 已显式配置 86400 秒、3 次。共享 upstream/recovery rule 只能移除相应模型 target，保留其他生产者及消费者。

## 发布阻塞（首次尝试）

本机 GitHub 网络不可用，已观察：

- SSH 22：`Connection closed ... port 22`。
- HTTPS fetch：`LibreSSL SSL_connect: SSL_ERROR_SYSCALL`。
- GitHub 官方 SSH 443：`Connection closed ... port 443`。
- GitHub CLI：`Get https://api.github.com/...: EOF`。
- 规范 `dkd deploy plan iac envs/prod --env prod --remote-path envs/prod/model_monitoring --repo kumadrama/doku-iac` 在解析 main 时失败：`resolve kumadrama/doku-iac branch main ... EOF`。没有 job id，未执行 apply。

GitHub connector 可创建功能分支和 PR；它不能代替 DKD 的构建、计划与部署控制面依赖。不使用临时脚本更新 Lambda 程序或本地 Terraform apply 绕过发布链路。

## 网络重试与部署准备进展

用户要求重试后，GitHub SSH/API 一度恢复。已合并管理 CLI PR #3、监控 PR #438 和制品校验修正 PR #439。训练 PR #80、IaC PR #1143 尚未合并；不能将源码合并或开发镜像构建视为生产部署。

- 开发运行时 CodeBuild `doku-train_tf-pipeline:b2a9b6bc-7957-4c14-8a02-606dd412cbeb` 成功，离线导入与锁定 SDK 检查通过；尚未构建 main 正式镜像。
- 监控基线 DKD plan `dj_249f8185-ace7-452f-bdec-a53a5adf4f73` 成功，无资源变更。
- Lambda plan `dj_6aef5d87-b79b-4fbd-8c4c-52b5fb6df637` 因 S3 缺少服务端 ChecksumSHA256 失败，未执行 apply。原始 ZIP 字节校验通过后，补传具备服务端校验的新版本并更新发布引用。
- FYP/Idea 与隔离 DTE 装配 `terraform validate` 通过；模块 `terraform test` 2 passed；应用输出 binding 合并单测 1 passed。平台调度保持关闭，尚未创建云资源。
- 重试 Lambda plan 时 GitHub HTTPS API 再次持续 EOF，未创建部署任务。SSH ls-remote 与 push 可用，但不能替代 DKD 的 API 依赖。当前等待 API 网络恢复，未执行任何 production apply、Endpoint 更新或旧日更入口切换。

## 恢复后执行顺序

1. 更新远端主线并复核四仓 PR/CI，按 PR 合入；平台代码构建固定 release image 和 controller/resource ZIP。旧线上任务继续使用原冻结运行时。
2. 完成 FYP/Idea 平台环境装配：组专用资源模板、运行时镜像/worker revision、显式 legacy Endpoint/role、read/write prefixes、legacy ingress targets、监控 registry snapshot 来源。模板不能写业务训练配方和调用方路由。
3. 通过 DKD plan 审查共享资源与 collector 接线，只 apply 本次范围。新 controller tick 保持关闭，输出生成 client/code bindings。
4. DTE 合成训练→导出→服务探针→重复触发/配置修改/恢复验收；核验实际镜像、IAM 和日志。新组的监控来源/权限先就绪，再启用 tick。
5. 重新抓取线上快照，等待旧任务完成；按管理 CLI 先 FYP LHUC 与 Idea 各一例 plan/apply，移除精确旧入口后等待全部投递窗口排空。排空中同一 pin 查询/恢复，不提前激活 managed ownership。
6. 验证 pilot 登记状态、配置摘要、原 EndpointConfig/权重及调用方 AppConfig 未变后，逐个接管剩余 FYP。验证下一次 managed 日更的固定 Endpoint 更新及监控身份。
7. 记录实际 deployment/job/registry receipts 和业务探针结果后，才能标记“已部署并接管”。

## Worker release declaration validation

DKD requires three-segment PSM names. Worker release units use `doku-modelworker-smoke`, `doku-modelworker-fyp` and `doku-modelworker-idea`; ECS family and log-group identities remain unchanged. The old four-segment declaration was rejected locally before any deployment job was created.
