# Spec 031：日更训练与模型监控的发布、配置归属

状态：2026-10-31 用户授权实施代码；本 Spec 的生产迁移尚未执行。PR #30 的 TF 修复已另行完成生产部署。

## 1. 背景与目标

Source: 用户要求将日更训练镜像、ECS 任务版本、模型监控程序版本移到业务发布系统，将日常运行开关和应用处理参数移到 AppConfig；本次要求明确 doku-train、doku-monitoring、doku-iac 分工和生产迁移。

目标是让日常程序发布、暂停恢复和处理预算调整无需修改 Terraform，同时保留不可变制品、可追踪发布、幂等恢复和回滚能力。每天更新模型继续由原日更链路完成，不等于每天构建镜像或发布 IaC。

相关权威材料：[Spec 028](../028-tf-endpoint-rollout/design.md)、[Spec 030](../030-selected-model-monitoring/design.md)、[既有树训练运行开关](../011-appconfig-runtime-policy/design.md)。本 Spec 不改变选模语义、特征合同、28+1 当前窗口、四任务指标、model_name、Endpoint 身份或业务调用方式。

## 2. 现状与范围

Source: 2026-10-31 读取 doku-train main `823bcbf`、doku-monitoring main `89e4174`、doku-iac origin/main `71b0470`。IaC 本地工作区包含旧草稿，设计以远端上述提交为准；生产执行前仍必须重新读取实际资源和版本。

- TF：上游 ECS 完成事件 → ready adapter Lambda 校验逐日产物 → EventBridge → ECS 编排 → SageMaker 训练/导出 → Endpoint 更新。
- `modules/tf_daily_pipeline` 当前由 Terraform 创建任务版本；EventBridge target 与 RunTask 权限绑定具体版本；输入转换器携带 config/binding 坐标与摘要。
- `envs/prod/tf_endpoint/ready_adapter.tf` 当前固定 ready adapter ZIP 与应用设置；迁移必须覆盖此 Lambda，不能只迁 ECS。
- 模型监控 Lambda 当前由 `modules/model_monitoring` 的 ZIP/source_code_hash 更新，环境变量混合资源绑定和扫描预算；消费 SQS，定时补偿并同步 Grafana。
- doku-train 根 `deploy/prod/task-definition.yml` 是已有树任务声明，不是 TF 编排器；新增 TF 独立发布单元，不覆盖它。
- 主线包含未发布到 TF 镜像的 PR #30 合同兼容修复。版本归属迁移与该修复发布必须分别记录，不宣称源码 main 等于所有线上程序版本。
- 当前 doku-tools `deploy-task/internal/ops/svc.go` 的 task 目标只注册任务版本；未据此证明它会切换 EventBridge。平台支持是实施前置条件，不能靠手改 ARN 当成完成。

## 3. 必须满足的需求

| 编号 | 需求与验收 |
| --- | --- |
| R1 | doku-train 独立发布 TF 编排镜像和 ready adapter，doku-monitoring 独立发布模型监控 Lambda。正常代码升级不改 IaC、不连带部署 notifier 或树任务。 |
| R2 | 发布只使用镜像 digest、ZIP 内容摘要与 S3 object_version；资源身份由受信 binding 提供，业务声明不能任意指定账号、角色或目标资源。 |
| R3 | ECS 发布必须验证实际触发目标使用新版本。任务注册成功但 target 未切换视为未激活；失败保持或恢复旧 target，保留旧版本。 |
| R4 | Terraform、发布系统对同一字段只能有一个写入所有者。交接后再次 plan 不回退应用代码或 target 版本，也不忽略整个资源的漂移。 |
| R5 | AppConfig 管 TF 的新训练准入、Endpoint 发布准入，以及监控采集、补偿、看板同步开关和扫描预算。开关不是强杀；已经发起的训练或 Endpoint 更新必须收敛到可恢复状态。 |
| R6 | 暂停期间保留 ready 事件、训练成功制品和监控事件；恢复按有界、幂等流程重放，不能把暂停记成训练成功、业务失败或指标已投递。 |
| R7 | 配置缺失、非法或过期不可静默放行新的训练/Endpoint 更新；已进入操作的收尾不因配置不可用中断。保留故障记录与可恢复状态。 |
| R8 | 离线实验、纯本地训练/导出不读 AppConfig。训练配方及 model_name 继续版本化，冻结配置与 Manifest；超参数和训练天数不变成运行中热修改项。 |
| R9 | 监控保留现有 event_id、payload claim、ack、catalog CAS、扫描游标、Grafana UID 和日志历史；不得因迁移产生多套模型目录或重复曲线。 |
| R10 | 迁移首个版本保持生产模型和指标语义；不为验收重训、不重做样本、不替换线上 Endpoint、不发送模拟故障通知。需要真实发布验收时单独授权并记录。 |
| R11 | 每次执行记录应用 release、config/binding 摘要、AppConfig 版本及开关判断。配置灰度、发布失败、回滚可审计；业务日志不包含 token 或完整样本。 |
| R12 | 保留原 TF 失败通知链路、oncall 和 ACK 行为；暂停单列状态。若有既有新鲜度告警须明确暂停影响，不通过全局关告警掩盖异常。 |

## 4. 三仓职责

| 仓库 | 拥有 | 不拥有 |
| --- | --- | --- |
| doku-train | TF 镜像/ready adapter 制品、独立 deploy 声明、训练配方、应用配置 schema、运行策略读取、暂停记录/恢复、发布兼容验证 | IAM/网络创建、Grafana 编译、业务 A/B 与推荐 serving |
| doku-monitoring | 模型监控 ZIP、独立 deploy 声明、应用配置 schema、采集与扫描预算、暂停补偿、看板同步和验证 | Terraform 资源、TF 训练选模、notifier 的隐式更新 |
| doku-iac | IAM、网络、集群、函数壳、规则/队列/DLQ、日志保留、AppConfig profile/validator/环境、受信 binding、初始制品和所有权交接 | 日常镜像/ZIP 版本、业务开关值和业务扫描预算的持续管理 |

发布平台属于 doku-tools。若 task→EventBridge 激活能力缺失，必须在平台补齐并验证，作为单独依赖；三个业务/IaC 仓库不得新增私有发布器绕过平台。具体接口草案见 design，不视为已实现。

## 5. 配置边界

- AppConfig：运行准入与应用处理预算；配置 schema/默认值随所属业务仓维护，由受控配置发布流程部署内容版本。
- deploy 声明：镜像/ZIP、ECS CPU/内存/临时磁盘、容器启动参数、已冻结 release 配方坐标。
- binding：账号/区域、允许的资源、角色、前缀、AppConfig profile 身份、物理目标；不能由 AppConfig 扩权。
- IaC：SQS batch_size/visibility/retention、Lambda 内存/超时/并发、调度频率、资源日志保留和告警装配，本期不搬迁。应用 batch_budget 与 SQS batch_size 不是同一参数。
- 训练/推理镜像 digest 和配方摘要属于不可变 release；每次 run 冻结，不跟随热配置变化。

## 6. 验收与非目标

验收必须覆盖：无训练/无发布副作用的兼容 smoke；程序升级不产生 IaC 代码差异；target 实际版本核对；应用配置更新无需 Terraform；训练期间关闭发布后可用原制品恢复；SQS 暂停跨重试/进程重启后零丢失；配置读取失败；发布失败与旧版本恢复；迁移后 Terraform 无回退；一次自然日更及监控补偿证据。

不实施 CANGen/Spec 029、不修改模型算法、不开发业务 serving、不重构所有树任务、不迁移所有监控程序、不新增每模型一套看板，不把基础设施紧急暂停能力删除。

待确认的是本文与 design 的整体方案。具体 IAM ARN、生产 release 与资源当前值在实施前盘点填入验收记录，禁止从文档示例直接发布。
