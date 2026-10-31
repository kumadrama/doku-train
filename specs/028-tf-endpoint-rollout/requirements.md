# Spec 028：TF 真实模型 Endpoint 与自动日更需求

状态：需求与设计已获用户确认，已授权本地代码实施；实现/验证进度见 tasks.md 和 verification.md，尚未授权云部署。

Source: 2026-10-30 用户要求按已讨论方案整理 requirements/design，覆盖真实模型核验、最小推理镜像、首次 Endpoint 部署与每日自动更新，业务推荐服务暂不做；Spec 027 已合并的本地能力为基础。

## 1. 目标与当前状态

目标：将 `doku-exp-f024-ln-lr001-20261029-134714` 的实际最优 checkpoint 部署为服务端可测试的固定名称 SageMaker TF Endpoint，随后自动执行 28 天 train + 1 天 validation、选模、导出、发布和 Endpoint 更新。

当前：Spec 027 已实现本地日更、导出、发布状态机和 AWS 模拟验证；尚未核验该真实实验、交付实际推理镜像或完成云端 Endpoint/日更验收。不能把代码合并、S3 上传或 API 已受理当作上线成功。

本次不增加“上传 S3 后结束”作为最终交付模式。S3 为不可变制品存储，Endpoint 为推理交付终点。

## 2. 范围与仓库责任

| 仓库/系统 | 本次责任 |
| --- | --- |
| doku-train | 历史 run 核验、模型导出、训练与编排镜像、首次发布和每日 Endpoint 更新、对账及回滚 |
| doku-train / recommend_tf/inference | 最小推理镜像、制品加载、请求张量适配、四任务响应；复用 TensorFlow Serving |
| doku-iac | 独立角色、网络、存储权限、ECS 任务、环境绑定及上游成功触发连接 |
| 上游数据系统 | 提供受信任样本发布成功证据和精确日 Manifest；必要时实现 ready adapter |
| doku-monitoring | 训练/发布失败、数据新鲜度、Endpoint 执行健康告警及部署告警对接 |

按用户后续明确指示，最小推理实现先放 `recommend_tf/inference/`，与 train 分离。不实现业务推荐服务、特征在线获取、召回、排序、实验分流或调用失败业务降级，不修改 doku-mono。跨仓仅依赖版本化合同和制品，不 import 训练内部实现。

## 3. 功能需求

### R1 真实模型核验

使用精确实验产物地址；核验成功状态、选模记录、实际 checkpoint、配置、Feature State、冻结输入及 lineage。现有本地记录显示 epoch 21，仅作待验证线索，不能硬编码替代 selection。旧实验没有 COMMITTED 时，生成经核验的文件摘要 evidence 和审核报告；文件存在及摘要可计算不等同于来源可信。

要求核对运行身份、选模与 checkpoint 身份、配置/状态一致性及原始来源记录。缺失证据必须失败并列出缺项，不凭空补造配置、词表或权重。首次使用历史实验真实日期，不要求重训 28+1，也不伪称历史模型具有新数据截止日。

### R2 导出与正确性

复用 Spec 027 exporter，交付 SavedModel、特征配置/状态、签名、lineage 和摘要。验证所选 checkpoint 与 SavedModel 重载后的四任务最终预测一致，覆盖默认值、OOV、空历史、截断、padding 和批量。核实 watch_seconds 的实际 predict_fn 及输出单位，不擅自追加变换。失败不改写原成功 run。

### R3 最小推理镜像

使用 CPU TensorFlow Serving；固定镜像和兼容依赖版本。加载 `doku-tf-inference/v1`，校验包清单和签名。请求为无标签具名批量特征，预处理严格遵循导出配置；词表/状态不可在线重新拟合，不沿用训练样本筛选条件。

响应仅包含四任务 completion、fast_swipe、immersive_click、watch_seconds 和实际模型版本，保持候选顺序。版本来自已加载包，不接受客户端指定版本冒充。仅在模型加载且可推理后报告健康；无效请求返回明确错误，不返回伪造分数。

镜像遵循 SageMaker 推理容器合同，不包含业务数据读取或推荐排序。不能假定当前模型包可直接由任意官方镜像加载；需实际容器测试。

### R4 首次部署

使用已核验历史 run 走 bootstrap，不提交新训练。先交付真实模型包和镜像，再以显式环境、角色、实例及告警配置部署固定名称 TF Endpoint。部署完成须同时满足目标 EndpointConfig 生效、服务可用、合成请求返回正确合同及实际模型版本。首次没有旧 TF 版本，不声称能由发布器回滚到树模型。

交付接入说明：Endpoint 名称、区域、调用身份与权限、请求/响应合同、合成样例、错误语义和模型版本。容量未确认时明确测试范围，不承诺生产 SLA。

### R5 每日自动训练与更新

复用 Spec 027 流水线；online_config 为 training_date=28、validation_date=1，配方与镜像固定。上游截止日 D 对应 train=[D-28,D-1]、validation=[D]，UTC 连续 29 天齐全且有成熟性证据。缺日不缩窗、不提交训练。每天从头训练，仅 train/validation，无隐式 test 或新增 AUC 门禁。

由上游成功发布触发一次性 ECS，ECS 提交 SageMaker Training Job 并负责导出及发布；不新增常驻训练 timer，不依赖树模型训练成功。正常日更只更换模型产物，固定 Endpoint 调用名称。

### R6 幂等、恢复与回滚

重复事件不得重复训练或发布；阶段状态持久化；恢复使用冻结输入，不解析新的上游指针。旧任务不能覆盖新数据版本。保留现有 ECS 执行者停止核验及 API 不明确时停止盲重试策略。

首次发布后验证后续更新：新资源就绪后 ALL_AT_ONCE，部署探测及告警失败恢复已保留旧配置。显式回滚不降低自动发布数据高水位。外部 Endpoint 修改应报冲突，不覆盖。自动触发启用前完成失败恢复和回滚演练。

### R7 安全、资源与监控

环境、区域、身份、S3/KMS、实例、执行预算和镜像 digest 均显式绑定；不新增生产默认值。训练、编排发布、推理加载、调用者权限分离。测试只用合成请求，不将真实样本或凭据写入源码、镜像或日志。

记录真实模型在明确 CPU/线程/候选批量/并发下的吞吐、延迟分位数、RSS；与调用方确认容量后才能声明支持目标负载。设置执行失败和模型数据新鲜度监控；不恢复训练指标质量门禁。暂不自动清理历史资源，保护在线与回滚引用。

## 4. 验收顺序

1. 真实历史 run 审核报告及证据完整，成功导出并通过数值一致性验证。
2. 最小推理容器完整请求链路与离线结果一致；健康检查、非法输入和版本响应测试通过。
3. 首次 Endpoint 实际调用通过，服务端接入说明可复现。
4. 手动触发一次完整 28+1 训练→更新流程，验证重复触发、故障恢复与回滚。
5. 启用自动触发，确认实际日更及运行监控；不能仅凭 Terraform apply 成功完成验收。

## 5. 实施/部署前待确定项

以下不阻塞本文整理，但阻塞相应代码落位或云验收：

- 已确定代码位置：`doku-train/recommend_tf/inference/`；未来拆仓仍通过制品合同解耦。
- [NEEDS CLARIFICATION: 历史 run 的精确 S3 地址及可信来源证据。]
- [NEEDS CLARIFICATION: 目标环境、区域、Endpoint 名称、S3/KMS、网络及角色绑定。]
- [NEEDS CLARIFICATION: CPU 推理镜像兼容版本、首期实例/数量、请求大小和批量上限。]
- [NEEDS CLARIFICATION: 上游成功信号的实际载荷、ready adapter 所属代码、失败事件重试机制。]
- [NEEDS CLARIFICATION: 部署观察/告警参数、容量目标及云部署执行授权。]

## 6. 关联文档

- [设计](design.md)
- [Spec 027 运行合同](../027-tf-daily-serving/operations.md)
- [Spec 027 验证记录](../027-tf-daily-serving/verification.md)

## 本地实施补充

Source: 本次用户仓库位置修订及实际代码。生产 inference 仅使用标准库适配并调用官方 CPU TensorFlow Serving 2.11.1，不导入 train。采用 HTTP 薄桥接承载原始特征协议，实际容器已加入 CI。IaC 仅新增隔离模块与生成器，未实例化环境；现有仓库未提交改动保持不动。EventBridge 触发默认关闭，新增 DLQ、任务退出失败与发布心跳告警；心跳不等于模型数据截止日新鲜度。后者及实际上游成熟性证据仍需跨仓装配。

Source: 2026-10-30 用户最新指示：第④⑤步继续实施，窗口改为 28 天训练 + 1 天 validation，取代此前等待 31 天的安排。
