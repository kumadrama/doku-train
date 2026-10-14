# Spec 010：模型与特征产物 Redis 分发需求

状态：用户已授权本仓实施；代码及验证见 tasks.md、verification.md，云端及消费者尚未接入。

Source: 2026-10-14 用户要求保留 S3，将线上模型和特征产物同步到 Redis，并明确两个服务的
版本一致性、发布顺序和清理逻辑。既有合同见 [Spec 009](../009-daily-tree-model-publication/design.md)。
用户后续明确要求开始实施代码，本轮仅实施 train 仓库，不修改线上资源。

## 1. 当前状态与范围

当前已实现并部署 S3 日更发布，线上所需包包含 Manifest、特征合同、拟合状态和各任务 UBJ。
本仓已新增 Redis 同步实现，尚未部署 Redis 分发或验收 mono 两个服务的版本消费协议。既有 PROD 清理因
consumer_protocol_ready=false 仅 dry-run。

Source: 本仓 comm/daily_publication.py、comm/publication.py、scheduled.py；本会话 PROD 手动
发布/清理及调度验收，iac PR #439/#440/#441、train PR #3/#4。此处是部署验收时状态，非持续监控结果。

本 Spec 增加 S3 → Redis 的分发合同；不改变训练算法、配置命名、日期窗口、发布质量门禁，
不把用户实时特征、训练数据或原始样本写入模型缓存。DTE 不新增部署。

## 2. 必须满足的需求

| ID | 需求 |
| --- | --- |
| R1 | S3 保持不可变制品及发布/生命周期权威，Redis 可从 S3 重建；model_type、model_name、model_version 语义沿用 Spec 009。 |
| R2 | Redis 同步线上加载必需的 Manifest、feature-contract、fitted-feature-state 和全部任务 UBJ；checkpoint、训练输入、训练配置和评估归档保留 S3。 |
| R3 | 特征组装服务与 predictor 独立部署，但一次请求只解析一次版本；RPC 携带名称、精确版本及合同摘要。禁止 predictor 再选 latest 或按目录推测版本。 |
| R4 | 两个服务均后台预加载并校验，热路径使用本地内存；Redis 不承担模型计算，不要求每个预测请求下载模型或访问 S3。 |
| R5 | 先成功发布 S3，再同步并校验完整 Redis 版本，最后原子切换 Redis 线上目录；不得暴露半包。Redis 成功不等于所有服务实例已就绪。 |
| R6 | S3 发布、Redis 分发、消费者就绪分别记录结果。同步失败保留旧线上目录，可独立重试同一版本，不重训、不伪造全链路发布成功。 |
| R7 | 并发、迟到重试、回滚、响应丢失不能让旧操作覆盖新发布。回滚只能选择仍完整保留且可用的精确版本；不能复活 deleting/deleted。 |
| R8 | 版本及其必要文件作为整体清理；沿用超过 30 天才进入候选、替换至少 24 小时、drain 至少 24 小时的保护，不用 Redis TTL 实现保留规则。 |
| R9 | 同时保护 S3 current、Redis 线上 current、活动生产/同步及两个服务的持久 pins。Redis 或消费者引用未知时不删除；超时不自动释放引用。 |
| R10 | Redis 清理与 S3 清理采用明确可恢复状态机；先阻止解析/加载，排空引用，再删 Redis，最后删 S3，删除失败可幂等恢复。 |
| R11 | Redis 数据丢失/故障时，已加载且持有有效 pin 的请求可按原版本继续；冷加载失败应明确返回不可用，不静默切换版本或使用未校验文件。 |
| R12 | 同名合同变化仍允许发布；不增加消费者兼容性质量门禁。消费者不支持新合同需报错并告警，不伪称已加载成功。 |
| R13 | train 管分发/恢复/回滚/清理用例；mono 管本地加载、RPC、就绪及引用；iac 管 Redis、网络/身份/容量与编排；monitoring 管监控告警。 |
| R14 | 凭据由环境装配，TLS/ACL 限制到环境和模型前缀；不使用全库清空或扫描年龄猜测所有权。同步包大小和总容量必须有预算。 |
| R15 | `publish_model.py` 与训练、评估、导出入口同级，负责 S3 发布与 Redis 同步/恢复的用例装配；`run.py` 统一编排，`scheduled.py` 只适配调度参数。底层算法、存储及并发控制留在 comm，不保留第二套完整流水线。 |

Source: 用户在本轮明确确认 publish_model.py 发布入口、run.py 统一编排的结构；随后授权开始实施代码。

## 3. 验收条件

- CLI 与调度适配器调用同一编排用例；独立发布/Redis 恢复不调用训练或导出；重复输入跳过训练仍可恢复分发。
- 更新发生在特征编码与 RPC 之间，predictor 仍只使用请求绑定版本；版本缺失返回明确错误。
- 任意文件写入、读回校验或目录切换失败，线上目录不暴露不完整版本；S3 成功不被回写成训练失败。
- 重试、回滚和 Redis failover 测试证明不混用版本；无法确认恢复状态时停止切换/删除。
- Redis 指针落后 S3、消费者失联、两个服务不同升级节奏时，旧版本仍受保护。
- cleanup dry-run 无持久变更；实际清理验证 current/pins/drain 保护及 Redis/S3 中断恢复。
- 消费者协议和跨存储清理扩展未验收前，真实删除保持关闭。

设计细节见 [design.md](design.md)。实施与证据见 [tasks.md](tasks.md)、[verification.md](verification.md)。
