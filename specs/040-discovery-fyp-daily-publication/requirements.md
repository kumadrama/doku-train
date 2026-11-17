# Discovery 复用 FYP 日更与加载

Source: 2026-11-17 用户要求“复用fyp吧”，承接保证日更模型可被 Discovery 加载的要求。

- 复用既有 FYP 日训、制品、官方 TF Serving Endpoint 更新和合成探测；不新建训练任务或推理 Endpoint。
- Discovery AI 保持独立部署，直接使用现有 AppConfig ModelSource、warmup 与原子 runtime 切换。消费端不增加 S3 journal 读取器。
- 日更任务通过可选 `discovery_appconfig` 资源接线，在部署探测通过后更新独立 Discovery profiles；FYP、Idea profiles 不写入。
- 保留特征合同、预算、Discovery 样本隔离及实际 release/version 归因。特征合同变化必须停止发布并人工审查。
- 引擎通过逻辑 alias 选择版本；请求中的后续批次锁定首批实际身份，混用或加载异常走现有降级。
- 当前线上实验保持 10%，不修改客户端。

当前状态：FYP 日更会更新 `doku-prod-tf-rerank-official`，业务 FYP/Discovery AppConfig 仍固定旧版。本变更是为 Discovery 接通发布联动，不宣称现有 FYP profiles 已经每日自动更新。
