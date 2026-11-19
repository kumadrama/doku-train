# Discovery 复用 FYP 日更与加载

Source: 2026-11-17 用户要求“复用fyp吧”，承接保证日更模型可被 Discovery 加载的要求。

- 复用既有 FYP 日训、制品、官方 TF Serving Endpoint 更新和合成探测；不新建训练任务或推理 Endpoint。
- Discovery AI 保持独立部署，直接使用现有 AppConfig ModelSource、warmup 与原子 runtime 切换。消费端不增加 S3 journal 读取器。
- 日更任务通过可选 `discovery_appconfig` 资源接线，在部署探测通过后更新独立 Discovery profiles；FYP、Idea profiles 不写入。
- 保留特征合同、预算、Discovery 样本隔离及实际 release/version 归因。特征合同变化必须停止发布并人工审查。
- 引擎通过逻辑 alias 选择版本；请求中的后续批次锁定首批实际身份，混用或加载异常走现有降级。
- 当前线上实验保持 10%，不修改客户端。

首次改造前，FYP 日更更新 `doku-prod-tf-rerank-official`，业务 FYP 与 Discovery 的 AppConfig 仍指向固定版本。首轮实施先接通了 Discovery 的日更发布。

## FYP 线上日更

Source: 2026-11-19 用户授权修改 `content-rec-svc`，接通 FYP 与 AI Engine 的版本切换，先验证 DTE 双版本、回退和 RT，再发布 `b710e4b2`。

- 显式配置的 FYP 路由请求稳定模型别名，每次请求使用 AI Engine 返回的实际 release 身份进行归因。
- 在原有训练任务中启用 FYP 模型发布。FYP 与 Discovery 使用同一份通过端点探测的模型制品，各自发布独立配置。
- FYP 保留独立端点的旧版用于回退。变更共享端点前，需要确认运行副本已经切到可用的回退模型。
- Discovery 实验保持 10%，保留各业务现有的特征、样本和实验配置。
