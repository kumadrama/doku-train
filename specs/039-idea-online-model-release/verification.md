# 验证

2026-11-17，本分支仅准备代码和测试，没有发布 worker、AppConfig 或 Endpoint。

- 只读确认 FYP 现有独立业务版本发布材料及已完成回执。提供的 TD17 binding 实际指向 `doku-prod-fyp-inner-watch-w02-lr003`，无业务模型列表/定义接线；不能把日训 Endpoint 更新误称为业务自动切流。
- 完整 inference 测试 20 项通过，日训 runtime-policy/Idea-window 测试 9 项通过；Ruff 通过。
- 新增测试覆盖独立版本命名、禁止更新/回滚已有 Endpoint、原 FYP 默认完成行为、配置模板与旧定义保留、models→list 顺序、重复恢复不重复发布、签名变化拒绝、外部更改保护、自动回滚不触旧路由，以及 CAS 的响应丢失和最后时刻竞争。
- 发布窗口已用现有 Idea TD5 做纯只读 SDK 检查：boto3/botocore 1.43.120、Python3.10.22、LatestDeploymentNumber=true、退出0。因此只需更新既有源码 bundle pin，不升级基础镜像或新增构建项目。
- SDK 官方 wheel SHA256 与 recommend_tf/uv.lock 匹配，botocore 1.43.120 的 StartDeployment shape 含 LatestDeploymentNumber；运行时不允许降级无 CAS。
- 完整 TensorFlow 训练测试需要仓库锁定 Python3.10/TF 环境，后续由现有 CI 执行。此次没有部署新执行服务，没有共包多模型，也没有运行新 observer。

当前 5d9e 独立 Endpoint 稳定化和首次实际 RPC 由发布窗口单独验收。此处不将配置 COMPLETE 当作每个 AI 副本已采纳，也不宣称下一次自然日训跨版本链路已经通过。
