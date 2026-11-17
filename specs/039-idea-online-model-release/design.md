# 设计

Source: 用户 2026-11-17 的收敛指示；已执行的 FYP 生产发布材料 `content-rec/model-recovery-release-20261107/{provision_endpoints,prepare_release,publish_config}.py`，对应部署回执 435/436/437。

真实 FYP 业务使用独立版本 Endpoint `doku-prod-fyp-84eb07`。独立日训各配方仍更新自己的实验 Endpoint，没有自动覆盖该业务 Endpoint。树模型 `recommend/train/publish_model.py` 的 S3→Redis 分发是另一条发布实现，不能替代 TF 路由热更新。

## 最小接线

原 `daily_pipeline` 只在存在 `idea_appconfig` 资源接线时启用适配：

1. 沿用原训练、导出和单模型封包。由 `endpoint_prefix + model_version[:12]` 选择独立版本 Endpoint，12 位名称碰撞会拒绝覆盖。
2. 复制当前 Idea active 的配置模板；特征配置必须相同，模型签名除版本外必须相同。新合同文件与 manifest 逐项校验，保留当前 max_history、batch 和超时。配置只保留上一版与候选定义，不再引用未使用的 mutable route。
3. 用现有 S3 immutable_put 冻结前后配置及其摘要。原 deploy_step 增加可选 create_only；默认行为保持不变。Idea 的该模式不更新或回滚已有 Endpoint。
4. 原部署器完成 InService 与显式模型名合成探测后，先发布模型定义再发布模型列表。沿用既有 AI 双文档加载、合法性校验和原子 generation 替换。
5. 配置两个部署都 COMPLETE 且逐字节读回后，完成原日训 release 记录。未启用 `idea_appconfig` 的 FYP 路径不创建 AppConfig 客户端。

资源接线包含 app/environment/profile/strategy ID 与 endpoint_prefix；不加入观察窗口、排空参数、副本策略或额外开关。配置仍限于 AI 的 2 MiB 协议。

## 重入与失败

hosted-version 使用 LatestVersionNumber CAS；StartDeployment 使用 LatestDeploymentNumber CAS。唯一版本 label 用于响应丢失后查回，不重复分配部署。固定 worker SDK 必须声明该字段，否则在训练/发布前拒绝。当前仓库锁定 botocore 1.43.120 已核验支持。

并发外部改配置、签名变化或 AppConfig 自动回滚都会保留活动 journal 并拒绝标为成功。上一版 Endpoint 一直保留；AI 热加载拒绝时继续使用原 generation。回滚使用 FYP 既有配置路由恢复流程，禁止用旧的 --rollback-version 原地替换某个版本 Endpoint。

模型定义发布完成不冒充业务已全量采纳。首次由发布者做实际 RPC 与自然流量验收；后续自然日训后继续核对模型版本、真实打分和快照。旧 Endpoint 的退役仍沿用受审查的发布流程，本适配不自动删除资源。

[AppConfig StartDeployment CAS](https://docs.aws.amazon.com/appconfig/2019-10-09/APIReference/API_StartDeployment.html)
