# 设计

Source: 用户 2026-11-17 的收敛指示；已执行的 FYP 生产发布材料 `content-rec/model-recovery-release-20261107/{provision_endpoints,prepare_release,publish_config}.py`，对应部署回执 435/436/437。

真实 FYP 业务使用独立版本 Endpoint `doku-prod-fyp-84eb07`。独立日训各配方仍更新自己的实验 Endpoint，没有自动覆盖该业务 Endpoint。树模型 `recommend/train/publish_model.py` 的 S3→Redis 分发是另一条发布实现，不能替代 TF 路由热更新。

## 最小接线

原 `daily_pipeline` 只在存在 `idea_appconfig` 资源接线时启用适配：

1. 沿用原训练、导出和单模型封包。由 `endpoint_prefix + model_version[:12]` 选择独立版本 Endpoint，12 位名称碰撞会拒绝覆盖。
   Idea 的就绪探测调用 Endpoint 默认模型。发布记录保留完整版本和 EndpointConfig，以便追溯。
2. 复制当前 Idea active 的配置模板；特征配置必须相同，模型签名除版本外必须相同。新合同文件与 manifest 逐项校验，保留当前 max_history、batch 和超时。模型配置保留上一版和候选，候选省略 `release.model_version`。发布器从摘要校验过的 active manifest 读取上一版版本，核对显式配置的旧版本字段。AI Engine 用候选 manifest 解析执行与日志身份。部署顺序为先升级 AI Engine，再发布 Idea 日更 worker。
3. 用现有 S3 immutable_put 冻结前后配置及其摘要。原 deploy_step 增加可选 create_only；默认行为保持不变。Idea 的该模式不更新或回滚已有 Endpoint。
4. 原部署器完成 InService 与默认模型合成探测后，先发布模型定义再发布模型列表。沿用既有 AI 双文档加载、合法性校验和原子 generation 替换。
5. 配置两个部署都 COMPLETE 且逐字节读回后，完成原日训 release 记录。未启用 `idea_appconfig` 的 FYP 路径不创建 AppConfig 客户端。

资源接线包含 app/environment/profile/strategy ID 与 endpoint_prefix；不加入观察窗口、排空参数、副本策略或额外开关。配置仍限于 AI 的 2 MiB 协议。

## 重入与失败

hosted-version 使用 LatestVersionNumber CAS；StartDeployment 使用 LatestDeploymentNumber CAS。唯一版本 label 用于响应丢失后查回，不重复分配部署。固定 worker SDK 必须声明该字段，否则在训练/发布前拒绝。当前仓库锁定 botocore 1.43.120 已核验支持。

并发外部改配置、签名变化或 AppConfig 自动回滚都会保留活动 journal 并拒绝标为成功。上一版 Endpoint 一直保留；AI 热加载拒绝时继续使用原 generation。回滚使用 FYP 既有配置路由恢复流程，禁止用旧的 --rollback-version 原地替换某个版本 Endpoint。

模型定义发布完成不冒充业务已全量采纳。首次由发布者做实际 RPC 与自然流量验收；后续自然日训后继续核对模型版本、真实打分和快照。旧 Endpoint 的退役仍沿用受审查的发布流程，本适配不自动删除资源。

如果需要回退到尚不支持省略版本的 AI Engine 镜像，先恢复带 `release.model_version` 的有效定义，
再回退镜像。可从对应定义的已校验 manifest 恢复显式版本。

## 固定 Endpoint

当 Idea binding 声明 `fixed_endpoint: true` 时，日更使用 binding 的 Endpoint 名称。
新模型复用原 `primary` variant 与 SageMaker 蓝绿更新。只有实时路由、上一版 EndpointConfig
及候选签名匹配才提交。发布完成需要默认模型探测回读候选实际版本。失败由 SageMaker 告警
或既有发布器恢复上一份 EndpointConfig。

当前 Idea 线上定义作为稳定特征合同。固定模式在日更期间读取 AppConfig 校验合同和路由，
更新只发生在 SageMaker 和 release journal。导出产物的额外模型版本字段用于实际打分和样本归因。
固定模式的上线顺序是 AI Engine、Endpoint 专属权限和告警、日更 worker 与其两个触发器。

[AppConfig StartDeployment CAS](https://docs.aws.amazon.com/appconfig/2019-10-09/APIReference/API_StartDeployment.html)
