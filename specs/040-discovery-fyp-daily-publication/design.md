# 设计

`daily_pipeline` 导出并冻结已有 canonical artifact，沿用既有 endpoint 更新和精确模型合成探测。可选 `DiscoveryPublication` 共用 Idea 已验证的 AppConfig CAS publisher 和持久化文档 journal；定义先发布，路由后发布，均完成后才推进原有 high_water。未配置 hook 的任务行为保持不变。

生产初次发布确认 AppConfig `COMPLETE` 可能早于 Data 读取端采纳。完成后若仍读到本次冻结的旧内容，且环境最新部署编号仍是本次部署，则返回 pending，沿用原轮询等待；不新建版本或再次部署。第三种内容或外部部署出现时继续报错，保留 CAS 和漂移保护。

资源标识来自 IaC 的 `discovery_online_release_binding`，通过已有 `prepare_tf_daily_deployment.py --online-release-outputs` 合并为不可变 binding。只授现有 FYP worker 对 Discovery 两个 profile 的读、建版本、部署权限；不授在线 AI 额外 S3 权限。

新模型的 feature-config 字节和去版本后的 signature 必须与当前合同相同。生成独立 release 身份并嵌入 manifest、signature、feature-config、feature-state，保留 DAG、warmup、scenes、require_dump 与原预算。定义文档保留旧路由所引用的定义，覆盖两份配置混读窗口；同版本发布无操作。

原始独立 bootstrap endpoint 以 `discovery_bootstrap_nn` 保留在列表中，供显式 AppConfig 回滚。加载器要求全部列出模型预热通过，备用 alias 不会自动接管失败的新版本；更新窗口中的全新副本可能在配置与 Endpoint 一致后才能启动。日更使用可变 endpoint，上一天模型不作为可调用备用 alias：其权重已被替换。端点更新与 AppConfig 采纳之间可能出现旧版本请求失败，沿用引擎 quality 降级。配置完成不代表所有 AI 副本已采纳；运行验证需要观察 warmup 与实际 RPC 身份。

回滚时先恢复 bootstrap 的独立 endpoint 路由，再回滚 worker binding / 引擎。若回滚共享日更 endpoint，必须协调恢复对应 AppConfig 版本；不能仅回退某一份配置。

现有 AI 镜像已支持该加载器，不需要为本功能增加加载代码或重发 FYP/Idea AI 服务。共享 Python runtime 的变化要求重新生成 Idea bootstrap source pin，此项为构建一致性，不表示部署 Idea。
