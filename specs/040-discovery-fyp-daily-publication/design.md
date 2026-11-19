# 设计

`daily_pipeline` 导出并冻结已有 canonical artifact，沿用既有 endpoint 更新和精确模型合成探测。可选 `DiscoveryPublication` 共用 Idea 已验证的 AppConfig CAS publisher 和持久化文档 journal；定义先发布，路由后发布，均完成后才推进原有 high_water。未配置 hook 的任务行为保持不变。

生产初次发布确认 AppConfig `COMPLETE` 可能早于 Data 读取端采纳。完成后若仍读到本次冻结的旧内容，且环境最新部署编号仍是本次部署，则返回 pending，沿用原轮询等待；不新建版本或再次部署。第三种内容或外部部署出现时继续报错，保留 CAS 和漂移保护。

资源标识来自 IaC 的 `discovery_online_release_binding`，通过已有 `prepare_tf_daily_deployment.py --online-release-outputs` 合并为不可变 binding。只授现有 FYP worker 对 Discovery 两个 profile 的读、建版本、部署权限；不授在线 AI 额外 S3 权限。

新模型的 feature-config 字节和去版本后的 signature 必须与当前合同相同。生成独立 release 身份并嵌入 manifest、signature、feature-config、feature-state，保留 DAG、warmup、scenes、require_dump 与原预算。定义文档保留旧路由所引用的定义，覆盖两份配置混读窗口；同版本发布无操作。

原始独立 bootstrap endpoint 以 `discovery_bootstrap_nn` 保留在列表中，供显式 AppConfig 回滚。加载器要求全部列出模型预热通过，备用 alias 不会自动接管失败的新版本；更新窗口中的全新副本可能在配置与 Endpoint 一致后才能启动。日更使用可变 endpoint，上一天模型不作为可调用备用 alias：其权重已被替换。端点更新与 AppConfig 采纳之间可能出现旧版本请求失败，沿用引擎 quality 降级。配置完成不代表所有 AI 副本已采纳；运行验证需要观察 warmup 与实际 RPC 身份。

回滚时先恢复 bootstrap 的独立 endpoint 路由，再回滚 worker binding / 引擎。若回滚共享日更 endpoint，必须协调恢复对应 AppConfig 版本；不能仅回退某一份配置。

现有 AI 镜像已支持该加载器，不需要为本功能增加加载代码或重发 FYP/Idea AI 服务。共享 Python runtime 的变化要求重新生成 Idea bootstrap source pin，此项为构建一致性，不表示部署 Idea。

## FYP 与 Discovery 联合发布

生产 worker 可增加 `fyp_appconfig`。在首次启用时，发布器保留现有独立端点作为 FYP 回退，并将已验证的新制品定义加入 FYP 模型配置。旧调用端仍可按原 release ID 请求。升级后的调用端可使用模型别名，在推理完成后校验服务、模型和特征合同，并记录实际版本。

后续日更开始时冻结两套 AppConfig 文档。若 FYP 当前路由使用即将更新的官方端点，先通过 AppConfig 切换 FYP 别名到独立回退端点。发布器等待当前 ECS 任务都记录该路由的激活事件，再更新官方端点。合成探测成功后依次发布 Discovery 和 FYP。状态保存各自的 AppConfig 版本和部署编号，重跑时从已记录的阶段恢复。

FYP 的 `training_dump`、特征链和请求预算来自 FYP 配置。新模型必须维持已有特征合同与推理签名。回退在 FYP 模型列表内切换别名，旧版端点持续可用。
