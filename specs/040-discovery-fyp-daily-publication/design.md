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

## Discovery 固定 Endpoint

主模型导出保留 FYP 原有的 `serving_default` 输入和分数输出，并增加 Discovery 专用的 `serving_with_model_version` 签名。两个签名引用同一组分数张量，Discovery 的签名额外返回实际执行模型的摘要。

先升级 Discovery 引擎。下一次共有主模型完成端点更新和现有探测后，发布器依据制品中的签名标记一次性切换 Discovery 的独立配置，保留 bootstrap 别名。后续日更在同一 Endpoint 原地更新模型，验证特征合同兼容后复用已生效的 Discovery 配置。FYP 继续按其原有版本路由和回退流程发布。

## FYP 三路实验模型日更

`inner_watch_lr003`、`inner_watch_w02_lr003` 和 `tf_rerank_ln_lr002_3feat_us_or_unknown` 使用独立的 FYP 发布绑定，其余 FYP 发布流程沿用现有入口。每个版本使用带摘要的 SageMaker Endpoint，只允许创建新端点或恢复同一份已记录的部署。发布器验证完整模型合同和精确版本推理探测后，将新定义加入 `models-config-prod`，确认回读后才切换该模型在 `fyp-model-list-prod` 的别名。候选版本与上一版本的定义都保留供混合读和回滚。

主模型和三路实验模型在创建 AppConfig 发布计划前共用按 ETag 更新的 S3 锁。锁记录发布任务的操作 ID 和 journal 路径，ECS 重启后由同一任务接续；外部配置变更由 AppConfig 发布器的版本和部署编号检查保护。三路待所有 AI Engine 副本报告加载新别名后，才将发布记录标为完成。如果五分钟后还有副本未加载，使用已保存的 AppConfig 文档切回旧别名，等待副本加载旧版本，再记录回滚。模型定义仍可用于排查，训练产物可从现有 journal 追溯。

生产先刷新现有主模型 worker 的绑定，使其使用同一个发布锁，再启用一路实验 worker。每个 worker 经已有 `prepare_tf_daily_deployment.py --online-release-outputs` 从已应用的 IaC 输出生成绑定并接入原 EventBridge 目标。日更的 Ready Lambda ZIP 与任务定义必须使用相同的配置及绑定 SHA 和 URI。Ready event 和恢复队列读取已经冻结的请求，换绑前须核对 journal 和未完成的请求。试跑需核对 AppConfig 两次部署的次序、AI Engine 实际业务请求版本及完整失败计数。所有旧 Endpoint 在实际路由和排空核对前继续保留。每次试跑前核对 SageMaker 配额和当前容量，为候选版本预留容量，定期清查已经退出使用的版本 Endpoint。
