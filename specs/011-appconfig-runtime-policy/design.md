# 设计

Source: 用户已确认的 #7–9 方案；代码入口 `scheduled.py`、`run.py`、`publish_model.py`。

`runtime_policy.py` 是运行入口适配，不进入模型/算法层。S3 制品路径要求明确的 DOKU_ENV=dte/prod，读取 doku application 下 doku-train/tree；file 本地实验不访问 AppConfig。复用现有 boto3，无新增依赖；已检查本仓没有可复用的 Python AppConfig 客户端。每次读取建立新会话，3/5 秒超时、最多两次尝试，校验显式布尔、正整数及大小预算顺序，错误不带 SDK 原始消息。

训练入口先检查 training；发布入口及 publish_prepared 在实际 S3 发布前重读 publication/redis。训练期间关闭发布会归档未发布诊断产物和 checkpoint，返回 skipped，registry 保留 paused run。恢复后可按现有导出/发布命令发布保留的训练目录，也可重跑 daily；paused 不当成已发布去重记录。

一次发布协议以发布前检查为准：进入 S3 发布后继续完成 Redis 同步，不在两阶段之间重新判定开关，以免主动制造版本分裂。开关不是事务中断器，已开始的操作不强杀；Redis 错误继续走既有 DistributionFailed / 精确版本恢复协议。重复事件触发的独立同步重新检查开关。cleanup 只检查自己的开关，Redis 分发暂停不解除清理的引用和退役保护。

云端手工 daily/publish/rollback/redis-sync/resume/rebuild 同样检查相应开关；取消孤儿操作、pin 和协议就绪命令保留既有显式确认，不由运行开关替代。只读解析和本地训练/export 不受影响。

Redis timeout_seconds、max_file_bytes、max_package_bytes、max_model_bytes 从新快照装配；Secret ID、ENVIRONMENT、CLUSTER 仍是部署绑定。redis=false 仍保留 Distribution 对象与原一致性检查，不回退 S3-only。

跳过日志有 status=skipped，不写 completed=true/false，因此既有失败指标不会把主动暂停当失败，成功新鲜度仍可能告警。读取失败继续报错并触发原失败路径。

## 发布顺序（未执行）

1. shared profile/schema → 环境初始化版本及 scoped IAM；prod 四项 true，DTE 四项 false（DTE 无自动训练工作流，人工验证前明确开启所需项）。
2. DTE 离线/受控验证后发布业务镜像，再做 prod 发布；每步另行获取部署授权。
3. 验证后单独移除 tree_publication_enabled / tree_redis_enabled 和旧数值环境变量；bootstrap PR 保留旧镜像仍依赖的字段、原调度启停及告警。训练 timer 永久保持 false，不因迁移打开。

回滚前确保旧镜像所需环境变量存在，运行开关不会控制旧代码。不得把 AppConfig 的关闭当成旧镜像已停机。
