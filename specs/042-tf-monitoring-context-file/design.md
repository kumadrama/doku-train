# 设计

Source: 已确认需求；`daily_pipeline.prepare_job`、`cloud_train.main`、`ReleaseState.save`。

## 当前故障

生产美国/未知国家模型在 cutoff 2026-11-19 的 CreateTrainingJob 前失败。上下文 572 字符，超过接口单值 512 限制；压缩 JSON 空白后仍有 541 字符。journal 的 training 阶段表示请求已冻结，并不证明作业已创建。线上仍使用 cutoff 2026-11-18 的成功版本。

## 目标与实现

`daily/monitoring-context.json` 通过原有 immutable_put 上传，文件内容保留全部字段。环境变量 `DOKU_MONITORING_CONTEXT_PATH` 指向 daily File channel 下该文件，`DOKU_MONITORING_CONTEXT_SHA256` 固定摘要。生成请求及实际创建作业前验证环境变量单值长度。新增训练容器 reader 在 daily_execute 前读取并校验；已冻结的合法 inline 请求继续可读，但两种传递方式同时出现时拒绝。

训练镜像与编排镜像须配套升级；训练源码摘要须与新镜像一致，不以旧镜像读取新协议。镜像版本变动必须进入真实 lineage，不伪称原训练代码完全相同。

## 旧 journal 恢复

`python -m recommend_tf.platforms.sagemaker.repair_monitoring_context` 默认只读输出原请求与候选请求；只有显式 `--apply` 才上传上下文和更新 journal。调用者显式提供 binding、operation、新训练镜像 digest 和训练源码 SHA256。

仅接受 active training、超长旧 inline 上下文、准确匹配的 STOPPED executor、确定不存在的同名 SageMaker job、原 operation 的 daily File/S3Prefix 输入及正确配置摘要。权限错误不视为不存在。只修改监控传输字段、训练镜像及源码摘要；原配置、样本输入、run URI、job 名、operation 和高水位不变。原请求存入 monitoring_context_repair，journal 通过 ETag 条件更新。竞争失败保留新持有者；可遗留相同内容的不可变上下文，重试安全。

工具不启动训练或发布。修复后原恢复入口对账同一确定性 job 名，沿用原输入。存在任何状态的同名作业均拒绝修复，已成功输出继续按原恢复流程导出发布。

本次限定模型的镜像、接线与验收顺序见[生产恢复计划](deployment-plan.md)。

共享运行时代码进入 Idea 源码包，按仓库打包工具同步本地部署清单的 ARTIFACT_KEY/ARTIFACT_SHA；这只固定当前源码，不代表部署 Idea，未来发布也须使用配套训练镜像。

2026-11-21 配套镜像升级、指标补报和树分发恢复见 [执行记录](rollout-20261121.md)。
