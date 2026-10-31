# Spec 028 本地交付与部署操作

本文件描述已实现入口；所有环境值必须显式传入。2026-10-31 已完成真实模型部署、28+1 日更、回滚与恢复验收，并启用上游成功事件自动触发；见 verification.md 最终记录。

## 1. 历史 run 核验

在具有只读 SageMaker/S3 权限的环境中，传入实际任务名称、其 DOKU_ARTIFACT_URI 对应前缀及原始配置文件。原始配置文件的字节摘要须匹配任务的 DOKU_CONFIG_SHA256；不能使用随意重新格式化后的 JSON。

```sh
python -m recommend_tf.platforms.sagemaker.audit_run \
  --region REGION --job-name JOB_NAME --run-uri s3://BUCKET/EXACT_RUN \
  --source-config /work/original-config.json --outputs /work/outputs.json \
  --output /work/audit --download-limit-bytes EXPLICIT_BYTES
```

审核先查询真实任务 Completed 状态、任务归档地址及原始配置摘要，再下载 run 产物；记录 S3 ETag/VersionId，下载受磁盘预算约束。核对 config、状态/选模、依赖锁后，调用 exporter 完成完整身份交叉校验与 CPU reload parity。最后再次检查源对象未变化，才写 `evidence.json`。源码环境摘要包含测试，而训练 lineage 源码摘要不含测试，二者记录在报告中，不错误地宣称它们应相等。

输出包含审核报告、下载 run、导出包；失败只有失败报告，没有可 bootstrap 的 evidence。报告来自任务与存储访问证据，不是数字签名/第三方证明；历史产物缺失时需要追溯原来源，禁止补造。

## 2. 镜像上下文

```sh
python -m recommend_tf.platforms.sagemaker.build_context --kind pipeline --output /tmp/pipeline-context
python -m recommend_tf.platforms.sagemaker.build_context --kind train --config /work/original-config.json --output /tmp/train-context
python -m recommend_tf.platforms.sagemaker.build_context --kind inference --output /tmp/inference-context
```

只生成 create-only 上下文，不构建或上传。pipeline/train 使用 uv.lock 导出带 hash 的依赖；pipeline 构建要求 PYTHON_IMAGE 指定固定 Python 3.10 CPU 基础镜像。推理镜像与参数详见 [inference README](../../recommend_tf/inference/README.md)。新镜像需将最终 digest 写入 binding；不使用 latest 发布。

## 3. ECS 启动输入

executor 参数为 `--region`、`--config-uri/--config-sha256`、`--ready-uri/--ready-sha256`、`--binding-uri/--binding-sha256`；所有 JSON 有 8 MiB 上限，先验证字节摘要再解析。执行者身份来自 ECS v4 metadata，检查 region/cluster 一致后调用已有 durable pipeline。不会解析主机当前日期猜测 cutoff。

首次额外使用 `--bootstrap-run` 和 `--bootstrap-evidence-uri/--bootstrap-evidence-sha256`；ready.cutoff 必须对应历史 run 的 validation_to。显式回滚使用 `--rollback-version`，失败重训使用 `--retry-id`，沿用 Spec 027 状态语义。

IaC 绑定生成器 `scripts/gen-tf-daily-binding.py` 接受显式 settings 和 Terraform root output `tf_daily_binding_resources`；不自动推断环境、不上传。该 output 由环境装配提供以下键：environment、region、ecs_cluster、kms_key、state_uri、work_prefix、model_prefix、inference_role_arn、training_role_arn、endpoint_name。当前未为未知环境添加实例。

## 4. 环境装配和触发

新增 IaC 模块位于 `doku-iac/modules/sagemaker_inference_access` 与 `modules/tf_daily_pipeline`。依赖显式角色、网络、镜像、S3/KMS 权限和告警接收主题。发布器接管 Endpoint 的当前配置，Terraform 不管理每日模型版本。

最小推理 `deployment.container_environment` 必须配置 README 中的所有预算；probe 使用原始特征协议，version_field=model_version、outputs_field=predictions，四任务与 outputs 一致。告警名称必须来自实际 Endpoint 健康监控，不能用任意占位名代替。

上游事件合同：source/detail-type 由环境指定；detail 包含 version=`doku-tf-ready-event/v1`、status=`published`、ready_uri、ready_sha256。事件引用的 ready 文件遵循 Spec 027 严格连续窗口合同。事件总线写入权限应限于可信生产者；基础设施不能凭任意对象写入推断成熟性。

触发默认关闭。先完成真实 run、推理镜像及首次 Endpoint，手动执行完整日更，验证重复执行、故障恢复和回滚，再启用。模块提供投递 DLQ 告警、任务退出失败告警、发布心跳告警；心跳不表示模型数据日期够新，需 doku-monitoring 对实际 cutoff 进一步监控。

## 5. 服务端交付清单

实际部署后填写 Endpoint 名称/区域、InvokeEndpoint 身份授权、wire 合同与 feature-config 对应版本、合成请求及预期模型版本、错误码和实测容量。示例使用 AWS SDK 的 InvokeEndpoint，不能把 S3 地址当作推理调用地址。本阶段不编写业务推荐服务。

Source: 2026-10-30 用户最新指示：第④⑤步继续实施，窗口改为 28 天训练 + 1 天 validation，取代此前等待 31 天的安排。

## 2026-10-30 rollout 更新

已获得云端实施授权，前文“未执行云端操作”仅记录最初交付状态，以 verification.md 的逐项执行证据为准。实际 rollout 使用独立 Terraform state `envs/prod/tf-endpoint/terraform.tfstate`，不修改共享 PROD root 的未提交环境变更。28+1 配置固定且训练配方不变。首次资源恢复记录和完整运行证据保存在本地忽略目录 `runs/spec028-rollout-20261030/`。

上游适配器代码 `ready_event.py` 打包为不可变 S3 Lambda 制品，由 IaC 绑定摘要和环境。它从 ECS 完成事件取真实任务 ARN，再查询 AWS 核验指定样本任务成功退出和 daily/project-training 命令，忽略 hourly join；按明示 cutoff 构造 29 天 ready。`_SUCCESS` 语义来自 doku-offline 的 `publish_day`：24 个小时源 marker、行数一致、Parquet 校验和行数、源 marker 未变化后 CAS 发布。不是从文件存在或机器时间推断 ready。
