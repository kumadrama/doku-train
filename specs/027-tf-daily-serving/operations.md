# Spec 027：本地入口与外部集成合同

Source: Spec 027 本地实施。所有示例路径为操作者提供的文件；本文不授权云执行。已有树模型不受本入口影响。

## 1. 日更训练与导出

平台先解析 ready、冻结 Manifest 和运行配置，再在 SageMaker 训练容器内调用：

```sh
python -m recommend_tf.train.daily_train_model \
  --config /work/frozen/config.json --cutoff 2026-10-07 \
  --frozen-input /work/frozen/input --output /work/run
```

日期仅为示例。配置沿用新版 TF 配方，新增 `online_config.training_date/validation_date`；不能把旧 `recommend` YAML 直接传入。既有 `cloud_train` 使用 `DOKU_MODE=daily` 及 `DOKU_CUTOFF`、`DOKU_FROZEN_INPUT` 完成相同装配，仍校验源码、配置和依赖摘要。

```sh
python -m recommend_tf.train.export_model \
  --run /work/run --output /work/export --outputs /work/outputs.json
```

outputs.json 为四个已配置任务到业务单位的映射。本次为 completion/fast_swipe/immersive_click → probability、watch_seconds → seconds；单位只是合同声明，导出器不擅自添加输出变换，必须核对配方 predict_fn。

新训练成功后写 `COMMITTED.json`，上传在所有 payload 与状态之后完成。导出验证文件摘要、成功状态、checkpoint/词表/config/lineage 身份，比较 EVAL checkpoint 与重新加载 SavedModel 的最终预测。

旧实验（包括首次指定 epoch 21）没有此提交记录：使用 `--evidence /work/audited-evidence.json` 提供经实际核验的文件摘要清单，格式为 `{"version":"doku-tf-run/v1","files":{"relative/path":"sha256"}}`。必须覆盖导出读取的状态、config、input、selected checkpoint、selection、metrics、feature-state、lineage，不能拿任务 Completed 状态替代完整性证据。本次没有下载或核验该真实实验的权重。

## 2. Ready adapter 合同

新增 `doku-tf-dataset-ready/v1` 是本仓接受的上游成功适配合同；不是声称现有上游已发出此格式。IaC/上游适配需将现有成功结果转换后交给一次性 ECS。

```json
{
  "version": "doku-tf-dataset-ready/v1",
  "timezone": "UTC",
  "cutoff": "2026-10-07",
  "sources": [
    {
      "date": "2026-10-01",
      "manifest_uri": "s3://EXPLICIT_BUCKET/IMMUTABLE_DAY/manifest.json",
      "manifest_sha256": "EXACT_FILE_SHA256",
      "contract_sha256": {"features": "EXACT_SHA256", "labels": "EXACT_SHA256"},
      "status": "published",
      "maturity_evidence": "UPSTREAM_VERSIONED_PUBLICATION_EVIDENCE"
    }
  ]
}
```

sources 示例省略其他日期；真实输入必须覆盖按配置计算的整个连续窗口并按日期排序。当前 reader 按 UTC 分区处理，因此拒绝其他 timezone，不以本机时区重新解释日期。每个日 Manifest 必须有既有日合同要求的 label_config_version、projection_sha256；合同允许的新增字段通过现有 merge/verify 逻辑处理。

maturity_evidence 是受信任上游成功适配提供的成熟性证明引用，本仓不重新归因标签或推定等待时长。不能从任意 S3 文件写事件直接合成 published。配置与 ready 的完整内容参与 operation digest，训练阶段仅读取冻结输入。

## 3. Serving 仓库交付合同

本仓没有 serving 实现。当前导出包合同 `doku-tf-inference/v1`：

```text
manifest.json
signature.json
feature-config.json
feature-state.json
lineage.json
saved_model/saved_model.pb
saved_model/variables/...
```

SavedModel 的 `serving_default` 接收已经组织、截断和 padding 的具名张量（键为 group__field）；图内保留现有特征变换、词表 lookup 和模型计算。签名仅包含预测真正依赖的输入，不包含训练标签。外部适配负责默认值、原始历史整理和张量组装，规则由 feature-config 声明；不提供本仓 Python 模块作为线上 import 合同。

因此不能直接把此包交给任意官方镜像并声称可用：serving 仓库必须提供理解该包目录与编码合同的 CPU TensorFlow Serving 镜像，确认签名/响应格式及 TF 兼容性。部署 binding 的 artifact_contract 必须显式为该版本。线上仍返回四任务结果及实际模型版本；候选顺序保持不变。

部署 probe 由调用方提供合成请求，配置 content_type/request/version_field/outputs_field/expected_rows/tasks，且 fixture_kind="synthetic"。probe 验证实际模型版本、行数、四任务名称和有限数值；本地导出测试承担数值 parity。不得放真实用户样本进 probe。

## 4. 平台运行绑定

`daily_pipeline` 是一次性执行命令，不创建 timer、IAM、网络或 ECS 资源：

```sh
python -m recommend_tf.platforms.sagemaker.daily_pipeline \
  --config /work/config.json --ready /work/ready.json \
  --binding /work/binding.json --executor-arn "$ACTUAL_ECS_TASK_ARN"
```

executor-arn 必须来自当前 ECS task 身份，不能跨任务复用固定字符串。绑定必须显式提供：

| 字段 | 含义 |
| --- | --- |
| environment、region、ecs_cluster | 明确环境和执行集群，无生产默认值 |
| state_uri | Endpoint 专属的 S3 条件更新状态对象；所有发布者必须共用 |
| work_prefix、model_prefix、kms_key | 任务冻结/训练归档路径、不可变模型版本路径和加密密钥 |
| poll_seconds、max_wait_seconds | 本次执行轮询间隔和总等待预算，正整数秒 |
| download_limit_bytes、export_timeout_seconds | 训练产物下载磁盘限额、独立导出进程超时 |
| outputs | 四个主任务与业务单位 |
| training_request | 显式 SageMaker CreateTrainingJob 参数模板，包括角色、OutputDataConfig、CPU 资源、StoppingCondition、固定 digest 镜像、源码/锁摘要环境变量等 |
| deployment | endpoint_name、image_uri（固定 digest）、execution_role_arn、artifact_contract、instance_count、instance_type、alarm_names、wait_seconds、termination_wait_seconds、timeout_seconds；可选 vpc_config |
| probe | 上节的合成请求与响应字段合同 |

平台覆盖 TrainingJobName、daily 输入 channel 及 DOKU 日更环境变量；不覆盖其他明确训练参数。训练镜像仍须包含已验证配方所需代码与依赖；一次性 ECS 编排镜像也需具备相同 TF 导出依赖。训练身份、部署身份与推理加载身份分别配置。

首次发布可加 `--bootstrap-run s3://EXPLICIT_RUN_PREFIX`，跳过训练；历史 run 另加 `--bootstrap-evidence /work/audited-evidence.json`。必须使用原实验真实数据截止日，不伪称已完成新 30+1 训练。

## 5. 恢复、回滚与保留

- 同操作重复执行先对账确定性 TrainingJobName；已有任务不会再次创建。相同发布者重启需等待 AWS 确认旧 ECS task 为 STOPPED，未知状态不自动接管。
- 日更阶段为 claimed/training/trained/exported/uploaded/deploying/deployed。导出和上传可复用成功 run；重复导出通过固定初始化节点种子、模型版本作用域的查找表名和确定性序列化保持包摘要一致。
- SageMaker 更新为 ALL_AT_ONCE，使用显式观察时间及告警；旧实例由平台处理。仅 API 返回不声明上线，还需 Endpoint 配置对账和实际 probe。
- probe 失败且存在旧配置时启动恢复；首次创建无旧 TF 配置时记录 deployment_failed，不能声称回滚到树模型。
- UpdateEndpoint 请求已尝试但其结果不明确、且查询不到待更新配置时，保留 active 状态并停止盲目重发。需要操作者核查 AWS 实际操作记录后处置，不能仅因超时解除锁。
- 终态训练失败不会被同 operation 的重跑伪装成成功。确认要重训可传入新的 `--retry-id`，不改变训练配方；默认不自动重训失败任务。
- 显式回滚：同命令增加 `--rollback-version VERSION`，目标必须是本状态历史中的成功版本。只降低在线版本，不降低自动发布数据高水位。
- 当前不自动删除历史模型包、镜像或 EndpointConfig，全部部署历史保留引用，防止误清理当前/回滚版本。外部清理器不得绕过这些引用；保留期限确认后再接入清理策略，未实现自动回收。
- IaC 不应在日常 apply 时把 EndpointConfig 改回旧版本；Endpoint 更新权威属于发布器。旧树模型 registry/Redis 清理不适用于这里。

## 6. 当前验收边界

本地合成训练、SavedModel parity 和 AWS 接口模拟测试不代表生产模型或外部镜像已验收。外部 ready adapter、serving 镜像、IaC/监控资源和消费方容量目标均需要后续集成。本仓没有启动真实训练、上传真实制品、创建 Endpoint 或修改 PROD。

### 复审后的恢复边界

上游缺日或不满足 ready 合同在认领前失败；认领后、训练 API 调用前的数据校验错误记录 `input_failed`，释放占用后使用显式 `--retry-id` 重试。已经冻结请求的训练和已完成操作不重新读取上游。

恢复同一操作时可增大 `poll_seconds`、`max_wait_seconds`、`download_limit_bytes`、`export_timeout_seconds`，这些执行预算不改变 operation 标识。其他配置变化仍视作不同操作，不能绕过活动认领。

制品目录包含逻辑模型版本及 manifest SHA256 两级；导出器源码摘要参与逻辑版本。Endpoint Model 资源名还绑定具体制品 URI。不要手工覆盖这些不可变对象。

Endpoint 被其他发布器修改时本流程拒绝覆盖，保留状态供对账；`UpdateRollbackFailed` 不代表恢复成功，仍持有认领并尝试恢复捕获的旧配置。不要通过删状态对象强行解锁。
