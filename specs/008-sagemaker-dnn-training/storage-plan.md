# Spec 008：存储授权与磁盘规划

状态：2026-10-20 用户确认存储方案与预算；未实施，未修改云资源。
Source: 本次 S3/KMS 元数据查询、冻结 Manifest 17 个分片的 HeadObject、本地文件逻辑大小统计。

## 现有资源事实

- 桶：`prod-data-lake-20260901071345278900000001`，区域 us-east-2。
- 默认 SSE-KMS，Bucket Key 开启；key：
  `arn:aws:kms:us-east-2:703844615824:key/bbf7f36f-8cd3-48f8-9539-839e8cbe7f38`。
  客户管理对称 key，Enabled；当前 key policy 允许账号通过 IAM 授权使用。
- 桶策略要求 HTTPS，所有 PutObject 必须显式携带 aws:kms 和上述完整 key ARN；
  不能只依赖默认桶加密。该限制也适用于配置包、TensorBoard 事件和平台归档。
- Versioning 为 Suspended；不能依赖版本控制恢复误覆盖，必须使用唯一 Run 路径与 create-only 写入。
- `ml/fyp-rank-train/` 下对象 180 天过期。全桶另有 7 天中止未完成 multipart 规则，
  该规则不会删除已完成对象。其余当前过期规则不覆盖建议实验前缀。
- `ml/sagemaker-experiments/` 查询时没有对象；此查询不创建目录。
- 冻结 Manifest 的 17 个 Parquet 分片 HeadObject 全成功，总计 1,901,425,563 bytes（1.77 GiB），
  均使用上述 SSE-KMS key。没有下载或打印真实样本。

## 建议授权范围

[INFERRED] 复用桶与现有 key，不新建桶/key，不放宽既有桶策略。物理资源标识仅作为核查证据，
实现时由 doku-iac 输出 binding，业务运行配置引用逻辑名。

| 身份 | 范围与操作 |
| --- | --- |
| 训练执行角色 | 只读 `ml/fyp-rank-train/daily/environment=prod/*`，包含被固定引用的 Manifest、合同与 shards；允许 GetObject，必要 ListBucket 限制为该前缀 |
| 训练执行角色 | 读取专用提交包 `ml/sagemaker-experiments/environment=prod/submissions/*` |
| 训练执行角色 | 写入/读取 `ml/sagemaker-experiments/environment=prod/runs/*`，包含归档、诊断与 TensorBoard；不授予 DeleteObject 或线上发布前缀写权限 |
| 本地提交者 | 读取指定输入，写 submissions；查看任务、传递指定训练角色；结果读取 runs |
| 本地 TensorBoard 查看者 | 仅 List/Get 实验 runs 下指标对象及必要 KMS 解密权限，不需要提交或发布权限 |

建议路径：

```text
ml/sagemaker-experiments/environment=prod/
  submissions/<run_id>/       # 配置、冻结 Manifest/合同与身份
  runs/<run_id>/artifacts/    # SageMaker 最终归档（平台可能附加 job 子路径）
  runs/<run_id>/diagnostics/
  runs/<run_id>/tensorboard/  # 周期上传的事件文件/不可变片段
```

运行期无需发现 latest。提交端校验所有被引用对象均在批准的输入范围内；
若使用旧的 daily-migration 或其他数据前缀，应先显式增加授权范围，不能扩大到全桶。
仅冻结合同，不复制原始 shards；输入 180 天过期后，不保证能重跑历史实验，
虽然实验模型和指标仍保留。长期数据快照保留不属于当前方案。

S3 权限按 bucket/object ARN 分开，ListBucket 配 s3:prefix；允许必要 GetBucketLocation。
上传使用 PutObject，必要时增加受限 AbortMultipartUpload/ListMultipartUploadParts，
不默认授予 DeleteObject、全桶写入或 bucket 配置权限。
冻结提交包和最终产物 create-only；增长中的事件不得覆盖最终产物，可采用唯一事件片段。
后续实现需明确服务自动归档路径与 job/run 的唯一映射，并测试重提不会覆盖。

KMS 最小数据访问操作：输入读取用 Decrypt；S3 上传用 GenerateDataKey，multipart 必要时也用 Decrypt；
校验需要时用 DescribeKey。限定 Resource 为上述 key，数据操作限定 ViaService 为
`s3.us-east-2.amazonaws.com`。因 Bucket Key 已开启，encryption context 允许桶 ARN；
若兼容未使用 Bucket Key 的历史对象，可附加批准对象 ARN 模式。前缀隔离由 S3 IAM 保证，
不能误把所有 KMS context 都限制为对象 ARN，否则 Bucket Key 路径会失败。
该 S3 key 不直接复用作训练 EBS VolumeKmsKeyId；磁盘加密保持平台默认加密方案，
若另需客户 key，单独审阅相应 grant/EC2 权限。

SageMaker OutputDataConfig.KmsKeyId 显式传入完整 key ARN；本地上传与周期指标上传同样显式指定 SSE-KMS。
不假设 TensorBoardOutputConfig 自动同步能满足当前强制加密请求头策略；首期设计采用可控的
事件上传器并做合同测试，真实端到端写入须另行授权后验证。

## 磁盘容量建议

[INFERRED] 初始 `VolumeSizeInGB: 100`，每任务独立配置，不由 15 并发放大单任务磁盘。
本次按逻辑字节统计，避免把 macOS 稀疏/压缩文件的 du 占用误当所需 Linux 卷容量：

| 项目 | 单任务基线 |
| --- | ---: |
| 17 天 Parquet 缓存 | 1.77 GiB |
| 训练编码 | 7.20 GiB |
| validation 编码 | 0.61 GiB |
| 两者编码合计 | 7.81 GiB |
| DNN/DeepFM Run 非编码文件（包含 resume） | 约 2.6 MiB |

上述训练约 164 万行；test 临时文件峰值没有独立磁盘采样，不能把最终目录大小当作峰值。
当前配置 cache_limit_bytes 为 20 GiB，这是缓存限制，不是整卷容量限制。
100 GiB 初始容量给输入缓存、编码、test 临时数据、事件和归档留出余量；镜像/系统占用
按实际平台挂载情况在 smoke 中记录。实现时报告目录/卷剩余空间，磁盘不足明确失败，不静默改变输入。

若按当前编码密度线性推算到 2800 万训练样本，仅训练编码约 123 GiB，100 GiB 不足；
并且这是估算而非容量实测。大窗口须重新规划（可评估 200–250 GiB）、流式编码改进或清理临时数据，
不能承诺当前磁盘和 32 GiB 内存覆盖该规模。数据天数仍由 YAML 声明，不新增固定天数限制。

## 跨仓落地与验收

- doku-iac：声明训练角色、上述前缀/KMS 权限与 binding；不改变现有数据过期策略。
  日志 30 天策略应核查 SageMaker 共享日志组既有使用者，不能未经评估改变他人保留期限。
- doku-train：100 GiB 运行配置、显式加密上传、输入范围校验、唯一输出、指标片段同步和磁盘证据。
- 只读结果证明资源配置，不证明新角色能实际读写；实现后用合成数据与授权 smoke 验证
  读取、multipart/归档、TensorBoard 同步、权限拒绝和卷峰值。

参考：[S3 Bucket Key context](https://docs.aws.amazon.com/AmazonS3/latest/userguide/configuring-bucket-key.html)、
[SageMaker 输出 key](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_OutputDataConfig.html)。

## 已确认预算

Source: 2026-10-20 用户接受预算并要求同步需求、设计和实施任务。
按每天 15 个一小时任务，计算与训练盘约 $216/月，预留总预算 $250–300/月；
详见 [需求预算](requirements.md#6-已确认运行参数与预算)。这不是云端费用自动限额。
