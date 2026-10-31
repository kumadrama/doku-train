# Minimal TF inference

Spec 028 用户确认先将最小推理实现放在 `recommend_tf/inference/`，不放入 train，也不导入训练内部实现。模型计算复用 TensorFlow Serving；不提供业务特征获取、召回或排序。

## 合同

- 制品：`doku-tf-inference/v1`，加载时校验完整清单、签名和实际版本。
- 请求：`{"schema_version":"doku-tf-request/v1","instances":[{...具名原始特征...}]}`。
- 原始字段、类型、默认及历史规则来自已核验包的 `feature-config.json`；不接收训练标签，不进行训练样本筛选。
- 响应：`{"model_version":"实际加载版本","predictions":[{"completion":0.1,"fast_swipe":0.2,"immersive_click":0.3,"watch_seconds":4.0}]}`，候选顺序不变。单位以真实模型验证结果为准。
- `/ping`：模型版本 1 在 TensorFlow Serving AVAILABLE 后返回 200；未就绪/故障 503。内部版本 1 是容器目录编号，业务模型版本始终来自 manifest。
- `/invocations`：POST JSON；无效输入 400，过大请求 413，不支持的 Content-Type 415，忙/推理故障 503。不会把异常详情或原始样本写入请求日志。

当前 wire 协议接收原始模型特征，默认/历史整理在薄适配层完成；词表 lookup、入图变换和预测在 SavedModel 内完成。此协议不能直接替换成裸 TensorFlow Serving 张量请求。

## 构建与本地验证

从仓库根运行；构建上下文必须独立生成，根 `.dockerignore` 为其他构建用途，不应直接 `docker build .`。

```sh
python -m recommend_tf.platforms.sagemaker.build_context --kind inference --output /tmp/tf-context
# 已验证的 CPU 2.11.1 官方镜像摘要；升级需重新验证。
docker build --platform linux/amd64 \
  --build-arg TF_SERVING_IMAGE=tensorflow/serving@sha256:ceb17c529b59a3f0ffa3ef87404e0f2cf0e155cdec4a723ed49b6be487e4c728 \
  -t tf-inference:test /tmp/tf-context
```

发布到 ECR 时必须使用 Docker distribution manifest v2（`application/vnd.docker.distribution.manifest.v2+json`）；ECS 能启动 OCI 镜像不代表 SageMaker 接受。使用 buildx image exporter 的 `oci-mediatypes=false`、关闭 provenance，并验证 ECR 返回的 manifest media type 后固定 digest。

运行必须显式传入以下环境变量（正整数，不设置生产容量默认值）：

| 变量 | 含义 |
| --- | --- |
| DOKU_MAX_BATCH | 每次请求候选数上限 |
| DOKU_MAX_HISTORY | 原始历史条数上限 |
| DOKU_MAX_REQUEST_BYTES | 请求体上限 |
| DOKU_CONCURRENCY | 同时执行的预测上限 |
| DOKU_INFERENCE_TIMEOUT_SECONDS | 后端请求超时，最多 50 秒 |
| DOKU_TF_THREADS | TF intra-op 线程数；inter-op 固定 1 |
| DOKU_STARTUP_TIMEOUT_SECONDS | 加载就绪等待上限 |

模型默认挂载 `/opt/ml/model`；本地可以用 DOKU_MODEL_ROOT 显式覆盖。发布器将 binding.deployment.container_environment 传入推理容器，并强制 DOKU_MODEL_VERSION 为本次制品版本。

实际容器回归见 CI `tf-inference-container`，使用 `recommend_tf.inference.tests.container_smoke` 生成合成模型并验证完整请求结果。测试代码可以引用训练 fixture；生产 inference 模块禁止引用 train。
