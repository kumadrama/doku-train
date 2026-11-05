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

## 官方 DLC 直通模式

新增 `official.py` 将既有 canonical SavedModel 重打包供 AWS 官方 TensorFlow Serving
容器使用；训练、导出算法和历史长度保持不变。原 Python app/raw 协议继续可用。
直通模式由 ai-engine Go 层实现原 request_adapter 的整理/padding，SageMaker 接收
`{"signature_name":"serving_default","inputs":...}`，响应为 `{"outputs":...}`。

```sh
python -m recommend_tf.inference.official \
  --source /path/to/canonical-package \
  --output /path/to/new-model.tar.gz \
  --probe /path/to/synthetic-raw-request.json
```

命令仅操作本地文件，输出部署摘要及转换后的合成 probe；不上传、不部署、不训练。
输出归档采用 `<doku_tf_完整模型摘要>/1/saved_model.pb` 和 `variables/` 布局；
合同文件保存在 `doku-contract/`，其中 manifest 原封不动描述 canonical 产物。
没有 `code/inference.py`，请求不经过 Python handler。

日更 **deployment** binding 使用 `inference_protocol: tensorflow-serving/v1`，
显式固定官方镜像 digest，container_environment 使用 `SAGEMAKER_TFS_*` 设置。
发布器自动设 `SAGEMAKER_TFS_DEFAULT_MODEL_NAME`，拒绝旧 DOKU_* 和 Python hook 设置；
模型包走独立 `/tensorflow-serving-v1/model.tar.gz`，同目录有 deployment.json。
probe.request 必须使用上述工具产生的标准 tensor 请求；不再需要 version_field /
outputs_field。每次探测携带具体 TFS 模型名和数字版本 1，验证任务集合、行数及有限数值。
省略协议字段时保留原行为。切换时为新模型创建独立 Endpoint，保留旧 Endpoint 供回滚。

接入 ai-engine [PR #11](https://github.com/kumadrama/ai-engine/pull/11) 时，使用本入口提供
其部署文档引用的模型包。发布顺序是：更新本仓源码 → 重建并固定 pipeline 镜像 digest →
生成并校验官方模型包/合成 probe → 显式更新部署 binding → 执行获授权的发布。
仅合入源码不会更新现有镜像、Endpoint 或日更配置。pipeline 构建上下文包含
`inference/official.py` 及其加载/请求适配依赖，避免运行时缺模块。

无需 TensorFlow 的发布回归：`make inference-check`。测试使用合成文件和内存中的云 API
替身，不启动训练或访问 AWS；覆盖打包、发布编排、旧协议与构建上下文的独立导入。
验证边界见 [Spec 028 verification](../../specs/028-tf-endpoint-rollout/verification.md)。
