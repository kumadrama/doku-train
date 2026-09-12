# 接口合同

## 文档状态

目标接口，尚未实现。接口名称描述职责，具体 Python 签名与 JSON Schema 在 Spec 001 design 中冻结。

Source: [系统概览](system-overview.md)、[组件划分](components.md)、[数据模型](data-model.md)。

## 上游输入接口

调用方必须提供：

1. 精确 Dataset Manifest URI；
2. Train Config URI 或本地路径；
3. 显式输出 staging 位置；
4. 非本地环境下的显式环境名与已配置身份。

系统读取 Manifest 后依次验证合同版本、Feature/Label 版本、分片时间范围、行数和校验和。
任一关键验证失败时停止训练，不回退到上一份数据，也不扫描 `latest`。

当前 producer 是已有数据流，本仓直接读取 S3。`doku-offline` 尚未启用；未来若接入，也只能
生产同一 Dataset Manifest，不能让本仓共享其内部 Python 类。序列化 schema 必须版本化，并由
producer/consumer contract test 共同验证。

## CLI 接口

Finder-compatible 目标入口：

```text
python -m recommend.train.run validate --manifest <s3-uri> --config <path>
python -m recommend.train.train_rerank train --manifest <s3-uri> --config <path> --output <s3-uri>
python -m recommend.train.train_rerank backtest --manifest <s3-uri> --config <path> --output <uri>
```

- `validate` 不启动训练，只检查数据/配置/兼容性；
- `train` 是完整每日用例：27 天 train、1 天 validation、hard/soft gate、28 天 refit、ONNX export；
- `backtest` 显式执行 24/2/2 或 rolling folds，不与每日训练隐式混用；
- `run.py` 是统一 composition root，`train_rerank.py` 保留 Finder 熟悉的精排入口；可选 console
  alias 只能转发到相同入口，不能实现第二套逻辑。

所有命令以非零退出码表示失败，并输出结构化、脱敏的错误摘要。CLI 不从当前目录、环境名或
历史运行猜测输入。

## Dataset Reader 接口

Dataset Reader 消费已验证 Manifest，并提供：

- schema inspection；
- 确定性的时间切分和分片顺序；
- 有界内存 batch iteration；
- 行数、正负例数和坏样本计数；
- 可中止的读取与明确错误分类。

Reader 不做网络结构相关变换，不随机丢弃样本；Spec 001 保留全部曝光，不实现负采样。

## Feature Transformer 接口

Feature Transformer 分为两个阶段：

- `fit`：评估阶段只在 27 天 training split 上拟合统计量和类别状态；生产 refit 在全部 28 天上
  重新拟合；
- `transform`：以对应阶段的冻结状态转换 training/validation、production refit 和未来在线输入。

Validation 数据不得参与 `evaluation_model` 的 `fit`。Transformer 输出稳定命名的 tensor batch
与输入签名，并可把 `FittedFeatureState` 序列化到 Artifact。统计特征必须是样本曝光前 7 天的
point-in-time 值；原始 `user_id` 不得出现在首版输入签名。

## Model 插件接口

每个模型插件提供：

- 稳定 `model_name=rerank` 和 `config.schema.yml`；
- 从验证后配置构建 CPU 模型的方法；
- 输入签名与 `effective_watch`、`completion`、`non_fast_swipe`、`immersive_click` 四个 logits；
- 按目标有效性 mask 和可配置 loss weight 计算 multi-task loss 的逻辑；
- 可导出状态及兼容性声明。

Runtime 负责训练生命周期，模型插件不得：

- 读取数据文件或 Manifest；
- 访问存储、网络或环境凭据；
- 自行启动 worker/process；
- 决定 Artifact 路径或线上发布；
- 根据 DTE/PROD 改变网络结构。

## Execution Strategy 接口

首阶段唯一实现是 `single_process_cpu`：一个 PyTorch 训练进程、一个完整模型副本、受控 CPU
线程和可配置数据读取 worker。

[INFERRED] Future strategy 可以沿用相同 batch/model/checkpoint 接口加入多进程或多机同步训练，
但必须通过独立 Spec 证明单机训练窗口不足。未来能力不得影响首阶段配置语义或制品格式。

## Checkpoint 接口

Checkpoint 至少保存：

- 模型与优化器状态；
- epoch/step；
- 随机数状态；
- config 与输入 lineage 摘要；
- early stopping 状态；
- Checkpoint 格式版本和校验和。

恢复时 lineage 或结构配置不兼容必须拒绝；不能“尽量加载”后继续产生看似成功的模型。

## Model Artifact 接口

逻辑制品布局：

```text
<model-name>/<model-version>/
├── manifest.json
├── model.onnx
├── signature.json
├── model-config.json
├── feature-schema.json
├── fitted-feature-state/
├── metrics.json
└── lineage.json
```

物理 bucket/prefix 由调用方与部署系统提供，不写死在训练代码。消费者先读取 `manifest.json`，
校验全部文件摘要、ONNX opset 和 signature 后才能加载模型。发布前必须在固定 fixture 上验证
PyTorch 与 ONNX Runtime 的四个输出在配置容差内一致。

线上 Go 或 C++ serving 只依赖 `model.onnx`、`signature.json` 与 Feature Schema，不依赖 PyTorch；
serving 的实现语言和进程边界不属于本仓。Go 首版可通过 ONNX Runtime C API 的窄适配层加载，
后续若出现 cgo 稳定性、性能或故障隔离证据，再把 Predictor 边界拆为独立 C++ 服务。

## 错误合同

错误至少分为：

- input contract invalid；
- unsupported schema/version；
- data integrity mismatch；
- label not mature / split invalid；
- resource budget exceeded；
- training numerical failure；
- checkpoint incompatible；
- evaluation hard gate failed / quality soft gate warning；
- artifact write/verification failed。

失败必须保留足够的结构化诊断和 `run_id`，但不得记录原始样本、凭据或完整敏感 URI 查询参数。
