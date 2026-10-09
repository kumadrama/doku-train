# 组件划分

## 文档状态

Implemented Locally。Spec 001 的 DNN 路径和 Spec 003 的 XGBoost 基线路径已经落地；生产容量
benchmark、SageMaker 和线上 Serving 仍未实现。

Source: [系统概览](system-overview.md)；用户于 2026-09-13 确认单仓库包含训练框架与模型，目录名
和内部命名尽量与 Finder 对齐。

## 目标目录

```text
doku-train/
├── AGENTS.md
├── docs/
│   ├── source/
│   └── architecture/
├── specs/
├── configs/                              # 可审查的非敏感训练配置
├── recommend/
│   └── train/
│       ├── comm/
│       │   ├── config.schema.yml
│       │   ├── model_params.py
│       │   ├── model_params_validator.py
│       │   ├── model_protocol.py
│       │   ├── training_pipeline.py
│       │   ├── checkpoint_agent.py
│       │   ├── feature_check.py
│       │   ├── errors.py
│       │   ├── resource_report.py
│       │   ├── metrics_utils.py
│       │   ├── training_output.py
│       │   ├── model_exporter.py
│       │   ├── offline_evaluate.py
│       │   ├── datasvr/
│       │   │   ├── dataset_manifest.py
│       │   │   ├── storage.py
│       │   │   ├── local_storage.py
│       │   │   ├── parquet_dataset.py
│       │   │   └── s3_storage.py
│       │   └── eval/
│       │       ├── eval_model_rerank.py
│       │       └── gates.py
│       ├── models/
│       │   ├── rerank/
│       │   │   ├── __init__.py
│       │   │   ├── config.schema.yml
│       │   │   ├── model_params.py
│       │   │   ├── data_utils.py
│       │   │   ├── feature_utils.py
│       │   │   ├── rerank_model.py
│       │   │   └── modules/
│       │   └── rerank_train/
│       │       ├── __init__.py
│       │       ├── config.schema.yml
│       │       ├── model_params.py
│       │       ├── data_utils.py
│       │       └── rerank_train_model.py
│       ├── offline/
│       ├── tools/
│       │   └── checkpoint/
│       ├── tests/
│       │   ├── unit/
│       │   ├── contract/
│       │   ├── smoke/
│       │   └── performance/
│       ├── run.py
│       ├── train_rerank.py
│       └── train_rerank_xgboost.py
├── pyproject.toml
├── uv.lock
└── Makefile
```

图中目录是当前已实现的主边界；省略了缓存、测试数据和内部辅助文件。

## 组件职责

### `comm`

承载 Finder `comm` 对应的公共训练能力：类型化合同、配置、训练生命周期、指标和训练产物写出。
`TrainingPipeline` 管理随机种子、优化器、early stopping、Checkpoint、27/1 评估和 28 天 refit；
首阶段执行策略只有 `single_process_cpu`。`comm` 不 import `models/rerank`。

### `comm/datasvr`

保留 Finder 熟悉的目录名，但职责是读取 S3 Dataset Manifest 和 Parquet 分片，不是网络服务。
`DatasetManifest` 固定输入 lineage，`ParquetDataset` 提供有界 batch，`S3Storage` 只负责对象读取、
校验与 create-only 写入。该目录不决定模型结构和业务门禁。

### `comm/eval`

`eval_model_rerank.py` 在冻结的 validation 切片上按四个目标分别计算 loss、AUC 和样本统计；`gates.py`
将数据、数值、Checkpoint 兼容性和可评估性建模为 hard gate，将模型质量变化建模为 soft gate。

### `models/rerank`

保存首个精排模型插件，文件命名对应 Finder 模型目录。`RerankModel` 是 PyTorch shared-bottom DNN，
输出 `effective_watch`、`completion`、`non_fast_swipe`、`immersive_click` 四个 logits；
`data_utils.py` 与 `feature_utils.py` 验证并编码模型专属张量和上游 7 天特征，不从原始事件聚合；
模型不访问 S3 或 Checkpoint。

### `models/rerank_train`

保存四目标 XGBoost CPU 基线。插件通过可重置 `DataIter` 和 `QuantileDMatrix` 有界消费已转换
batch，为每个目标训练一个 `binary:logistic` Booster，以 raw margin 适配公共评估器，并负责
确定性 `checkpoint.xgb.zip` 编解码。插件固定 `hist`/CPU，不访问 S3、Manifest 或输出位置。

### `offline`

保存本仓的离线评估、回放和 backtest 入口。该名称对应 Finder 目录，与尚未启用的
`doku-offline` 仓库没有运行时依赖关系。

### `comm/checkpoint_agent.py`、`comm/training_output.py` 与 `comm/model_exporter.py`

`CheckpointAgent` 按 Finder 原位置保存和恢复 PyTorch 训练状态；`tools/checkpoint` 保留批量检查、
迁移等离线工具；`TrainingOutputWriter` 以 run-scoped、create-only 语义写出 production
Checkpoint、指标、Feature State 和 lineage。DNN 继续使用 `TrainingOutputWriter`；XGBoost 通过
`ModelTrainingOutputWriter` 写入不同名称的模型文件，同时保持三个共享 JSON 合同不变。
`ModelExporter` 只保留未来导出的 Protocol，不提供 ONNX 实现，也不把 Checkpoint 声明为线上制品。

### `run.py`、`train_rerank.py` 与 `train_rerank_xgboost.py`

`run.py` 是统一 composition root；`train_rerank.py` 编排 DNN，`train_rerank_xgboost.py` 编排树模型
的 validate、train、evaluate、refit 与 write-training-output 用例。入口只解析参数和装配对象，
不承载特征算法、网络层或 Booster 算法。

## 依赖方向

```text
run.py / train_rerank.py
          │
          ▼
comm/training_pipeline.py ───────→ DNN model protocol
     │          │                       ▲
     │          ├→ comm/eval            │
     │          ├→ comm/checkpoint_agent.py
     │          └→ comm/training_output.py
     ▼                                  │
comm/datasvr                         models/rerank

train_rerank_xgboost.py ─────────→ models/rerank_train
          │                                 │
          └→ comm/datasvr + comm/eval ←────┘
```

- 具体模型只在 composition root 中按注册名装配；
- 不允许 `training_pipeline.py` 出现 `if model_name == ...`；
- 不允许 `models` 反向调用 CLI、Training Output Writer 或存储适配器；
- 外部存储 SDK 必须位于 adapter 边界，领域合同不暴露 SDK 类型。

## 与 Finder 的对应关系

| Finder 概念 | doku-train 目标 | 处理方式 |
| --- | --- | --- |
| `recommend/train/comm` | `recommend/train/comm` | 保留目录，内部按模块维持单一职责 |
| `recommend/train/models` | `recommend/train/models` | 保留同仓模型插件边界 |
| `models/*/model_params.py` | `models/rerank/model_params.py` | 同名文件，Doku 使用类型化 PyTorch 参数 |
| `models/*/data_utils.py` | `models/rerank/data_utils.py` | 同名文件，消费 Manifest batch |
| `models/*/feature_utils.py` | `models/rerank/feature_utils.py` | 同名文件，执行合同化 7 天特征编码 |
| `models/*/*_model.py` | `models/rerank/rerank_model.py` | 同名风格，Doku 实现多目标 PyTorch DNN |
| Tree baseline plugin | `models/rerank_train/rerank_train_model.py` | 同名风格，Doku 实现四目标 CPU XGBoost |
| `run.py` / `train_rerank.py` | 同名入口 | 不拼 shell 命令 |
| `comm/datasvr` | 同名目录 | 无 server；仅 S3/Manifest/Parquet adapter |
| `comm/checkpoint_agent.py` | 同名同位置 | PyTorch 训练恢复，不作为发布格式 |
| `tools/checkpoint` | 同名工具位置 | 保留训练 Checkpoint 离线检查边界 |
| Finder 模型导出工具 | `comm/model_exporter.py` | 仅保留接口；具体格式与实现进入后续 Spec |
| distributed session/embedding | 后续独立 Spec | 首阶段不实现 |
