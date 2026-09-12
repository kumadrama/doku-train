# 组件划分

## 文档状态

目标设计，尚未创建代码目录。

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
│       │   ├── training_pipeline.py
│       │   ├── checkpoint_agent.py
│       │   ├── feature_check.py
│       │   ├── metrics_utils.py
│       │   ├── artifact_utils.py
│       │   ├── offline_evaluate.py
│       │   ├── datasvr/
│       │   │   ├── dataset_manifest.py
│       │   │   ├── parquet_dataset.py
│       │   │   └── s3_storage.py
│       │   └── eval/
│       │       ├── eval_model_rerank.py
│       │       └── gates.py
│       ├── models/
│       │   └── rerank/
│       │       ├── __init__.py
│       │       ├── config.schema.yml
│       │       ├── model_params.py
│       │       ├── data_utils.py
│       │       ├── feature_utils.py
│       │       ├── rerank_model.py
│       │       └── modules/
│       ├── offline/
│       ├── tools/
│       │   ├── checkpoint/
│       │   └── onnx/
│       │       └── gen_onnx.py
│       ├── tests/
│       │   ├── unit/
│       │   ├── contract/
│       │   └── smoke/
│       ├── run.py
│       └── train_rerank.py
├── pyproject.toml
├── uv.lock
└── Makefile
```

目录在对应 Spec 的 design 批准后逐步创建；本图不表示当前已实现。

## 组件职责

### `comm`

承载 Finder `comm` 对应的公共训练能力：类型化合同、配置、训练生命周期、指标和制品组装。
`TrainingPipeline` 管理随机种子、优化器、early stopping、Checkpoint、27/1 评估和 28 天 refit；
首阶段执行策略只有 `single_process_cpu`。`comm` 不 import `models/rerank`。

### `comm/datasvr`

保留 Finder 熟悉的目录名，但职责是读取 S3 Dataset Manifest 和 Parquet 分片，不是网络服务。
`DatasetManifest` 固定输入 lineage，`ParquetDataset` 提供有界 batch，`S3Storage` 只负责对象读取、
校验与 staging 写入。该目录不决定模型结构和业务门禁。

### `comm/eval`

`eval_model_rerank.py` 在冻结的 validation 切片上按四个目标分别计算 loss、AUC 和样本统计；`gates.py`
将数据/数值/导出校验建模为 hard gate，将模型质量变化建模为 soft gate。

### `models/rerank`

保存首个精排模型插件，文件命名对应 Finder 模型目录。`RerankModel` 是 PyTorch shared-bottom DNN，
输出 `effective_watch`、`completion`、`non_fast_swipe`、`immersive_click` 四个 logits；
`data_utils.py` 与 `feature_utils.py` 验证并编码模型专属张量和上游 7 天特征，不从原始事件聚合；
模型不访问 S3 或 Checkpoint。

### `offline`

保存本仓的离线评估、回放和 backtest 入口。该名称对应 Finder 目录，与尚未启用的
`doku-offline` 仓库没有运行时依赖关系。

### `comm/checkpoint_agent.py`、`tools/checkpoint` 与 `tools/onnx`

`CheckpointAgent` 按 Finder 原位置保存和恢复 PyTorch 训练状态；`tools/checkpoint` 保留批量检查、
迁移等离线工具；`gen_onnx.py` 生成 `model.onnx`，校验输入输出
签名，并对 PyTorch 与 ONNX Runtime 的同一 fixture 输出做容差内一致性测试。Checkpoint 不能作为
线上制品。

### `run.py` 与 `train_rerank.py`

`run.py` 是统一 composition root；`train_rerank.py` 编排精排的 validate、train、evaluate、refit
与 export 用例。二者只解析参数和装配对象，不承载特征计算或网络层定义。

## 依赖方向

```text
run.py / train_rerank.py
          │
          ▼
comm/training_pipeline.py ───────→ model protocol
     │          │                       ▲
     │          ├→ comm/eval            │
     │          ├→ comm/checkpoint_agent.py
     │          └→ tools/onnx/gen_onnx.py
     ▼                                  │
comm/datasvr                         models/rerank
```

- 具体模型只在 composition root 中按注册名装配；
- 不允许 `training_pipeline.py` 出现 `if model_name == ...`；
- 不允许 `models` 反向调用 CLI、Artifact Writer 或存储适配器；
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
| `run.py` / `train_rerank.py` | 同名入口 | 不拼 shell 命令 |
| `comm/datasvr` | 同名目录 | 无 server；仅 S3/Manifest/Parquet adapter |
| `comm/checkpoint_agent.py` | 同名同位置 | PyTorch 训练恢复，不作为发布格式 |
| `tools/checkpoint` / `tools/onnx/gen_onnx.py` | 同名工具位置 | 离线检查工具 + ONNX 发布 |
| distributed session/embedding | 后续独立 Spec | 首阶段不实现 |
