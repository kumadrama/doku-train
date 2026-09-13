# doku-train

`doku-train` 是 Doku 推荐精排模型的 CPU-first 离线训练仓库。它直接读取 S3 上不可变的
Dataset Manifest 和 Parquet 分片，用 PyTorch 完成四目标 DNN 的训练、验证及 28 日 production
refit，并写出可审计的训练结果。

主要面向需要新增特征、调整模型或发起真实训练的算法开发者。第一次使用建议从下文的
`validate → train` 路径开始。

## 当前能力

- 单机、单进程 CPU 训练，模型为 shared-bottom DNN；
- 目标为 `effective_watch`、`completion`、`non_fast_swipe`、`immersive_click`；
- 前 27 个成熟自然日训练、最后 1 日验证，硬门禁通过后使用全部 28 日重新训练；
- numeric/categorical 特征、冻结词表、缺失值/OOV、point-in-time 7 日统计特征检查；
- 精确 S3 URI、SHA-256、行数、标签成熟度与连续日期校验；
- PyTorch Checkpoint、指标、Fitted Feature State 与 lineage 四类输出。

首版不负责样本生产、S3/IAM/调度资源创建、自动发布、线上 serving、ONNX、GPU 或多机训练。
`ModelExporter` 目前只有接口，不参与训练门禁；AUC 及其变化只产生 warning，不保证新模型 AUC
一定高于旧模型。

## 首次真实训练

### 1. 准备环境

需要 Python 3.12、[`uv`](https://docs.astral.sh/uv/) 以及对训练 bucket 的读取权限和对输出 prefix
的写入权限。AWS 身份使用运行环境的标准凭据链，例如实例/任务角色或本地已配置的 profile；
不要把 access key 写入仓库或训练配置。

```bash
uv sync --group dev
```

### 2. 先验证输入

假设上游已经生成：

```text
s3://<training-bucket>/rerank/datasets/2026-09-12/
├── manifest.json
├── contracts/features.json
├── contracts/labels.json
└── shards/day=YYYY-MM-DD/part-*.parquet
```

把下一节的训练配置保存为 `configs/rerank.yml`，然后执行：

```bash
uv run python -m recommend.train.run rerank validate \
  --manifest 's3://<training-bucket>/rerank/datasets/2026-09-12/manifest.json' \
  --config configs/rerank.yml \
  --run-id rerank-20260913-001
```

成功会输出 `status=VALID`、Manifest digest 和总行数。验证失败时不要启动训练，应先按结构化错误
中的 `category` 和 `stage` 修复输入。

### 3. 启动训练

```bash
uv run python -m recommend.train.run rerank train \
  --manifest 's3://<training-bucket>/rerank/datasets/2026-09-12/manifest.json' \
  --config configs/rerank.yml \
  --output 's3://<model-bucket>/rerank/training-runs' \
  --run-id rerank-20260913-001
```

`run-id` 必须唯一。输出采用 create-only 语义；重复使用已有 `run-id` 会返回
`TRAINING_OUTPUT_CONFLICT`，不会覆盖之前的结果。

周度或模型结构变更时，可显式检查 24/2/2 backtest 窗口：

```bash
uv run python -m recommend.train.run rerank backtest \
  --manifest 's3://<training-bucket>/rerank/datasets/2026-09-12/manifest.json' \
  --config configs/rerank.yml \
  --run-id rerank-backtest-20260913
```

当前 `backtest` 只验证并生成时间窗口，不写 production 训练结果。

## 输入合同

Manifest 是唯一的数据入口。训练不会扫描 bucket，也不会解析 `latest`。以下是结构示例；所有
URI、行数、时间和 SHA-256 必须由上游 producer 按真实对象生成，不能直接复制占位值：

```json
{
  "manifest_version": "1",
  "dataset_id": "rerank-exposures",
  "dataset_version": "2026-09-12",
  "feature_schema_version": "features-v1",
  "feature_schema": {
    "uri": "s3://<training-bucket>/rerank/datasets/2026-09-12/contracts/features.json",
    "size_bytes": 2048,
    "sha256": "<64-char-lowercase-sha256>"
  },
  "label_definition_version": "labels-v1",
  "label_definition": {
    "uri": "s3://<training-bucket>/rerank/datasets/2026-09-12/contracts/labels.json",
    "size_bytes": 1024,
    "sha256": "<64-char-lowercase-sha256>"
  },
  "as_of_ms": 1789142400000,
  "label_maturity_hours": 24,
  "shards": [
    {
      "uri": "s3://<training-bucket>/rerank/datasets/2026-09-12/shards/day=2026-08-16/part-000.parquet",
      "size_bytes": 123456789,
      "sha256": "<64-char-lowercase-sha256>",
      "row_count": 1000000,
      "event_date": "2026-08-16",
      "min_event_time_ms": 1786809600000,
      "max_event_time_ms": 1786895999999
    }
  ],
  "row_count": 28000000,
  "content_sha256": "<canonical-manifest-sha256>"
}
```

要求包括：

- 覆盖连续 28 个已成熟自然日；每天可以有多个 shard；
- `row_count` 等于所有 shard 行数之和，`content_sha256` 是排除自身字段后的 canonical JSON
  SHA-256；
- Feature Schema、Label Definition 和 shard 必须位于 Manifest 所在的受限 S3 prefix 内；
- 每个 Parquet batch 至少包含 `event_time_ms`、配置中的模型特征，以及四目标所需标签列；
- `completion` 由 `watch_duration_ms / content_duration_ms >= 0.95` 计算；
- 其他三个目标分别提供标签列和对应的 `*_valid` mask；
- 名称以 `_7d` 结尾的统计特征同时提供 `__window_start_ms` 与 `__window_end_ms`，窗口必须严格对应
  当前曝光之前的 7 天；
- 首版模型特征不得包含原始 `user_id`。

完整类型定义见
[`dataset_manifest.py`](recommend/train/comm/datasvr/dataset_manifest.py) 和
[`data-model.md`](docs/architecture/data-model.md)。

## 训练配置

下面的配置可作为 `configs/rerank.yml` 起点。特征名必须与 Feature Schema 和 Parquet 列一致。
当前 `categorical_cardinalities` 须与类别特征数量一致且每项至少为 2；训练装配时会根据本次冻结
词表的 missing/OOV 和实际类别数生成最终 embedding cardinality。

```yaml
model_name: rerank
checkpoint_dir: runs/rerank-20260913-001
seed: 7
batch_size: 4096
max_epochs: 5
early_stopping_patience: 2
bad_row_policy: fail

execution:
  strategy: single_process_cpu
  intra_op_threads: 16
  inter_op_threads: 1
  reader_workers: 0
  prefetch_batches: 1
  memory_limit_bytes: 68719476736

split:
  train_days: 27
  validation_days: 1
  refit_days: 28

optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0001

metrics:
  minimum_valid_rows: 10000
  auc_warning_floor: 0.5

model_params:
  numeric_features:
    - age_hours
    - effective_watch_rate_7d
  categorical_features:
    - country
    - device_type
  categorical_cardinalities:
    - 512
    - 32
  embedding_dim: 8
  shared_hidden_dims:
    - 128
    - 64
  tower_hidden_dim: 32
  activation: relu
  dropout: 0.1
  loss_weights:
    effective_watch: 1.0
    completion: 1.0
    non_fast_swipe: 1.0
    immersive_click: 1.0
```

首轮真实训练建议根据机器核数调整线程和 batch，但不要静默改变数据切分、随机种子或特征合同。
配置 schema 位于
[`comm/config.schema.yml`](recommend/train/comm/config.schema.yml) 和
[`rerank/config.schema.yml`](recommend/train/models/rerank/config.schema.yml)。

## 查看训练结果

成功 Run 只写出以下四个对象：

```text
s3://<model-bucket>/rerank/training-runs/<run-id>/
├── checkpoint.pt
├── metrics.json
├── fitted-feature-state/state.json
└── lineage.json
```

- `checkpoint.pt`：28 日 refit 后的 production-role PyTorch Checkpoint；
- `metrics.json`：四目标 validation 指标、门禁、训练 trace 与 CPU/RSS/吞吐资源报告；
- `fitted-feature-state/state.json`：28 日重新拟合的归一化统计、类别词表和 OOV 状态；
- `lineage.json`：代码 revision、依赖锁、配置、Manifest、Feature/Label 版本与 seed。

训练完成前后都会严格加载 production Checkpoint。任一 hard gate 失败不会进入 refit；AUC floor
是 soft gate，`WARN` 会保留在 `metrics.json`，但不会单独阻止 refit。

## 常见失败

| 分类 | 含义与处理方向 |
| --- | --- |
| `INPUT_CONTRACT_INVALID` | 配置、URI、schema 或必要字段不合法；先运行 `validate` 定位 |
| `DATA_INTEGRITY_MISMATCH` | 对象大小、SHA-256、Parquet 行数或内容不一致；重新生成 Manifest |
| `LABEL_NOT_MATURE` | 最新 shard 尚未跨过标签成熟时间；等待成熟或回退数据截止日 |
| `SPLIT_INVALID` | 未覆盖连续 28 个自然日或时间范围重叠；修正分片日期 |
| `RESOURCE_BUDGET_EXCEEDED` | RSS 超过配置上限；检查 batch、特征维度和机器内存 |
| `NUMERICAL_FAILURE` | 特征、loss、gradient 或参数出现 NaN/Inf；检查数据和学习率 |
| `CHECKPOINT_INCOMPATIBLE` | role、格式、模型结构或 lineage 不匹配；不要宽松加载 |
| `EVALUATION_GATE_FAILED` | 某目标样本量不足、单类或不可评估；检查标签 mask 与验证日分布 |
| `TRAINING_OUTPUT_CONFLICT` | `run-id` 已存在；使用新的唯一标识 |
| `TRAINING_OUTPUT_WRITE_FAILED` | S3 写入或回读失败；检查权限、prefix 和对象完整性 |

错误输出包含 `run_id`、`stage`、`category` 和脱敏摘要。训练代码不会打印完整样本或带签名的
URI。

## Finder 目录对应

| doku-train | Finder 对应概念 | 职责 |
| --- | --- | --- |
| `recommend/train/comm` | 公共训练框架 | 配置、数据读取、训练流水线、评估、Checkpoint |
| `recommend/train/comm/datasvr` | datasvr | S3/Manifest/Parquet adapter，不是常驻服务 |
| `recommend/train/models/rerank` | rerank model | 特征转换、shared-bottom 网络与 masked loss |
| `recommend/train/run.py` | 统一入口 | Typer composition root |
| `recommend/train/train_rerank.py` | 精排用例 | validate、27/1 evaluation、28 日 refit、输出写入 |
| `recommend/train/offline` | 离线评估 | 后续离线分析边界，不依赖 `doku-offline` 仓库 |
| `recommend/train/tools` | 工具 | Checkpoint 等离线工具入口 |

目录名保持可对应，但实现不复制 Finder 的 TensorFlow GPU、参数服务器或内部平台依赖。

## 本地开发

```bash
make check
```

该命令检查锁文件、格式、lint、mypy、unit、contract 和纯 CPU smoke。单独运行开发规模容量探针：

```bash
make capacity
```

当前设计容量是每天约 100 万有效曝光、最多 28 个成熟自然日，即约 2800 万行。仓库已经具备
流式读取、在线数值统计和 RSS 预算中止，但尚未完成生产容量基准；开发 fixture 的耗时不能作为
生产 SLA。首次生产 benchmark 建议从 32 vCPU / 128 GiB 开始，并记录吞吐、峰值 RSS 和各阶段
耗时。

## 文档索引

- [协作与架构约束](AGENTS.md)
- [系统概览](docs/architecture/system-overview.md)
- [组件划分](docs/architecture/components.md)
- [数据模型](docs/architecture/data-model.md)
- [接口合同](docs/architecture/interfaces.md)
- [README 信息架构](docs/architecture/developer-readme-design.md)
- [Spec 001 需求](specs/001-cpu-training-scaffold/requirements.md)
- [Spec 001 设计](specs/001-cpu-training-scaffold/design.md)
- [Spec 001 验证证据](specs/001-cpu-training-scaffold/verification.md)
