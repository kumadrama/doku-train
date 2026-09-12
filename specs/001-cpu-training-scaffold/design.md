# Spec 001：Finder-compatible CPU 训练脚手架设计

## 状态

Design Baseline for Review；`requirements.md` 已审核通过，本文与 `tasks.md`、`verification.md` 同步形成
待审核的实现基线。当前不开始训练代码迁移。

## 设计目标

在不复制 Finder 私有实现的前提下，让熟悉 Finder 的开发者能够按目录和文件名快速找到 Doku
对应能力；同时把底层实现替换为适合当前规模的 PyTorch 单机 CPU、S3 Dataset Manifest 和可恢复
训练产物合同。线上推理格式尚未确定，不在首版训练闭环中预设。

首版每天消费最多 28 个已成熟自然日、约 2800 万条有效曝光，训练一个 shared-bottom 四目标
精排 DNN。该容量是 benchmark 目标，不是未经测量的 SLA。

## 设计决策追踪

本文所有规范性决策通过下表映射到已审核需求；后文标题标注对应决策编号。

| 决策 | 设计结论 | 需求 |
| --- | --- | --- |
| D001 | 保留 Finder 的 `recommend/train` 目录、入口与常用文件名，不复制私有实现 | R001 |
| D002 | Python 3.12、`uv`、CPU-only PyTorch，并复用 Doku 既有依赖范围和工具配置 | R001, R002, NFR003 |
| D003 | 公共配置与模型配置均使用 `extra=forbid` 的类型化模型，并校验 schema 快照 | R002 |
| D004 | 精确 S3 Manifest 是唯一输入；URI、版本、时间范围、行数与 SHA-256 fail closed | R003, NFR004 |
| D005 | 复用 Object Store port 形态隔离 S3，本地 fixture 走同一 port，不依赖 `doku-offline` | R003, NFR003, NFR004 |
| D006 | Parquet 逐 shard、逐 batch 读取；每日 27/1 评估，通过后 28 天 refit | R004, NFR001, NFR002 |
| D007 | 上游提供 point-in-time 7 天统计值；训练侧验证/编码并冻结 Feature State | R005, NFR002 |
| D008 | 原始 `user_id` 不进签名；全曝光、四目标 mask；完播按 95% 行级规则生成 | R004, R005, R006 |
| D009 | 首模为 PyTorch shared-bottom + 四个 tower，masked BCE 默认等权 | R006 |
| D010 | `TrainingPipeline` 注入模型插件；Checkpoint 严格校验 lineage；refit 使用选中 epoch | R006, R007, NFR002 |
| D011 | refit 前的数据/数值/Checkpoint/可评估性为 hard gate，AUC 及相对变化为 soft gate；未来导出不参与 refit 门禁 | R008 |
| D012 | 只产出四类 run-scoped 训练结果并保留无实现的 `ModelExporter`；ONNX Artifact 与 parity 后置 | R009 |
| D013 | `run.py`/`train_rerank.py` 使用统一 composition root 和结构化错误合同 | R010, NFR004 |
| D014 | 首版只有 `single_process_cpu`；32 vCPU/128 GiB 仅为首次 benchmark 起点 | NFR001 |
| D015 | unit/contract/smoke/failure/capacity 五层验证并由 `make check` 汇总 | NFR003, R001 |

## 复用结论

- `doku-train` 当前没有可直接复用的训练代码；Finder 只复用目录和职责命名，因其私有 GPU/
  TensorFlow 代码与当前目标不兼容，不复制实现；
- 依赖和工程约定优先对齐现有 `doku-mono/reco-offline`：Python 3.12、`uv`、Pydantic 2、
  Boto3、PyArrow、PyTorch CPU、pytest、ruff、mypy；两仓不形成 Python import 依赖；
- S3 边界复用 Doku 已验证的 `Protocol + injected boto3 client + allowed bucket/prefix` 形态，
  但只在本仓实现训练所需的最小 `ObjectStore`；
- AUC、Parquet 和 S3 分别使用 scikit-learn、PyArrow 与 Boto3，不自建同类基础实现；首版依赖中
  不包含 ONNX、ONNX Runtime 或 `onnxscript`。

## 方案选择（D001）

评估过三种组织方式：

1. 完全复制 Finder 目录和 TensorFlow/GPU 运行语义：最容易表面对照，但会携带无用的 Session、
   分布式和私有平台概念，拒绝。
2. 完全按新的 `src/doku_train/{contracts,runtime,...}` 拆分：边界清晰，但与用户熟悉的 Finder
   距离太大，拒绝。
3. 保留 Finder 的 `recommend/train` 顶层、常用子目录和文件名，在文件内部使用类型化合同与窄
   接口重建：采用。

`comm/datasvr` 是唯一有意保留的历史语义名称。它只包含 S3/Manifest/Parquet reader，不启动
server；README 和类型名都必须明确这一点。

## 目录与命名（D001, D002, D015）

```text
doku-train/
├── AGENTS.md
├── docs/
│   ├── source/
│   └── architecture/
├── specs/
│   └── 001-cpu-training-scaffold/
├── configs/
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
│       │   └── checkpoint/
│       ├── tests/
│       │   ├── unit/
│       │   ├── contract/
│       │   ├── smoke/
│       │   └── performance/
│       ├── run.py
│       └── train_rerank.py
├── pyproject.toml
├── uv.lock
└── Makefile
```

命名约束：

- 模块/函数/配置键使用 `snake_case`，类型使用 `CamelCase`；
- 对应核心类型为 `TrainingPipeline`、`DatasetManifest`、`ParquetDataset`、`S3Storage`、
  `RerankModel`、`RerankModelParams`、`RerankAdapter`、`GateResult`、`CheckpointAgent` 和
  `TrainingOutputWriter`、`TrainingOutputUris`、`ModelExporter`；评估函数固定为
  `evaluate_rerank`；
- 四个目标名固定为 `effective_watch`、`completion`、`non_fast_swipe`、`immersive_click`；
- 不引入 Finder 私有基础设施缩写，例如 `tfsvr`、`weps`、`psstor`、`uin`。

## 依赖方向（D001, D005, D010）

```text
run.py / train_rerank.py
          │
          ▼
TrainingPipeline ───────────────→ Model protocol
   │       │                          ▲
   │       ├→ Evaluator / Gates       │
   │       └→ CheckpointAgent         │
   ▼                                  │
DatasetManifest / ParquetDataset   RerankModel
   │
   ▼
S3Storage
```

- `comm`、`tools` 和 `offline` 不 import `models/rerank`；
- `run.py` 在 composition root 根据注册表装配模型，`TrainingPipeline` 不按模型名分支；
- `RerankModel` 只接收 tensor batch，不读取文件、S3、环境变量或 Checkpoint；
- 模型专属特征编码留在 `models/rerank/feature_utils.py`，通用合同和生命周期留在 `comm`。

## 输入与 S3 边界（D004, D005）

当前已有数据流负责在样本事件时间点计算 7 天统计特征、生成四目标标签和 Parquet 分片。本仓不从
原始事件重新聚合特征，而是验证并编码已交付值。

调用方必须显式传入精确 `s3://.../manifest.json`。Manifest v1 固定：

- `manifest_version="1"`、`dataset_id`、`dataset_version`；
- `feature_schema_version/uri/sha256` 与 `label_definition_version/uri/sha256`；
- UTC epoch milliseconds `as_of_ms`、`label_maturity_hours`，以及每个分片的 `event_date`、UTC 时间范围；
- 每个对象的精确 URI、行数、字节数和 SHA-256；
- 全量 `row_count` 和排除自身 `content_sha256` 后的 canonical JSON SHA-256。

`S3Storage` 不扫描 `latest`，不使用 bucket 列表推断输入。测试通过同一 Storage protocol 使用本地
fixture，不能访问真实云环境。

输出由调用方显式传入 run-scoped 前缀。`TrainingOutputWriter` 使用 create-only 语义写入四类训练
产物；同一 `run_id` 已有对象时拒绝覆盖。写入中断只形成失败 Run 的诊断，不能声明为完整训练结果。
本 Spec 不定义线上消费者可加载的推理 Manifest。

`doku-offline` 当前未启用且不是运行依赖；未来 producer 只要生成同版本 Manifest 即可接入。

## 数据、特征与标签（D006, D007, D008）

每行是一条有效曝光，首版保留全部样本，不做负采样。每个目标使用独立 label 与 `*_valid` mask；
缺失、未归因完成或不适用不等于负例。首版 `bad_row_policy` 只允许 `fail`：发现非法值时记录
`bad_row_count` 并中止，不提供静默丢弃分支。

首版模型特征只允许使用 Feature Schema 中已存在的数据流特征：

- 数值特征保存缺失 mask，真实 0 不与缺失混淆；
- 稳定低/中基数类别使用版本化词表和 OOV；无法冻结词表的类别可以在配置明确后使用固定 hash；
- hash seed、bucket 数、词表内容和 OOV index 都进入 Fitted Feature State；
- 7 天统计值必须以曝光事件时间为截止点，contract test 检查窗口边界，训练侧不重新聚合；
- 原始 `user_id` 可作为审计/去重列存在，但不得进入 tensor signature。

四个目标共享稳定列名。上游必须提供 `watch_duration_ms` 与 `content_duration_ms`；训练数据适配器
以 `content_duration_ms > 0` 且 `watch_duration_ms / content_duration_ms >= 0.95` 生成 canonical
`completion`。这是确定性的行级转换，不承担跨事件归因。其他三个目标的精确业务阈值/事件归因
由上游 Label Definition 版本化；训练框架只消费结果列和 mask，不把未确认阈值写死。

## 模型设计（D009）

`RerankModel` 使用 PyTorch CPU：

```text
encoded numeric + categorical embeddings
                  │
                  ▼
        shared-bottom MLP
       ┌──────────┼──────────┬──────────┐
       ▼          ▼          ▼          ▼
effective_watch completion non_fast_swipe immersive_click
    tower/logit  tower/logit  tower/logit  tower/logit
```

隐藏层、激活、dropout、embedding 维度、每个 tower 和 loss weight 由
`models/rerank/config.schema.yml` 校验。每个目标先对有效行计算 mean BCE-with-logits，再按配置
weight 加权并按本 batch 中 active weight 归一化；默认四目标等权，避免标签缺失率直接改变任务权重。

选择 PyTorch 是因为首版只需要 CPU 训练且团队没有 TensorFlow Serving 基建。PyTorch Checkpoint
只是训练产物，不被声明为 Go/C++ serving 合同；待线上进程边界与 runtime 确定后，再独立选择
ONNX 或其他推理格式。

## 每日训练状态机（D006, D010）

```text
VALIDATE_INPUT
      ↓
TRAIN_EVALUATION_MODEL (days 1..27)
      ↓
EVALUATE (day 28)
      ↓
DATA/METRIC HARD_GATES ──fail──→ FAILED
      │ pass
      ▼
REFIT_PRODUCTION_MODEL (days 1..28)
      ↓
WRITE_TRAINING_OUTPUT → RELOAD_CHECKPOINT
```

每日调度的输入截止点初始为 D-2，对应 24 小时标签成熟假设。只有量化上游到达延迟并修改 Label
Definition 后，才可推进至 D-1。

评估模型使用 early stopping 选择 `selected_epoch`。生产 refit 从新随机初始化开始，使用同一模型、
优化器、seed 派生规则和全部 28 天数据，固定训练 `selected_epoch` 次，不再查看 validation 或执行
early stopping。这样既让模型吃到最新成熟数据，也避免在 refit 阶段偷看验证指标。

周度或模型结构变更可以显式运行 `backtest`，使用 24/2/2 或 rolling folds。Backtest 产物不能由
每日命令隐式生成，也不能直接冒充 production refit 训练产物。

## 评估与门禁（D011）

每个目标至少输出 valid/positive/negative count、BCE loss 和 ROC-AUC。没有正负两类或低于配置的
最小有效样本量时标记 `NOT_EVALUABLE`。

Hard gates：

- Manifest/schema/checksum/时间窗口/标签成熟度正确；
- 四个目标满足最小可评估样本要求；
- loss、梯度和参数无 NaN/Inf；
- evaluation Checkpoint 与配置、模型结构和输入 lineage 兼容并可回读。

Soft gates：

- 每个目标 AUC 的绝对值和 warning threshold；
- candidate 相对 previous run 的 AUC 变化，但两者必须带相同的当前 validation slice digest；
  不接受不同日期的历史 AUC，首版未注入当前切片上重评的 previous 时记录为未比较；
- cohort 指标或分布漂移 warning。

Soft gate 初期只记录 `WARN`，不阻止 refit。原因是后验分布会变化，单次离线 AUC 波动不能证明
新模型必然更差；线上 A/B 决策属于后续发布 Spec。

任何未来 exporter 都位于 refit 之后，不是 R008 hard gate。它失败时只把独立导出尝试标为失败，
不得把已经成功的 production refit 或训练 Run 回滚为失败。

production Checkpoint 回读和 training output create-only 校验也发生在 refit 之后，属于 R009 的训练
结果交付条件，不冒充 refit 前门禁；失败时必须保留“refit 已完成、训练输出未完成”的诊断事实。

## Checkpoint、训练产物与导出边界（D012）

PyTorch Checkpoint 位于 Run 私有路径，包含模型/优化器/RNG/epoch/配置和输入 lineage。恢复时任一
逻辑摘要不一致都拒绝加载，不做部分权重兼容。

28 天 production refit 成功后，首版只写出：

```text
<output>/<run-id>/
├── checkpoint.pt
├── metrics.json
├── fitted-feature-state/
│   └── state.json
└── lineage.json
```

`checkpoint.pt` 是 production refit 的训练恢复格式；`metrics.json` 同时承载评估、门禁和资源报告；
Feature State 来自全部 28 天的重新拟合；`lineage.json` 记录 source revision、锁文件摘要、配置摘要、
Dataset Manifest 摘要和 seed。调用方以 TrainExecution 成功以及四个 URI 的返回值识别完整结果，
不能从对象列表猜测成功状态。

`comm/model_exporter.py` 只定义 `ModelExporter` Protocol 及其类型化请求/结果，不注册实现、不由
`TrainingPipeline` 调用，也不引入 ONNX 依赖。该接口的输入只能引用成功的四类训练产物。待
Go/C++ serving 方案确定后，后续 Spec 再定义 ONNX Artifact、signature、parity、交付状态与加载
合同；即使提前实验，也只能在 refit 成功后调用，失败只阻止推理制品交付。

## CPU 与容量（D002, D014）

唯一执行策略为 `single_process_cpu`。配置显式固定 PyTorch intra/inter-op threads、DataLoader
worker、batch size、prefetch 和内存上限；不引入 CUDA wheel、DDP、MPI、PS 或多机容错。

首次生产 benchmark 建议从 32 vCPU / 128 GiB 开始，记录数据量、读吞吐、训练吞吐、峰值 RSS、
每阶段耗时和总耗时。若单机无法进入每日窗口，再以证据创建多进程/多机独立 Spec。

## 错误与可恢复性（D004, D010, D012, D013）

错误使用稳定分类：`INPUT_CONTRACT_INVALID`、`DATA_INTEGRITY_MISMATCH`、`LABEL_NOT_MATURE`、
`SPLIT_INVALID`、`RESOURCE_BUDGET_EXCEEDED`、`NUMERICAL_FAILURE`、`CHECKPOINT_INCOMPATIBLE`、
`EVALUATION_GATE_FAILED`、`TRAINING_OUTPUT_CONFLICT`、`TRAINING_OUTPUT_WRITE_FAILED`。

失败输出包含 `run_id`、阶段和脱敏摘要，不记录完整样本、原始特征、凭据或带签名的 URI。失败
Run 可保留内部 Checkpoint 和诊断，但不得声明四类训练产物完整，且不能覆盖既有 Run 输出。未来
exporter 使用独立错误和状态，不属于本 Spec 的训练状态机。

## 验证设计（D015）

- unit：配置拒绝未知字段、27/1 切分、28 天 refit epoch、mask loss、95% 完播边界、OOV/缺失、
  hard/soft gate、Checkpoint lineage；
- contract：小型 28 日 Parquet fixture、Manifest checksum、point-in-time 7 天特征边界、S3 adapter
  fake、四目标模型输出；
- smoke：纯 CPU 执行 validate → evaluation train → evaluate → refit → write-training-output →
  checkpoint reload；
- failure：未成熟标签、未来特征、单类目标、NaN/Inf、错误 checksum、不兼容 Checkpoint 和训练
  输出覆盖尝试都必须 fail closed；
- capacity：在目标机器记录 28 日数据的吞吐、峰值 RSS 和阶段耗时，不把开发机结果当生产 SLA。

实现阶段的统一入口为 `make check`，最终证据写入本 Spec 的 `verification.md`。

## 明确不在本 Spec 的内容

- 上游标签 SQL 和未确认的有效观看/快滑/沉浸点击业务阈值；
- raw `user_id`、超大 embedding、参数服务器或在线更新；
- TensorFlow、TF Serving、独立 C++/Go serving；
- ONNX 导出、推理 Artifact、输入输出 signature、PyTorch/ONNX parity；
- 自动上线、active 指针、A/B、灰度和回滚；
- 多机训练、GPU、超参搜索和流式增量训练。
