# 数据模型

## 文档状态

目标逻辑模型。字段名用于冻结职责和 lineage，最终序列化形式由 Spec 001 design 固定。

Source: [系统概览](system-overview.md)、[组件划分](components.md)；用户于 2026-09-13 确认当前
数据流直接通过 S3 交付训练数据、首版四个目标、完播阈值 95%、保留全部曝光和 7 天统计特征。

## Dataset Manifest

一次训练输入的唯一入口，指向不可变数据分片。

| 字段 | 语义 |
| --- | --- |
| `manifest_version` | Manifest 序列化合同版本 |
| `dataset_id` | 数据集稳定身份 |
| `dataset_version` | 本次冻结内容版本 |
| `feature_schema_version` | 特征定义与编码语义版本 |
| `label_definition_version` | 标签定义、成熟期和归因口径版本 |
| `as_of_ms` | 数据可见性的冻结截止时间 |
| `partitions` | 有序分片 URI、事件时间范围、行数和校验和 |
| `row_count` | 全部有效样本行数 |
| `content_digest` | Manifest 规范化内容摘要 |

训练命令必须接收调用方指定的精确 Manifest，不发现 `latest`，不把目录列表结果当合同。

## Feature Schema

描述模型可消费的输入，不包含物理存储凭据。

每个 Feature 至少包含：

- 稳定名称与版本；
- 类型：numeric、categorical 或明确批准的固定长度集合；
- 来源列与适用范围；
- 缺失、真实 0、OOV 和未适用的编码语义；
- 数值变换或类别词表策略；
- 是否允许参与训练、评估切片和在线推理。

训练阶段拟合出的 normalization statistics、vocabulary/hash parameters 与 OOV mapping 称为
`FittedFeatureState`，必须与权重一起交付。

## Label Definition

标签合同独立于模型，至少固定：

- 标签列和数据类型；
- 正负例定义；
- 归因窗口与成熟延迟；
- 无效曝光、延迟事件和缺失标签处理；
- 可选样本权重语义；
- 标签版本。

Spec 001 固定支持四个二分类目标；每个目标由独立列、有效性 mask、归因窗口和版本定义：

| 稳定名称 | 语义 |
| --- | --- |
| `effective_watch` | 有效观看；精确业务阈值由上游版本化 Label Definition 提供 |
| `completion` | `watch_duration_ms / content_duration_ms >= 0.95`，时长必须为有效正值 |
| `non_fast_swipe` | 非快滑；精确事件窗口由上游版本化 Label Definition 提供 |
| `immersive_click` | 点击进入沉浸页；精确事件归因由上游版本化 Label Definition 提供 |

精确事件 SQL 和尚未确认的业务阈值不在训练框架中硬编码。框架必须验证四个目标的定义版本，
对无效或未成熟目标使用 mask，不得把缺失当负例。

## Sample Record

每行对应一次有效曝光，首版保留全部行，不做负采样。逻辑字段分为：

- `event_time`、稳定样本键和必要的审计列；
- Feature Schema 批准的 point-in-time 特征，统计特征窗口固定为曝光前 7 天；
- 四个 label 列及对应 `*_valid` mask；
- 可用于数据去重或审计、但不进入模型输入的标识列。

原始 `user_id` 首版不得出现在模型输入签名中。后续若使用，应以独立 Spec 选择受控词表、本地
hash embedding 或外部 embedding；是否建设参数服务器由容量和更新需求决定，不由出现 ID 自动触发。

## Train Config

`TrainConfig` 是可审查配置，不包含凭据或环境专属密钥：

- model name 与 model-specific config；
- execution strategy，首阶段固定为 `single_process_cpu`；
- batch size、epoch、optimizer、learning rate 与 early stopping；
- CPU thread、reader worker 与内存预算；
- 每日 27/1 评估、28 天 refit 规则，以及可选的周度 backtest 规则；
- 随机种子；
- 指标及门禁；
- Checkpoint 与 Artifact 的调用方指定位置。

配置经 schema 校验后生成规范化摘要；未知字段默认拒绝，避免拼写错误被静默忽略。

## Training Run

每次尝试都有独立 `run_id`，状态不等同于 Artifact 状态。

| 字段 | 语义 |
| --- | --- |
| `run_id` | 训练尝试身份 |
| `source_revision` | 代码 Git revision |
| `config_digest` | 规范化 Train Config 摘要 |
| `dataset_manifest_digest` | 精确输入 lineage |
| `dependency_lock_digest` | 运行环境依赖版本 |
| `seed` | 主随机种子 |
| `started_at` / `finished_at` | 运行审计时间，不进入模型逻辑版本 |
| `status` | `CREATED/RUNNING/SUCCEEDED/FAILED` |
| `resource_report` | CPU、内存、吞吐、峰值 RSS 和耗时 |

失败 Run 可以保留诊断元数据，但不能生成“可发布”的 Artifact Manifest。

## Metric Set

指标记录必须绑定：

- Dataset/Feature/Label 版本；
- validation 或 backtest 时间范围；
- 每个目标的有效样本数、正例数和负例数；
- 每个目标的 loss、AUC、实现版本和值；
- cohort 定义和 cohort 样本量；
- 门禁阈值及通过/失败结果。

样本不足的 cohort 必须标记不可评估，不能以 0 冒充真实指标。

## Model Artifact Manifest

可交付制品的唯一入口：

| 字段 | 语义 |
| --- | --- |
| `artifact_format_version` | 制品合同版本 |
| `model_name` / `model_version` | 模型身份与不可变逻辑版本 |
| `run_id` | 产出该制品的训练尝试 |
| `dataset_manifest_digest` | 数据 lineage |
| `feature_schema_version` | 在线/离线特征合同 |
| `label_definition_version` | 训练目标口径 |
| `files` | 权重、配置、特征状态、指标文件及各自摘要 |
| `content_digest` | 整个制品的规范化内容摘要 |
| `compatibility` | ONNX opset、CPU runtime、输入签名与四目标输出签名 |

Checkpoint 与 Model Artifact 使用不同格式版本和目录，不允许线上消费者读取训练 Checkpoint。

## 时间切分与模型角色

每日任务只读取最近 28 个已成熟自然日，按事件时间形成确定窗口：

```text
older ── 27 days evaluation-train ── 1 day validation ── maturity gap ── as_of_ms
       └──────────────── 28 days production-refit ────────────────┘
```

1. `evaluation_model` 只在前 27 天拟合 Feature State 和模型参数，在第 28 天验证；
2. hard gates 通过后，`production_model` 使用相同冻结配置在全部 28 天重新拟合 Feature State 和
   模型参数；
3. 只有 `production_model` 导出可交付 ONNX；两种模型的角色和输入窗口必须写入 Run Metadata；
4. 初始标签成熟延迟按 24 小时设计，调度默认使用截至 D-2 的成熟数据；在量化延迟分布并通过独立
   合同变更后才可推进到 D-1；
5. 24/2/2 或 rolling folds 只用于周度基准和模型结构变更 backtest，不是每日生产模型的数据切分。

框架禁止随机跨日切分，也不得让 validation 数据参与 `evaluation_model` 的特征状态拟合。

## 物理产物

训练恢复目录与发布目录严格分开：

```text
runs/<run_id>/checkpoints/checkpoint.pt

artifacts/<model_name>/<model_version>/
├── manifest.json
├── model.onnx
├── signature.json
├── model-config.json
├── feature-schema.json
├── fitted-feature-state/
├── metrics.json
└── lineage.json
```

`checkpoint.pt` 可以包含 PyTorch/optimizer/RNG 状态，但不进入线上加载合同。`model.onnx` 与
`signature.json` 是 Go 或 C++ 推理侧的框架中立边界。
