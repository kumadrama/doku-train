# Spec 003：XGBoost 精排基线需求

## 状态

Implemented Locally。用户于 2026-10-09 明确要求在 `models/rerank_tree/` 增加 XGBoost
模型插件，用当前数据与 DNN 做本地能力比较；实现与验证证据见本 Spec 的 `verification.md`。

## 需求

### R101 插件边界

系统必须在 `recommend/train/models/rerank_tree/` 提供独立 XGBoost 插件；公共训练模块不得按
模型名分支，模型插件不得访问 S3、Manifest、环境凭据或训练输出位置。

### R102 同口径输入

XGBoost 必须消费与 DNN 相同的不可变 Manifest、27 日训练/1 日验证/28 日 refit、四目标标签、
有效性 mask、point-in-time 检查和冻结 Feature State。首版不得使用原始 `user_id`。

### R103 四目标模型

系统必须为 `effective_watch`、`completion`、`non_fast_swipe`、`immersive_click` 分别训练一个
二分类 Booster。每个目标只使用自身 mask 为真的样本，固定使用 CPU `hist` 和
`binary:logistic`，并输出原始 margin 作为公共评估器的 logit。

### R104 有界数据读取

训练必须通过 XGBoost `DataIter` 和 `QuantileDMatrix` 分 batch 消费数据，不得先把 28 日原始
特征拼接成单个 NumPy 数组。batch size、线程数、最大 bin 和内存上限必须显式配置。

### R105 评估与 refit

评估模型必须在 27 日训练数据上训练，并以第 28 日验证数据按目标选择最佳 boosting round；
hard gate 通过后，生产模型必须从空模型开始，在全部 28 日数据上按各目标选定 round 重新训练。

### R106 指标可比性

XGBoost 必须复用现有四目标指标和门禁，写出相同含义的 AUC、loss、样本计数、资源报告和
validation slice digest。AUC floor 仍为 soft warning。

### R107 训练制品

成功 Run 必须以 create-only 语义写出且严格回读：

```text
<output>/<run-id>/
├── checkpoint.xgb.zip
├── metrics.json
├── fitted-feature-state/state.json
└── lineage.json
```

`checkpoint.xgb.zip` 必须包含版本化 manifest 和四个 UBJ Booster；它只是训练 Checkpoint，不是
线上 Serving 合同。

### R108 入口与失败语义

统一 CLI 必须增加 `rerank-tree validate|train|backtest`。失败必须沿用结构化、脱敏错误合同；
不支持的配置、空目标样本、非有限预测、内存超限、门禁失败和制品回读失败都必须 fail closed。

### R109 非目标

本 Spec 不实现 SageMaker、GPU、多机 XGBoost、超参搜索、在线 Endpoint、ONNX、Treelite、Go/C++
加载合同、自动发布或在线实验。
