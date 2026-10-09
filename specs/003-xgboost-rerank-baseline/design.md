# Spec 003：XGBoost 精排基线设计

## 决策摘要

- `R101`：新增 `models/rerank_train`，训练算法、配置和 checkpoint codec 留在插件内；S3 与生命周期
  编排留在 composition root。
- `R102`：直接复用 `models/rerank` 已实现的 `RerankBatch`、Feature State、标签计算和时间切分，
  避免建立第二套特征语义。
- `R103`：每个目标一个 Booster，不使用实验性的多输出树；预测指定 `output_margin=True`。
- `R104`：用可重置的 `DataIter` 构造 `QuantileDMatrix`，输入 batch 只在回调期间转成 NumPy；
  `hist` 使用压缩后的 DMatrix，不保留完整原始矩阵。
- `R105`：evaluation 阶段保存每目标 `best_iteration + 1`；production 阶段为每目标重新训练精确
  round 数，不续训 evaluation Booster。
- `R106`：插件实现现有 `RerankLogitAdapter` 的 `logits` 形状，直接复用 `evaluate_rerank` 与
  `build_metric_gates`。
- `R107`：四个 Booster 以 UBJ 存入确定性 ZIP；ZIP manifest 固定目标集合、格式版本、round 数和
  XGBoost 版本。写出后从对象存储回读、严格解包并执行一次 validation batch 预测。
- `R108`：Typer 新增 `rerank-train` 命令组；composition root 装配插件，不修改 PyTorch
  `TrainingPipeline`，避免把梯度优化器抽象强加给树模型。
- `R109`：SageMaker 和推理 Artifact 后续独立设计。

依赖分发使用平台 marker：Linux/生产锁定最小 CPU-only `xgboost-cpu`，macOS 开发机使用官方
`xgboost` wheel 并要求本地 `libomp`。锁文件不得包含 NVIDIA/NCCL 包。由于 macOS 的 PyTorch 与
XGBoost 都使用 OpenMP，统一 CLI 和测试进程在加载 PyTorch 前先初始化 XGBoost；这不改变模型
或数据语义。

## 组件

```text
run.py
  └── train_rerank_xgboost.py
        ├── comm/datasvr + Manifest/S3
        ├── models/rerank feature/data contract
        ├── models/rerank_train
        │     ├── model_params.py
        │     ├── data_utils.py
        │     └── rerank_train_model.py
        ├── comm/eval
        └── comm/training_output.py
```

## 数据矩阵

数值特征沿用 DNN 的标准化值与 missing indicator；类别特征沿用冻结 vocabulary 的
missing/OOV/index 编码，并通过 XGBoost `feature_types="c"` 标记为类别列。这样 DNN 与 XGBoost
拥有相同的信息集合，且 validation 不参与 evaluation Feature State 拟合。

## Checkpoint

```text
checkpoint.xgb.zip
├── manifest.json
└── models/
    ├── effective_watch.ubj
    ├── completion.ubj
    ├── non_fast_swipe.ubj
    └── immersive_click.ubj
```

ZIP 条目名必须是白名单，禁止绝对路径和 `..`；manifest 目标必须与四目标精确一致。插件从字节
加载 Booster，不把 ZIP 解压到工作目录。

## 资源语义

`QuantileDMatrix` 会把 batch 压缩为量化矩阵，仍可能随行数和特征数增长；所以它满足“不把原始
28 日 NumPy 全量拼接”的约束，但不声明无限规模。全量容量结论必须由 `make capacity` 或
SageMaker 单机 CPU 实测给出。
