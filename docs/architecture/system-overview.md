# doku-train 系统概览

## 文档状态

Implemented Locally。Spec 001 的 CPU-first 训练闭环已经实现并通过本地验证；约 2800 万行的
生产容量 benchmark 尚未执行，本文不把 Finder 能力或开发 fixture 耗时描述成 Doku 生产能力。

Source:

- 用户于 2026-09-12 确认：训练框架与模型放在同一 `doku-train` 仓库，整体结构参考 Finder，
  首阶段建设 CPU 训练架构和 AI-native 文档体系。
- `finder-train@cbd00cdc32f9e843204eda52d52480d71fde0105` 的
  `recommend/train/README.md`、`run.py` 与 `train_rerank.py`。
- `finder-models@0abdcc692a84319833d871e30c0bfd7ee6a22080` 的
  `recommend/train/models/`。
- 用户于 2026-09-13 确认：当前直接接 S3，首版采用四目标模型、保留全部曝光、不使用原始
  `user_id`，目录和文件命名尽量与 Finder 对齐；`doku-offline` 尚未启用。
- 用户于 2026-09-13 确认：首版只产出 PyTorch Checkpoint、指标、Feature State 和 lineage，保留
  `ModelExporter` 接口但不实现 ONNX；未来导出仅在 refit 后执行，失败不改变 refit 结果。

## 目标

`doku-train` 将版本化训练数据转换成可验证、可复现、可回读的精排训练结果：

```text
existing dataflow
  └─ S3 Dataset Manifest + immutable Parquet shards
       ↓
doku-train
  ├─ contract validation
  ├─ point-in-time 7-day feature validation/encoding
  ├─ PyTorch CPU multi-task training
  ├─ temporal evaluation + full-window refit
  └─ checkpoint + metrics + fitted feature state + lineage
       ↓
future exporter (outside Spec 001)
```

系统首阶段服务每天约 100 万条有效曝光。默认使用最近最多 28 个已成熟自然日，即约 2800 万
曝光样本，完成简单 DNN 的单机 CPU 训练。规模目标要求流式读取和受控内存，不等于承诺未经
基准测试的固定完成时间。

## 核心原则

1. **训练数据与训练代码解耦**：本仓不生成曝光标签，也不直读业务存储；只消费冻结 Manifest。
2. **框架与模型解耦**：`recommend/train/comm` 不知道具体网络结构；模型通过窄接口提供前向
   计算和损失。
3. **CPU-first**：首阶段只有 CPU 单机路径，没有隐藏的 GPU 依赖和伪分布式抽象。
4. **契约优先**：输入、特征、标签、指标和训练产物全部带版本与 lineage。
5. **时间一致性**：训练切分与特征值均遵循事件时间和标签成熟时间，防止未来信息泄漏。
6. **训练与发布解耦**：首版只完成训练；未来导出失败不改变已完成的 refit，不切换线上版本。
7. **Finder 可映射**：保留 `recommend/train/comm`、`models`、`offline`、`tools`、`run.py` 与
   `train_rerank.py` 等熟悉入口，同时替换其 GPU 和私有平台实现。
8. **AI-native**：系统事实写入 architecture，功能决策进入 Spec，验证证据与实现任务一一对应。

## Finder 参考边界

### 继承

- 公共训练能力与业务模型分层；
- 配置驱动的统一训练入口；
- 数据读取、训练、评估、Checkpoint 和模型导出分阶段；
- 模型配置和实现位于模型自己的命名空间；
- 工具链通过稳定接口消费训练产物。
- `recommend/train` 的目录层级和 `model_params.py`、`data_utils.py`、`feature_utils.py`、
  `rerank_model.py` 等常用文件名。

### 不继承

- TensorFlow GPU Session、CUDA/NCCL/MPI 与多 rank 启动逻辑；
- 分布式 Embedding、参数服务器、WePS/PSStor 和内部代理；
- 固定容器目录、内部域名、平台 token 与进程清理脚本；
- 通过 shell 拼接命令、运行时动态 patch 或模型名条件分支组织系统。

## 生命周期

1. 调用方提供精确 Dataset Manifest、训练配置和输出位置。
2. `recommend/train/comm/datasvr` 直接从 S3 读取精确 Manifest，验证分片、校验和、标签成熟度
   与时间范围；该目录不运行常驻数据服务。
3. 数据流水线按 batch 读取全部有效曝光，验证并编码上游交付的 point-in-time 7 天统计特征；
   本仓不从原始事件重新聚合。
4. 每日评估模型使用最近 28 个已成熟自然日中的前 27 天训练、最后 1 天验证；不另留长期测试
   天，避免生产模型少看最新成熟数据。
5. `TrainingPipeline` 以 PyTorch 在单进程 CPU 上训练 shared-bottom 四目标 DNN，并按目标有效性
   mask 计算损失。
6. 数据、数值、Checkpoint 兼容性和指标可评估性使用硬门禁；AUC 及相对上一版本的质量变化先
   作为软门禁，必须在同一当前验证切片上比较。
7. 硬门禁通过后，以相同配置在全部 28 个成熟日 refit 生产模型。
8. 成功 Run 向调用方指定位置写出 `checkpoint.pt`、`metrics.json`、
   `fitted-feature-state/state.json` 和 `lineage.json`，并回读 Checkpoint。首版 `ModelExporter`
   只有接口、没有实现，也不在训练状态机中调用。

## 阶段演进

- **Spec 001**：AI-native 文档与 Finder-compatible CPU 单机训练脚手架，支持 shared-bottom 四目标
  DNN 的最小闭环。
- 后续模型 Spec：在不改变 Runtime 接口的前提下增加 DeepFM/DCN 等模型与业务标签。
- 容量证据表明单机无法满足训练窗口后，才设计多进程或多机执行策略。
- 发布编排、线上灰度和回滚属于独立系统及独立 Spec，不与训练循环耦合。
- ONNX Artifact、输入输出 signature、PyTorch/ONNX parity 和 Go/C++ 加载合同在 serving 方案确定后
  单独设计；即使提前实验，也只能在 refit 成功后导出。
- 周度基准或模型结构变更可使用 24/2/2 或滚动窗口做独立 backtest；该 backtest 不替代每日
  27/1 评估与 28 天 refit。
