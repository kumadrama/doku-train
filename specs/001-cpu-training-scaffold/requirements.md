# Spec 001：CPU 训练脚手架需求

## 状态

Requirements Confirmed；用户已确认进入 `design.md`，尚未开始代码实现。

## 背景

Doku 当前每天约有 100 万条有效曝光，希望先在 CPU 上训练简单 DNN 精排模型。Finder 的训练
仓库提供了框架/模型分层、统一入口、评估和导出等结构参考，但其 GPU、分布式 TensorFlow、
参数服务器及内部平台依赖不适用于 Doku。

本 Spec 建立 AI-native、CPU-first 的最小训练闭环，使后续模型开发建立在稳定的数据合同、
可复现运行和不可变训练产物之上。

Source:

- 用户于 2026-09-12 确认的单仓、CPU 训练与 AI-native 目录要求；
- 用户于 2026-09-13 确认首版只产出 Checkpoint、指标、Feature State 和 lineage；ONNX 及 parity
  后置，任何未来导出失败不得阻止或否定已完成的 refit；
- `docs/architecture/system-overview.md`；
- `finder-train@cbd00cdc32f9e843204eda52d52480d71fde0105`；
- `finder-models@0abdcc692a84319833d871e30c0bfd7ee6a22080`；
- `doku-offline@577bf6c1c6bed1f95a64cccacdae9d5f82905de3`。

## 目标

1. 建立与 Finder 可直接映射的 `recommend/train` Python 包和统一验证入口。
2. 从调用方指定的不可变 Dataset Manifest 读取 Parquet 训练分片。
3. 在单机 CPU 上以 PyTorch 完成 shared-bottom 四目标 DNN 的训练、验证与生产 refit。
4. 支持 27/1 每日评估、28 天 refit、Feature Schema、Label Definition、Checkpoint 与 early stopping。
5. 生成可回读的 PyTorch Checkpoint、指标、Fitted Feature State 和 lineage，并保留未来模型导出的
   稳定接口边界。
6. 为增加新模型保留稳定插件边界，但不实现不必要的分布式能力。

## 非目标

- 曝光采集、标签归因、特征派生或训练样本生产；
- GPU/CUDA、NCCL、MPI、参数服务器、Embedding 分片或多机训练；
- 小时级在线学习、流式增量训练和自动超参数搜索；
- 在线 Ranker 服务、模型 active 指针切换、A/B 灰度或自动回滚；
- 云调度、S3/IAM/集群资源创建和 PROD 发布；
- 在首个 Spec 中实现 DeepFM、DCN、序列模型或 raw `user_id` embedding；
- TensorFlow Serving、独立 C++ serving、Go serving 或线上 Predictor 集成。
- ONNX 导出、推理 Artifact、PyTorch/ONNX parity 与线上加载合同；这些能力在 Go/C++ serving 方案
  确定后进入独立 Spec。

## 功能需求

### R001：项目与依赖

- 项目必须使用 `recommend/train` 包、`pyproject.toml` 和锁文件；
- 顶层和常用文件名必须按 Finder 对应为 `comm`、`models`、`offline`、`tools`、`run.py`、
  `train_rerank.py`、`model_params.py`、`checkpoint_agent.py`、`data_utils.py`、`feature_utils.py`、
  `model_exporter.py`、`rerank_model.py`；
- `comm/datasvr` 可以作为 Finder 对应名保留，但实现必须是 S3/Manifest/Parquet reader，不得启动
  常驻数据服务；
- 依赖必须支持纯 CPU 安装，不下载或要求 CUDA/NVIDIA runtime；
- 根 `Makefile` 必须提供 `make check`，统一执行格式、lint、类型检查、测试和 CPU smoke train；
- 生成缓存、数据分片、Checkpoint 和模型权重必须被 Git 忽略。

### R002：类型化配置

- Train Config 必须经过 schema/类型校验，未知字段默认拒绝；
- 配置必须显式给出模型、随机种子、batch、epoch、优化器、学习率、early stopping、CPU 线程、
  reader worker、时间切分、指标和资源预算；
- 首阶段执行策略只能是 `single_process_cpu`；
- DTE/PROD、输入和输出位置不得有生产默认值；
- 训练框架固定为 PyTorch CPU，Checkpoint 为 PyTorch 格式；本 Spec 不确定可发布推理格式，也不得
  引入 ONNX 运行时依赖。

### R003：输入合同验证

- 训练必须接收精确 Dataset Manifest，禁止扫描或解析 `latest`；
- Manifest 必须固定 Dataset、Feature Schema、Label Definition、`as_of_ms`、分片、行数和校验和；
- 系统必须在训练前验证合同版本、分片存在性、schema、时间范围和内容完整性；
- 任一关键校验失败必须 fail closed，且不得创建完整训练产物；
- 当前 producer 是已有数据流，本仓直接读取 S3；不得依赖尚未启用的 `doku-offline`。

### R004：数据读取与切分

- Dataset Reader 必须以有界内存读取 Parquet batch，不要求一次加载最多约 2800 万条曝光样本；
- 每日 `evaluation_model` 必须用前 27 个成熟自然日训练、最后 1 个成熟自然日验证；
- hard gates 通过后，`production_model` 必须使用相同配置在全部 28 个成熟日 refit；
- 24/2/2 或 rolling folds 只允许作为显式的周度/模型变更 backtest；
- 只允许使用在冻结 `as_of_ms` 前已经成熟的标签；
- 相同 Manifest、配置和随机种子必须产生相同切分与样本顺序；
- 坏样本处理策略必须显式配置并计数，不能静默丢弃；
- 首版必须保留全部有效曝光，不得负采样；标签缺失通过每个目标独立 mask 表达。

### R005：特征处理

- 首阶段支持 numeric 与 categorical 特征；序列特征不在本 Spec；
- Feature Transformer 只能在 training split 上拟合归一化统计与类别状态；
- 缺失、真实 0、OOV 和未适用必须具有不同的合同语义；
- Fitted Feature State 必须可序列化，并随本次训练产物一起写出；
- validation 必须复用 evaluation training split 的冻结状态，不得重新拟合；production refit 在全部
  28 天上重新拟合并写出自己的 Fitted Feature State；
- 统计特征必须来自样本事件时间之前的 7 天窗口并通过 point-in-time 检查；
- 原始 `user_id` 不得进入首版模型输入签名。后续引入 ID 特征必须独立设计词表/hash/OOV，且不
  默认要求参数服务器。

### R006：Rerank 多目标 DNN

- 首个模型命名空间为 `models/rerank`，入口为 `rerank_model.py`，模型名为 `rerank`；
- 模型必须采用 shared-bottom DNN，并输出 `effective_watch`、`completion`、`non_fast_swipe`、
  `immersive_click` 四个二分类 logits；
- `completion` 必须按 `watch_duration_ms / content_duration_ms >= 0.95` 生成；其他标签的精确业务
  阈值由版本化 Label Definition 提供，训练代码不得硬编码未确认口径；
- multi-task loss 必须按目标有效性 mask 计算，loss weight 可配置且默认等权；
- 隐藏层数量、维度、激活、dropout 和类别 embedding 维度必须可配置并受范围校验；
- 模型实现不得读取文件、访问网络、启动进程或决定输出路径；
- `TrainingPipeline` 必须通过模型注册接口装配模型，不得按模型名写业务条件分支。

### R007：训练生命周期

- Runtime 必须设置并记录 Python、数值库与 PyTorch 的随机种子；
- 必须支持 epoch/step 指标、梯度有效性检查、early stopping 和资源预算中止；
- 出现 NaN/Inf、数据耗尽异常或资源预算超限时必须失败并保留诊断；
- Checkpoint 必须包含模型、优化器、step、随机状态、配置和输入 lineage；
- 恢复时合同、模型结构或 lineage 不兼容必须拒绝。

### R008：评估与门禁

- 必须按四个目标报告 validation 的有效样本数、正负例数、loss 和 AUC；
- 指标实现必须带版本；关键 cohort 指标可通过配置启用；
- 样本量不足的指标必须标记不可评估，不能返回伪造的 0；
- schema、数据完整性、label maturity、NaN/Inf、evaluation Checkpoint 兼容性和指标可评估性必须
  是 hard gate；
- AUC 绝对值及其相对上一版本变化初期必须是 soft gate，只生成 warning；比较必须使用同一当前
  validation 切片，不能拿不同日期的历史 AUC 直接门禁；
- refit 前的 hard gate 未通过时 Run 可以保留诊断，但不得 refit 或生成完整训练产物；
- 模型导出不属于本 Spec 的 hard gate。未来的导出只能在 28 天 refit 成功后执行；导出失败只能
  阻止推理制品交付，不得改变已经完成的 refit 或训练 Run 状态。

### R009：训练产物与导出边界

- 每个成功 Run 只能形成以下四类首版训练产物：`checkpoint.pt`、`metrics.json`、
  `fitted-feature-state/state.json` 和 `lineage.json`；
- `checkpoint.pt` 必须来自 28 天 production refit，并可在纯 CPU PyTorch 环境回读；Checkpoint
  至少包含模型、优化器、epoch/step、随机状态、配置和输入 lineage；
- 四类产物必须写入调用方指定的 run-scoped、create-only 路径；失败 Run 不得覆盖既有输出，也
  不得把部分写入声明为完整训练结果；
- 训练产物不是可部署推理 Artifact，线上消费者不得把 `checkpoint.pt` 当成稳定 serving 合同；
- `comm/model_exporter.py` 必须保留类型化 `ModelExporter` 接口，但本 Spec 不提供具体实现、不依赖
  ONNX，也不生成推理 Artifact；
- `ModelExporter` 只接受成功 refit 后的训练产物作为输入。未来 exporter 的成功或失败必须使用
  独立状态表达，不得回写或否定已经完成的 refit；
- ONNX Artifact、输入输出 signature、PyTorch/ONNX parity 以及 Go/C++ 加载兼容性必须在 serving
  方案确定后由独立 Spec 定义。

### R010：CLI 与错误

- `recommend/train/run.py` 必须提供统一 composition root；`train_rerank.py` 至少提供 `train` 和
  `backtest` 用例，`train` 内部按阶段执行 validate、evaluate、refit 和 write-training-output；
- 所有失败返回非零退出码，并输出包含 `run_id`、错误分类和脱敏摘要的结构化信息；
- CLI 不得通过 shell 字符串拼接调用内部训练步骤；
- 日志不得包含原始敏感特征、完整样本、token、密码或云凭据。

## 非功能需求

### NFR001：容量

- 设计容量为每天约 100 万有效曝光、最多 28 个已成熟自然日；
- 训练必须支持分片/流式读取，并受配置的内存预算约束；
- 首次真实容量验证必须记录 CPU 型号/核数、内存、数据量、吞吐、峰值 RSS 和总耗时；
- 在目标机器完成基准测试前，本 Spec 不承诺固定训练 SLA。

### NFR002：可复现性

- 每个 Run 必须记录代码 revision、依赖锁摘要、配置摘要、输入 Manifest 摘要和随机种子；
- 在同一锁定环境中重放时，数据切分、样本顺序和逻辑结果必须确定；
- 允许审计时间等非逻辑元数据不同，但必须从模型版本摘要中排除。

### NFR003：可测试性

- 单元测试覆盖合同、时间切分、缺失/OOV、模型接口、Checkpoint 和门禁；
- 合同测试使用小型确定性 Parquet fixture；
- CPU smoke test 必须完成
  `validate → 27/1 train/evaluate → 28-day refit → write-training-output → checkpoint reload`；
- 测试不得访问真实业务数据库、真实用户数据或外部云环境。

### NFR004：安全与操作

- 仓库和 fixture 不保存真实用户样本或凭据；
- 默认命令只能操作本地临时路径；访问远程输入/输出必须由调用方显式提供；
- 未经用户当次明确授权，测试和开发命令不得启动云任务、上传制品或触碰 PROD。

## 验收条件

Spec 001 实现完成时必须同时满足：

1. `make check` 在纯 CPU 环境退出码为 0；
2. 锁文件中不存在 GPU-only 运行时依赖；
3. 固定 fixture 能完整执行
   `validate → 27/1 train/evaluate → 28-day refit → write-training-output → checkpoint reload`；
4. 时间泄漏、Manifest 摘要不匹配、未成熟标签、OOV、NaN/Inf 和不兼容 Checkpoint 均有失败测试；
5. 四类训练产物能够回读，Checkpoint、指标、Feature State 和 lineage 内容与本次 Run 一致；仓库
   存在 `ModelExporter` 接口但不存在 ONNX 实现或依赖；
6. 失败路径不会声明完整训练产物，也不会覆盖已有输出；未来导出失败不改变已成功的 refit 状态；
7. requirements、design、tasks 与 verification 对同一范围无矛盾，并附执行证据。

## 后续规格候选

- 真实业务 Feature/Label 合同与首个训练 Dataset；
- DeepFM/DCN 等模型插件及基线对照；
- 生产容量验证与训练调度；
- 模型发布、在线兼容性、灰度与回滚；
- ONNX Artifact、输入输出 signature、PyTorch/ONNX parity 与 Go/C++ serving 加载合同；
- 有容量证据后的多进程或多机执行策略。
