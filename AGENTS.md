# doku-train 协作约束

本文件适用于仓库内全部目录，是本仓唯一的 Agent 协作规则入口。不要新增子目录
`AGENTS.md`；需要补充的约束统一更新到本文件，并同步维护文档索引。

## 仓库目标

`doku-train` 是 Doku 推荐精排模型的 CPU-first 训练仓库。训练框架与模型实现在同一仓库，
但必须通过稳定接口解耦。容量规划参考每天约 100 万条有效曝光、28 个自然日，
不将该规模作为输入天数限制。实际日期与阶段由配置声明，提供单机 CPU 上可复现的
训练、评估、检查点和训练产物写出能力。

借鉴 Finder 的结构时，只继承“公共训练能力与模型实现分层、配置驱动、统一入口、评估与
导出分离”等边界；不得复制 TensorFlow GPU、CUDA、NCCL、MPI、参数服务器、WePS/PSStor、
内部数据服务或平台进程管理代码。

## 仓库边界

| 本仓负责 | 本仓不负责 |
| --- | --- |
| CPU 训练运行时、模型定义、损失与优化 | 曝光日志采集、标签归因和训练样本生产 |
| 直接从 S3 读取精确 Manifest、Parquet 分片并校验特征合同 | 直读业务 PostgreSQL、Redis 或线上 RPC 拼训练数据 |
| 时间切分、训练、评估、Checkpoint | GPU 训练和未经批准的多机训练 |
| 框架专属训练 Checkpoint、指标、Feature State 与 lineage | 推理 Artifact、在线 Ranker 服务和业务可见性判断 |
| 本地/CI 验证入口 | S3、调度器、IAM、计算集群等基础设施创建 |

相邻系统职责：

- 当前已有数据流负责离线特征派生、标签与样本构建，并通过 S3 上的不可变 Manifest 交付训练
  数据；本次使用 `doku-offline` 已发布 v4 数据，经版本化 adapter 消费，不能 import 上游实现作为运行依赖；
- 在线系统的消费合同尚未确定；后续 exporter 只能读取成功训练产物，不能引用训练内部实现；
- `doku-monitoring` 负责对生产训练任务、制品发布和在线模型进行监控告警。
- 数据湖导出编排（RDS 快照与 DynamoDB PITR 导出）属于 `doku-data-export`；
- 在线服务的读写路径与 RPC 实现属于 `doku-mono`；
- 共享 Go 基础库属于 `doku-lib`，生成的 Kitex 代码属于 `doku-kitex-gen`；
- Terraform、IAM、S3 桶、EventBridge 规则与环境装配属于 `doku-iac`；

跨仓协作只能通过版本化数据合同或制品合同完成，不能通过复制内部类型或直接 import 实现。

## 文档与 AI-native 工作流

### 文档分层

```text
doku-train/
├── AGENTS.md
├── docs/
│   ├── source/                 # 原始设计输入；默认只读
│   └── architecture/           # 当前有效的系统级整理
└── specs/
    └── <NNN-feature-name>/
        ├── requirements.md
        ├── design.md
        ├── tasks.md
        └── verification.md
```

- `docs/source/` 保存用户提供的原始文档或原样导出。除用户明确要求同步原文外，不修改既有
  内容；不要把 Agent 推断写回原始文档。
- `docs/architecture/` 保存系统级、跨 Spec 的稳定设计。当前行为与目标不一致时必须分别标注
  “当前状态”和“目标状态”。
- 系统级结论应附 `Source:`。由已有材料推导的内容标记 `[INFERRED]`；缺少关键信息时标记
  `[NEEDS CLARIFICATION: ...]`，并在进入实现前向用户确认。
- 功能变更统一进入 `specs/<NNN-feature-name>/`。先确认 `requirements.md`，再确认
  `design.md`，然后生成 `tasks.md`；`verification.md` 必须给出可执行证据，不以“人工看起来
  正常”作为唯一验收方式。
- 代码、配置或合同变化必须同步更新对应 architecture/Spec；禁止只改代码不改目标设计。

### 当前文档索引

- [Spec 016：Finder 结构公共 loss](specs/016-shared-nn-loss/requirements.md)、[设计](specs/016-shared-nn-loss/design.md) — 需求与设计已确认；公共 loss 已实施，验证见 [验证](specs/016-shared-nn-loss/verification.md)。
- [Spec 017：树阈值分桶](specs/017-tree-threshold-binning/requirements.md)、[设计](specs/017-tree-threshold-binning/design.md)、[任务](specs/017-tree-threshold-binning/tasks.md)、[验证](specs/017-tree-threshold-binning/verification.md) — 已实现并通过本地验证；四任务云训练及阈值导出已验证。

- [Spec 015：NN 无参历史 Attention](specs/015-parameter-free-attention/requirements.md)、[设计](specs/015-parameter-free-attention/design.md)、[任务](specs/015-parameter-free-attention/tasks.md)、[验证](specs/015-parameter-free-attention/verification.md) — 本地实现与完整验证通过；尚未部署。

- [Spec 014：SageMaker 树实验](specs/014-tree-sagemaker-experiments/requirements.md)、[设计](specs/014-tree-sagemaker-experiments/design.md)、[任务](specs/014-tree-sagemaker-experiments/tasks.md)、[验证](specs/014-tree-sagemaker-experiments/verification.md) — 本 PR 交付平台树模型与阶段支持；实验编排脚本和本地实验产物不包含在本 PR。

- [Spec 008：SageMaker CPU 实验训练需求](specs/008-sagemaker-dnn-training/requirements.md)、[设计](specs/008-sagemaker-dnn-training/design.md)、[验证](specs/008-sagemaker-dnn-training/verification.md)、[存储规划](specs/008-sagemaker-dnn-training/storage-plan.md)、[任务](specs/008-sagemaker-dnn-training/tasks.md) — 已整理会话确认需求；PROD 单机 CPU DNN/DeepFM 实验、15 并发规划；首期本地提交、AWS 网页看 AUC/loss、不用 Studio；本地实现及原生 amd64 容器验证已通过，尚未云端部署；step 曲线使用本地 TensorBoard。

- [Spec 013：每日 test AUC 看板](specs/013-test-auc-dashboard/requirements.md)、[设计](specs/013-test-auc-dashboard/design.md)、[任务](specs/013-test-auc-dashboard/tasks.md)、[验证](specs/013-test-auc-dashboard/verification.md) — 训练评估日志与 monitoring 四任务多模型看板；仅本地实现，未部署。

- [Spec 012：跨日合同演进](specs/012-daily-contract-evolution/requirements.md)、[设计](specs/012-daily-contract-evolution/design.md)、[任务](specs/012-daily-contract-evolution/tasks.md)、[验证](specs/012-daily-contract-evolution/verification.md) — 标签版本合并、旧日期新增特征默认值与逐日证据冻结；生产恢复按本次明确授权执行。

- [Spec 011：AppConfig 运行开关](specs/011-appconfig-runtime-policy/requirements.md)、[设计](specs/011-appconfig-runtime-policy/design.md)、[任务](specs/011-appconfig-runtime-policy/tasks.md)、[验证](specs/011-appconfig-runtime-policy/verification.md) — 训练、发布、Redis 分发与清理的独立运行控制；禁止将独立训练 timer 打开。

- [Spec 010：模型与特征产物 Redis 分发需求](specs/010-redis-model-distribution/requirements.md) 与
  [设计](specs/010-redis-model-distribution/design.md) — 保留 S3，新增 Redis 分发、跨服务版本一致性及统一清理；本仓已实现，云端/消费者待接入。
  [实施任务](specs/010-redis-model-distribution/tasks.md)、[验证](specs/010-redis-model-distribution/verification.md)。

- [Spec 009：树模型每日训练与发布需求](specs/009-daily-tree-model-publication/requirements.md) 与
  [设计](specs/009-daily-tree-model-publication/design.md) — 已实现本仓入口；云端与消费者接入待验证。


- [Spec 006：按日投影训练输入](specs/006-daily-projected-input/design.md) — 固定日期目录、投影兼容与训练输入冻结。

Spec 004 与本文件的新规则取代 Spec 001/003 及尚未更新的 architecture/README 中
固定目标、固定日期、强制 refit、旧配置和评估门禁的描述；这些旧描述仅说明历史实现，
不能作为新增约束。实施时同步迁移对应文档及测试，不复制旧语义。

Source: 2026-10-10 用户确认的新配置唯一权威与门禁移除决策。

- [Spec 004：CPU 三模型对比需求](specs/004-three-model-comparison/requirements.md) — 固定 v4
  数据集、6/1/1 实验切分、drama 历史 pooling 与已确认的七任务实验需求。
- [Spec 004：声明式配置与实验设计](specs/004-three-model-comparison/design.md) — schema 驱动、
  公共编码、Finder 式 FM、全仓审查处置与实施顺序；整体方案已确认，实施中；DNN/DeepFM 共用 rerank。
- [Spec 004：字段合同与三模型配置示例](specs/004-three-model-comparison/config-contract.md) —
  Finder 字段对应、表达式范围、历史截断/状态、树标量及独立设计示例。
- [Spec 004：实施任务](specs/004-three-model-comparison/tasks.md) — 功能迁移与验证进度。
- [Spec 005：行为标签接入](specs/005-behavior-label-delivery/design.md) — nullable 标签与三模型验证。
- 实验 YAML、结果报告、权重和运行产物仅保留本地，不随源码发布；测试使用独立合成 fixture。
- [Spec 004：审查与验证](specs/004-three-model-comparison/verification.md) — 既有重构审查及主仓验证证据。
- [原始资料登记](docs/source/README.md) — 原始设计输入的登记与只读规则。
- [系统概览](docs/architecture/system-overview.md) — 仓库定位、上下游边界与阶段演进。
- [组件划分](docs/architecture/components.md) — 目标目录、组件职责和依赖方向。
- [数据模型](docs/architecture/data-model.md) — Dataset、Feature、Run 与 Training Output 元数据。
- [接口合同](docs/architecture/interfaces.md) — 输入 Manifest、训练入口、模型插件和训练输出。
- [开发者 README 信息架构](docs/architecture/developer-readme-design.md) — 面向算法开发者的
  S3-first 上手路径、命令示例边界和验证要求。
- [S3-first Developer README 实施计划](specs/002-s3-first-readme/tasks.md) — CLI 模块入口、根
  README、合同测试和文档状态同步的 TDD 步骤。
- [Spec 003：XGBoost 精排基线需求](specs/003-xgboost-rerank-baseline/requirements.md) — 当实现或
  修改 `models/rerank_tree`、配置声明的 Booster、树模型训练产物或本地 DNN 对比入口时。
- [Spec 003：XGBoost 精排基线设计](specs/003-xgboost-rerank-baseline/design.md) — XGBoost 插件
  边界、流式 DMatrix、历史评估/refit 生命周期（已由 Spec 004 替代）和制品格式。
- [Spec 003：XGBoost 精排基线任务](specs/003-xgboost-rerank-baseline/tasks.md) — 依赖、文件范围、
  实施顺序和完成条件。
- [Spec 003：XGBoost 精排基线验证](specs/003-xgboost-rerank-baseline/verification.md) — 需求到
  自动化测试及本地 smoke train 的追踪关系。
- [Spec 001：CPU 训练脚手架需求](specs/001-cpu-training-scaffold/requirements.md) — 首阶段范围与验收条件。
- [Spec 001：CPU 训练脚手架设计](specs/001-cpu-training-scaffold/design.md) — Finder 对应目录、
  PyTorch CPU 多任务训练、S3 输入、四类训练产物与导出边界。
- [Spec 001：CPU 训练脚手架任务](specs/001-cpu-training-scaffold/tasks.md) — TDD 实施顺序、依赖、
  文件范围、完成条件与提交边界。
- [Spec 001：CPU 训练脚手架验证](specs/001-cpu-training-scaffold/verification.md) — 需求、测试、
  验收条件与运行证据追踪。

## 代码结构硬约束

目标 Python 包沿用 Finder 的 `recommend/train` 命名；具体目录按 Spec 001 分阶段创建：

```text
recommend/train/
├── comm/                       # 公共合同、数据读取、训练流水线与评估
│   ├── datasvr/                # Finder 对应命名；实际是 S3/Manifest reader，不是网络服务
│   └── eval/
├── models/
│   ├── rerank/                  # DNN 与 Finder 式 FM，共用实现，由 fm_layers 区分
│   └── rerank_tree/             # 配置声明任务的 XGBoost 精排基线
├── offline/                    # 离线评估与回放，不代表 doku-offline 仓库
├── tools/
│   └── checkpoint/
├── tests/
├── run.py                      # 统一 composition root
└── train_rerank.py             # 三模型统一精排训练用例
```

依赖规则：

- `comm` 不反向依赖任何具体模型；`comm/datasvr` 是历史对应名，不得实现常驻数据服务；
- `models` 实现统一 Model 接口，不读取环境变量、不访问存储、不启动训练任务；
- `comm/training_pipeline.py` 通过接口装配 Dataset、Model、Evaluator 和 CheckpointAgent，不按模型名
  写条件分支；
- `run.py` 和 `train_rerank.py` 只做参数解析与用例装配，不包含特征算法或模型层定义；
- 模型专属配置放在模型自己的命名空间，公共配置不得反向依赖模型实现细节。

命名规则：

- 与 Finder 可直接对应的文件优先沿用 `config.schema.yml`、`model_params.py`、`data_utils.py`、
  `feature_utils.py`、`training_pipeline.py`、`checkpoint_agent.py`、`model_exporter.py`、
  `rerank_model.py`；
- Python 文件和函数使用 `snake_case`，类型使用 `CamelCase`，配置键和配置声明的目标名使用稳定的
  `snake_case`；
- 不沿用只代表 Finder 私有基础设施的名称和缩写，例如 `tfsvr`、`weps`、`psstor`、`uin`；
- Finder 仓库为结构参考，不复制其未授权的私有源码、配置、模型参数或内部地址。

## CPU-first 与可扩展性

- 默认执行设备必须是 CPU；首阶段只支持单机单训练进程，可使用受控的数据读取 worker 和
  CPU 线程池。
- 依赖清单不得引入 CUDA wheel、NCCL、GPU-only 算子或要求 NVIDIA runtime 的镜像。
- 线程数、DataLoader worker 数、batch size 和内存上限必须显式配置；不得通过探测生产机器
  后静默改变训练语义。
- 训练数据必须分片、流式或批量读取，不得要求把 28 天全量样本一次性装入内存。
- 为未来执行策略保留窄接口，但不得预先实现 MPI、参数服务器或多机容错。多机 CPU/GPU、
  Embedding 分片或增量训练必须单独立 Spec，经容量证据证明有必要后再建设。
- DNN 训练框架固定为 PyTorch CPU，XGBoost 基线固定为 CPU `hist`；二者分别产出训练
  Checkpoint、指标、Feature State 和 lineage。推理格式尚未确定，线上 Go/C++ 消费者不得把
  任一训练 Checkpoint 当成稳定加载合同。

## 数据、特征与标签约束

- 训练只接受不可变 Dataset Manifest 指向的数据分片；禁止发现 `latest` 后直接训练。
- Manifest 必须固定 dataset、feature schema、label definition、数据截止时间、分区、行数和
  校验和；缺失任一关键版本时 fail closed。
- 按 YAML 显式声明训练、验证和测试日期及执行阶段；按事件时间切分，不随机穿越自然日。
  无强制 refit、固定天数或隐式回测窗口；成熟性依据版本化上游证据，不臆造等待小时数。
- 特征处理必须由版本化 Feature Schema 驱动。训练时拟合的词表、归一化统计和 OOV 策略必须
  随训练产物一起写出，未来线上合同不得自行重建。
- 缺失输入按 YAML 的 feature_default_values 填充，不自动生成缺失指示特征或额外 embedding。
  用户可显式选0或-1等默认值；OOV/padding 与标签有效性仍按各自合同处理。
- 用户和内容标识只能作为经过合同批准的特征使用；日志不得打印原始敏感特征、完整样本或凭据。
- 首版不把原始 `user_id` 作为模型输入，不做负采样；保留全部有效曝光，并以标签有效性 mask
  处理多目标缺失。
- 新 schema 是运行时权威，YAML 是实验行为的唯一声明；不保留旧格式、旧目标、旧完播口径、
  固定 27/1/28 或 refit 兼容路径。任务、标签、mask、loss、指标、特征、网络和日期均由配置驱动。
  本次四主任务、神经网络三辅助任务、6/1/1 等是实验配置值，不是框架常量；详见 Spec 004。
- 上游统计特征必须满足样本事件时间的 point-in-time 合同；窗口长度与字段映射由合同和配置
  声明，不硬编码 7 天或字段后缀。本仓负责验证和编码，不从原始行为日志重新聚合。
- 移除训练指标质量门禁：不因某指标不可评估、AUC 下限或下降幅度阻断训练或自动发出质量
  警告，不保留隐藏阈值或预建可选门禁系统。指标不可计算时记录 null、原因和样本统计，
  继续计算其他指标并保存已完成训练产物。数据/执行错误仍如实失败，不能伪造成功。
- 选模指标及权重由 YAML 声明；无法完成选模时显式记录状态并保留可用 checkpoint，
  不静默更换指标或声称已经选出最佳模型。

## 可复现性与训练产物

- 每次训练必须记录 `run_id`、源码 revision、配置摘要、随机种子、Dataset Manifest digest、
  Feature Schema version、Label Definition version 和依赖锁文件 digest。
- 相同的冻结输入、配置、源码和运行环境应产生相同的数据切分与逻辑训练结果；时间戳等运行
  元数据不得参与模型逻辑版本计算。
- 成功 DNN Run 只写出 `checkpoint.pt`，成功 XGBoost Run 只写出 `checkpoint.xgb.zip`；二者都
  同时写出 `metrics.json`、`fitted-feature-state/state.json` 和 `lineage.json`。路径必须
  run-scoped 且 create-only；训练失败不得覆盖已有输出。
- `ModelExporter` 只保留接口，不提供 ONNX 实现。推理 Artifact、signature、parity 与 Go/C++
  serving 合同必须在方案确定后由独立 Spec 定义。
- 未来 exporter 只能读取成功训练产物，不要求固定天数或 refit；导出失败不得修改已完成训练
  的状态。本仓首阶段不负责切换线上 active 版本。

## 配置、安全与外部操作

- Python 项目使用 `pyproject.toml` 与锁文件管理依赖；新增依赖前先确认标准库或既有依赖不能满足。
- DTE/PROD、输入 Manifest、输出位置和身份都必须显式传入，不设置生产默认值。
- 仓库不保存 token、密码、云凭据、环境专属密钥或真实用户样本。
- 未经用户在当前请求中明确授权，不得启动云训练任务、上传模型、修改 S3/调度/IAM、发布模型
  或触碰 PROD。
- 不使用 shell 字符串拼接执行训练步骤；外部命令使用参数数组并检查退出码。

## 验证与交付

- 根目录最终提供统一 `make check`，覆盖格式、lint、类型检查、单元测试、合同测试和最小 CPU
  smoke train；在 Spec 001 实现前，文档变更至少运行 `git diff --check` 并检查所有索引链接。
- 训练逻辑必须覆盖固定随机种子、时间切分、OOV/缺失、Checkpoint 恢复、指标不可计算的记录及失败不声明
  完整训练输出。
- 涉及 28 天容量目标时记录机器 CPU、内存、数据量、吞吐、峰值 RSS 和总耗时；不把开发机的
  偶然耗时声明为跨机器 SLA。
- 完成声明必须附刚运行的验证命令和结果；不得用“应该通过”代替证据。

## Git 约束

- 禁止直接更新远端 `main`/`master`；所有变更通过短生命周期分支和 PR 合入。
- 禁止 force push 覆盖远端历史；确需更新自己的 PR 分支时只使用 `--force-with-lease`。
- 不使用 `--no-verify` 绕过检查，不提交生成缓存、数据集、Checkpoint、模型权重或凭据。

## 2026-10-13 Dataset 简化（当前规则）

Source: 用户确认单 Manifest 地址方案。dataset 只声明 manifest_uri；旧版本/摘要、adapter、
aws_profile、allowed_prefixes 和训练内 daily_prefix 说明被本规则取代。
版本与摘要从 Manifest 读取并校验，AWS 身份由启动环境提供，无生产默认值。
每次 Run 冻结 Manifest/合同原文及摘要，resume/test 使用原冻结记录；不重新解析日更指针。
设计及验证索引：[Spec 007](specs/007-manifest-only-dataset/design.md)、[验证](specs/007-manifest-only-dataset/verification.md)。

## 2026-10-13 树模型每日发布（目标规则，Spec 009）

Source: 用户确认每日训练、独立评估、导出与 S3 发布、版本解析、30 天清理方案；
用户随后授权本仓实施，仍未授权云端操作。

在 Spec 009 树模型范围内，以下目标规则取代上文对应首阶段限制，其他约束继续有效：

- 本仓新增推理模型包、S3 上传/发布清单、回滚与清理能力；基础设施归 iac，消费者实现归 mono。
  exporter 仍只读取成功训练产物，不导入训练内部实现；不提供 ONNX。
- model_type/model_name 均由配置指定；前者选择实现，后者是可复用业务名称。同名不同合同
  允许正常发布，不加兼容性门禁；不可变版本加可更新清单，不覆盖旧版本文件。
- 新增独立发布指标门禁，由 YAML 声明，本次阈值 0.60；不阻断训练产物保存，不恢复旧训练门禁。
- 日更窗口由配置声明并解析为冻结日期列表，训练最多 30 自然日，validation/test 各一天，
  三段各至少一天，允许缺日；不够三段就跳过，不做隐式降级。
- 上游训练投影成功后触发单次流水线；本仓不常驻计时。按创建时间保留 30 天，保护当前发布
  版本及被替换后至少 24 小时的版本；清理须与发布/回滚和活动引用协调。
- 具体需求和目标设计以文档索引中的 Spec 009 为准。上述均非当前实现能力声明。

Source: Spec 009 实施证据见 [任务](specs/009-daily-tree-model-publication/tasks.md) 与
[验证](specs/009-daily-tree-model-publication/verification.md)。本仓本地/S3 入口已实现，
消费端仍是跨仓目标；上述目标不代表已在生产运行。
