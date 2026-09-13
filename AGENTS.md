# doku-train 协作约束

本文件适用于仓库内全部目录，是本仓唯一的 Agent 协作规则入口。不要新增子目录
`AGENTS.md`；需要补充的约束统一更新到本文件，并同步维护文档索引。

## 仓库目标

`doku-train` 是 Doku 推荐精排模型的 CPU-first 训练仓库。训练框架与模型实现在同一仓库，
但必须通过稳定接口解耦。首阶段面向每天约 100 万条有效曝光、最多 28 个已成熟自然日的
滚动训练集，提供单机 CPU 上可复现的训练、评估、检查点和训练产物写出能力。

借鉴 Finder 的结构时，只继承“公共训练能力与模型实现分层、配置驱动、统一入口、评估与
导出分离”等边界；不得复制 TensorFlow GPU、CUDA、NCCL、MPI、参数服务器、WePS/PSStor、
内部数据服务或平台进程管理代码。

## 仓库边界

| 本仓负责 | 本仓不负责 |
| --- | --- |
| CPU 训练运行时、模型定义、损失与优化 | 曝光日志采集、标签归因和训练样本生产 |
| 直接从 S3 读取精确 Manifest、Parquet 分片并校验特征合同 | 直读业务 PostgreSQL、Redis 或线上 RPC 拼训练数据 |
| 时间切分、训练、评估、Checkpoint | GPU 训练和未经批准的多机训练 |
| PyTorch Checkpoint、指标、Feature State 与 lineage | 推理 Artifact、在线 Ranker 服务和业务可见性判断 |
| 本地/CI 验证入口 | S3、调度器、IAM、计算集群等基础设施创建 |

相邻系统职责：

- 当前已有数据流负责离线特征派生、标签与样本构建，并通过 S3 上的不可变 Manifest 交付训练
  数据；`doku-offline` 尚未启用，未来只能作为相同合同的可选 producer，不能成为本仓运行依赖；
- 在线系统的消费合同尚未确定；后续 exporter 只能读取成功训练产物，不能引用训练内部实现；
- `doku-iac` 负责存储、调度和计算资源等基础设施；
- `doku-monitoring` 负责对生产训练任务、制品发布和在线模型进行监控告警。

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

- [原始资料登记](docs/source/README.md) — 原始设计输入的登记与只读规则。
- [系统概览](docs/architecture/system-overview.md) — 仓库定位、上下游边界与阶段演进。
- [组件划分](docs/architecture/components.md) — 目标目录、组件职责和依赖方向。
- [数据模型](docs/architecture/data-model.md) — Dataset、Feature、Run 与 Training Output 元数据。
- [接口合同](docs/architecture/interfaces.md) — 输入 Manifest、训练入口、模型插件和训练输出。
- [开发者 README 信息架构](docs/architecture/developer-readme-design.md) — 面向算法开发者的
  S3-first 上手路径、命令示例边界和验证要求。
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
│   └── rerank/                 # 首个 shared-bottom 多目标 DNN
├── offline/                    # 离线评估与回放，不代表 doku-offline 仓库
├── tools/
│   └── checkpoint/
├── tests/
├── run.py                      # 统一 composition root
└── train_rerank.py             # 精排训练用例
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
- Python 文件和函数使用 `snake_case`，类型使用 `CamelCase`，配置键和四个目标名使用稳定的
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
- 首版训练框架固定为 PyTorch CPU，只产出 PyTorch Checkpoint、指标、Feature State 和 lineage。
  推理格式尚未确定，线上 Go/C++ 消费者不得把 Checkpoint 当成稳定加载合同。

## 数据、特征与标签约束

- 训练只接受不可变 Dataset Manifest 指向的数据分片；禁止发现 `latest` 后直接训练。
- Manifest 必须固定 dataset、feature schema、label definition、数据截止时间、分区、行数和
  校验和；缺失任一关键版本时 fail closed。
- 每日 27/1 训练/验证与 28 天 refit 按事件时间切分，不随机穿越自然日；只使用已经成熟的标签。
- 特征处理必须由版本化 Feature Schema 驱动。训练时拟合的词表、归一化统计和 OOV 策略必须
  随训练产物一起写出，未来线上合同不得自行重建。
- 缺失值、真实 0、OOV 和未适用必须可区分；禁止用默认 0 掩盖缺失。
- 用户和内容标识只能作为经过合同批准的特征使用；日志不得打印原始敏感特征、完整样本或凭据。
- 首版不把原始 `user_id` 作为模型输入，不做负采样；保留全部有效曝光，并以标签有效性 mask
  处理多目标缺失。
- 首版固定四个输出目标：`effective_watch`、`completion`、`non_fast_swipe`、
  `immersive_click`；`completion` 定义为观看时长达到内容时长的 95% 及以上。
- 上游交付的统计特征必须按样本事件时间做 point-in-time 7 天窗口聚合，禁止包含曝光后的行为；
  本仓负责验证和编码，不从原始行为日志重新聚合。

## 可复现性与训练产物

- 每次训练必须记录 `run_id`、源码 revision、配置摘要、随机种子、Dataset Manifest digest、
  Feature Schema version、Label Definition version 和依赖锁文件 digest。
- 相同的冻结输入、配置、源码和运行环境应产生相同的数据切分与逻辑训练结果；时间戳等运行
  元数据不得参与模型逻辑版本计算。
- 成功 Run 只写出 `checkpoint.pt`、`metrics.json`、`fitted-feature-state/state.json` 和
  `lineage.json`，路径必须 run-scoped 且 create-only；训练失败不得覆盖已有输出。
- `ModelExporter` 只保留接口，不提供 ONNX 实现。推理 Artifact、signature、parity 与 Go/C++
  serving 合同必须在方案确定后由独立 Spec 定义。
- 未来 exporter 只能在 28 天 refit 成功后运行；导出失败只阻止推理制品交付，不得否定已完成的
  refit 或修改训练 Run 状态。本仓首阶段不负责切换线上 active 版本。

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
- 训练逻辑必须覆盖固定随机种子、时间切分、OOV/缺失、Checkpoint 恢复、评估门禁和失败不声明
  完整训练输出。
- 涉及 28 天容量目标时记录机器 CPU、内存、数据量、吞吐、峰值 RSS 和总耗时；不把开发机的
  偶然耗时声明为跨机器 SLA。
- 完成声明必须附刚运行的验证命令和结果；不得用“应该通过”代替证据。

## Git 约束

- 禁止直接更新远端 `main`/`master`；所有变更通过短生命周期分支和 PR 合入。
- 禁止 force push 覆盖远端历史；确需更新自己的 PR 分支时只使用 `--force-with-lease`。
- 不使用 `--no-verify` 绕过检查，不提交生成缓存、数据集、Checkpoint、模型权重或凭据。
