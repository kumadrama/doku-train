# Spec 001：CPU 训练脚手架验证与追踪

## 状态

本地实现验证完成。Spec 001 的代码、合同、纯 CPU smoke 和开发规模容量探针已通过；每天约
100 万有效曝光、连续 28 天的目标机器生产 benchmark 尚未执行，因此 NFR001 保持 `PARTIAL`，
本文件不把 112 行 fixture 的结果冒充生产 SLA。

## 状态定义

| 状态 | 含义 |
| --- | --- |
| `PLANNED` | 测试名称、命令和通过条件已冻结，尚未存在可运行实现 |
| `PASS` | 在记录的 source revision 上重新运行命令并满足通过条件 |
| `PARTIAL` | 实现与本地证据通过，但需求中明确要求的生产环境证据尚待执行 |
| `FAIL` | 命令已运行但至少一个通过条件不满足 |
| `BLOCKED` | 实现存在，但外部依赖或授权使验证无法运行；必须记录明确原因 |

## 需求到测试追踪

| 需求 | 设计决策 | 实现任务 | 自动化/证据 | 通过条件 | 状态 |
| --- | --- | --- | --- | --- | --- |
| R001 项目与依赖 | D001, D002, D015 | T1, T12 | layout/dependency tests; `make check` | Finder 路径可导入；CPU index 显式；无 GPU、TensorFlow 或 ONNX runtime；总检查退出 0 | PASS |
| R002 类型化配置 | D002, D003 | T2 | `test_model_params.py`; schema check | 未知字段、非 CPU 策略失败；schema 与 Pydantic 一致；无 export 配置 | PASS |
| R003 输入合同 | D004, D005 | T3 | Manifest/storage tests | 版本、UTC、行数、SHA、URI、prefix 任一错误 fail closed；无真实 AWS 请求 | PASS |
| R004 数据与切分 | D006, D008 | T4, T5 | Parquet/split/label tests | batch 有界；不丢行；每日 27/1、refit 28、backtest 24/2/2 | PASS |
| R005 特征处理 | D007, D008 | T5 | feature unit/contract tests | 只从 train fit；缺失≠0；OOV 独立；7 天 point-in-time；拒绝 raw `user_id` | PASS |
| R006 多目标 DNN | D008, D009, D010 | T5, T6, T9 | model/label/import-boundary tests | 四个稳定输出；0.95 含边界；masked BCE；`comm` 不 import 具体模型 | PASS |
| R007 训练生命周期 | D010 | T7, T9 | checkpoint/reproducibility/pipeline tests | seed 可重放；role-separated Checkpoint；NaN/Inf/内存超限失败；lineage 不兼容拒绝 | PASS |
| R008 指标与门禁 | D011 | T8, T9, T11 | metric/gate/pipeline/smoke tests | 四目标 count/loss/AUC；不可评估时 block；AUC 只 warn；无 export preflight；通过后一定进入 refit | PASS |
| R009 训练产物与导出边界 | D012 | T7, T9, T10, T11 | training-output/exporter-boundary/smoke tests | 只写四类输出；production Checkpoint 可回读；`ModelExporter` 无实现；未来 export 状态与 Run 独立 | PASS |
| R010 CLI 与错误 | D013 | T11 | CLI/smoke tests | validate/train/backtest 同一装配；阶段为 validate/evaluate/refit/write-output；结构化脱敏错误 | PASS |
| NFR001 容量 | D006, D014 | T4, T9, T12 | capacity probe | 逐 shard/batch；内存预算中止；记录 CPU/RSS/吞吐/耗时；不伪造 SLA | PARTIAL |
| NFR002 可复现 | D003, D006, D007, D010, D012 | T2, T4, T5, T7, T10 | reproducibility/checkpoint/output tests | 相同输入/config/source/lock/seed 的切分和逻辑结果稳定；lineage 完整 | PASS |
| NFR003 可测试 | D005, D015 | T1–T12 | `make check` | unit、contract、smoke、schema、format、lint、mypy 全部退出 0 | PASS |
| NFR004 安全操作 | D004, D005, D013 | T1, T3, T10, T11, T12 | storage/CLI/security tests | fixture 无真实用户；默认不访问云；prefix 受限；日志无凭据/raw ID | PASS |

## 验收条件追踪

| 验收条件 | 证据命令 | 需要保留的结果 | 状态 |
| --- | --- | --- | --- |
| AC1 `make check` 纯 CPU 通过 | `make check` | 各阶段退出码与 pytest 汇总 | PASS |
| AC2 锁文件无 GPU-only/ONNX runtime 依赖 | dependency-policy test | 包策略断言和 `uv.lock` digest | PASS |
| AC3 固定 fixture 全链路 | smoke test | evaluation/refit epoch、四个输出 URI、Checkpoint reload | PASS |
| AC4 关键失败路径 | unit + contract tests | 泄漏、SHA、成熟度、OOV、NaN、Checkpoint、输出冲突断言 | PASS |
| AC5 四类训练输出可回读 | training-output contract + smoke | Checkpoint、metrics、Feature State、lineage 内容和摘要 | PASS |
| AC6 导出边界不影响 refit | exporter-boundary + pipeline tests | 无 pre-refit exporter；`ModelExporter` 无实现；refit 成功状态不被未来 exporter 改写 | PASS |
| AC7 四份 Spec 文档一致 | document traceability review; `git diff --check` | R/NFR/AC 均映射；无未决占位；diff 无空白错误 | PASS |

## 分层验证清单

### Unit

```bash
uv run pytest recommend/train/tests/unit -v
```

必须覆盖配置、日期切分、Feature fit/transform、完播边界、模型输出与 loss、role-separated
Checkpoint、指标、门禁、pipeline 状态机、`ModelExporter` 接口边界、CLI 参数和安全策略。

关键否定断言：

- `TrainingPipeline` 构造和执行接口中不存在 exporter、`pre_refit_validator` 或 export callback；
- AUC warning 不阻止 production refit；
- 代码与锁文件不包含 ONNX/ONNX Runtime/`onnxscript`；
- `ModelExporter` 只有 Protocol，没有 concrete implementation 或 composition-root registration。

### Contract

```bash
uv run pytest recommend/train/tests/contract -v
```

必须覆盖 Manifest canonical digest、对象位置、Parquet checksum/row count、Feature/Label 版本，以及
以下精确 Training Output 合同：

```text
<output>/<run-id>/checkpoint.pt
<output>/<run-id>/metrics.json
<output>/<run-id>/fitted-feature-state/state.json
<output>/<run-id>/lineage.json
```

测试必须证明文件集合不多不少、目标已存在时拒绝覆盖、部分写入不返回成功 bundle，以及 JSON
正文和 production-role Checkpoint 均可回读。

### Smoke

```bash
uv run pytest recommend/train/tests/smoke/test_cpu_train.py -v
```

必须在 CPU、本地 fixture 上完成：

```text
validate → fit evaluation features → 27-day train → day-28 evaluate
→ hard/soft gates → fit production features → 28-day refit
→ strict production checkpoint reload → write/read four training outputs
```

测试必须阻止 Boto3 client 构造，证明本地 smoke 没有外部网络依赖。Smoke 不导出 ONNX，也不调用
`ModelExporter`。

### Failure semantics

以下失败必须断言稳定错误分类，并断言不会返回完整 `TrainingOutputUris`：

| 场景 | 错误分类 | 对 refit 的影响 |
| --- | --- | --- |
| Manifest/schema/label digest 不一致 | `DATA_INTEGRITY_MISMATCH` | refit 前失败 |
| 分片越过 `as_of_ms - maturity` | `LABEL_NOT_MATURE` | refit 前失败 |
| 不是 28 个连续成熟自然日 | `SPLIT_INVALID` | refit 前失败 |
| 四目标无有效 mask 或 loss/gradient 非有限 | `NUMERICAL_FAILURE` | refit 前失败 |
| 目标单类/样本不足导致指标不可评估 | `EVALUATION_GATE_FAILED` | refit 前失败 |
| RSS 超过显式预算 | `RESOURCE_BUDGET_EXCEEDED` | 发生阶段失败 |
| Checkpoint lineage/结构/role 不一致 | `CHECKPOINT_INCOMPATIBLE` | 发生阶段失败 |
| 训练输出目标已经存在 | `TRAINING_OUTPUT_CONFLICT` | 已完成 refit 保留成功事实，输出尝试失败 |
| 训练输出写入或 Checkpoint 回读失败 | `TRAINING_OUTPUT_WRITE_FAILED` | 已完成 refit 保留成功事实，输出尝试失败 |

未来 exporter 的失败不在 Spec 001 可执行路径中。后续 Spec 必须验证其失败仅令独立 export attempt
失败，不得修改 production refit 和 Training Run 的既有状态。

### Capacity

```bash
make capacity
```

首次生产规模验证建议从 32 vCPU / 128 GiB 起测，记录但不预设 SLA：

- source revision、dependency lock digest、config digest、Manifest digest；
- row/shard/byte 数；CPU 型号和核数；可用内存；
- download/decode/feature-fit/train/evaluate/refit/write-output 各阶段耗时；
- samples/s、bytes/s、峰值 RSS、内部与交付 Checkpoint 大小；
- 是否在调度窗口内完成。该字段是观测结果，不是 Spec 001 的硬门禁。

资源报告嵌入 `metrics.json`，不得为了容量证据增加第五个训练输出文件。

## 证据写入规则

实现完成后，每次状态改为 `PASS` 必须在同一提交中附一条不可变证据记录：

```text
source_revision=<40-char git sha>
executed_at=<UTC RFC3339>
command=<exact command>
exit_code=0
test_summary=<passed/skipped counts>
machine=<cpu count and memory>
dataset_manifest_digest=<sha256 or local-fixture digest>
config_digest=<sha256>
seed=<integer>
selected_epoch=<integer>
checkpoint_reloaded=<true>
training_output_digests=<checkpoint,metrics,feature-state,lineage sha256>
```

## 不可变本地证据

以下证据对应实现 revision `5ce27d00c7a9259e7f7ca74e83825f3265cb0b4d`。后续只修改本验证
文档与任务勾选，不改变被测代码。

```text
source_revision=5ce27d00c7a9259e7f7ca74e83825f3265cb0b4d
executed_at=2026-09-12T20:29:37.657386Z
command=PATH=/Users/yujiatang/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin UV_CACHE_DIR=/private/tmp/doku-train-uv-cache RUFF_CACHE_DIR=/private/tmp/doku-train-ruff-cache MYPY_CACHE_DIR=/private/tmp/doku-train-mypy-cache make check
exit_code=0
test_summary=72 passed, 1 deselected; uv lock, Ruff format/lint, mypy all passed
machine=arm; cpu_count=15; total_memory_bytes=51539607552; available_memory_bytes=23257432064
dataset_manifest_digest=eda8a176375d06f14fbe79a55b12d2d6ae7934970e35fbff4191730f4f7a22e1
config_digest=10c6fd572c1982e5ab25a7b31539e678916440431ac3a9ffeba3b79d2a989280
dependency_lock_digest=b7e59d68d6f1d70ee7ace9a50f452cb95482081a83f17b26034fb863bcf91376
seed=7
selected_epoch=1
checkpoint_reloaded=true
training_output_digests=checkpoint.pt:a9e88e23aebc38f8cc934beacf80f491e24bf5fafac7ef76b6534b60791403a0,metrics.json:46aaba2dd9a18a7c96bb40bcddd94a27f6da068e4e55f1809b897bdffa3d3dd6,fitted-feature-state/state.json:0e234ff1e67ab30a276e2d4f72df32d1945b6613d23374737add403f374cc130,lineage.json:e82dcc476cf6279eb4d93b69825af1c159d16d8a9d702e93bf9673bec61625d9
gates=effective_watch:PASS,completion:PASS,non_fast_swipe:PASS,immersive_click:WARN
resource=row_count:112,batch_count:28,byte_count:114940,peak_rss_bytes:403587072,wall_seconds:0.685003000,rows_per_second:163.502934,bytes_per_second:167794.885570
resource_stages=validate_input:0.003481375,feature_fit_evaluation:0.013870875,feature_fit_production:0.011716333,train_evaluate_refit:0.653207250,checkpoint_verify:0.001843375
```

容量探针在同一 revision 上执行：

```text
command=PATH=/Users/yujiatang/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin UV_CACHE_DIR=/private/tmp/doku-train-uv-cache make capacity
exit_code=0
test_summary=1 passed in 9.05s
```

生产容量待办：在建议的 32 vCPU / 128 GiB 起始机器上，用已成熟的连续 28 日、约 2800 万行数据
运行相同探针，补齐真实下载/解码/训练/评估/refit/输出耗时与调度窗口结论。在该记录产生前，
NFR001 不得由 `PARTIAL` 改为 `PASS`。
