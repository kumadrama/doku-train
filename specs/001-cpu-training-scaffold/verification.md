# Spec 001：CPU 训练脚手架验证与追踪

## 状态

Verification Baseline。`requirements.md` 已审核，`design.md` 与 `tasks.md` 已建立追踪关系；按用户
要求当前未实施代码，因此所有运行时证据状态均为 `PLANNED`，不以文档检查冒充实现通过。

## 状态定义

| 状态 | 含义 |
| --- | --- |
| `PLANNED` | 测试名称、命令和通过条件已冻结，尚未存在可运行实现 |
| `PASS` | 在记录的 source revision 上重新运行命令并满足通过条件 |
| `FAIL` | 命令已运行但至少一个通过条件不满足 |
| `BLOCKED` | 实现存在，但外部依赖或授权使验证无法运行；必须记录明确原因 |

## 需求到测试追踪

| 需求 | 设计决策 | 实现任务 | 自动化/证据 | 通过条件 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| R001 项目与依赖 | D001, D002, D015 | T1, T12 | `test_project_layout.py`; `test_dependency_policy.py`; `make check` | Finder 路径可导入；CPU index 显式；无 GPU runtime；总检查退出 0 | PLANNED |
| R002 类型化配置 | D002, D003 | T2 | `test_model_params.py`; schema-check | 未知字段、非 CPU 策略失败；两份 schema 与 Pydantic 完全一致 | PLANNED |
| R003 输入合同 | D004, D005 | T3 | `test_dataset_manifest.py`; `test_s3_storage.py` | 版本/UTC/行数/SHA/URI/prefix 任一错误 fail closed；无真实 AWS 请求 | PLANNED |
| R004 数据与切分 | D006, D008 | T4, T5 | `test_parquet_dataset.py`; `test_data_utils.py`; `test_labels.py` | batch 有界；row_count 不丢失；每日 27/1、refit 28、backtest 24/2/2 | PLANNED |
| R005 特征处理 | D007, D008 | T5 | `test_feature_check.py`; `test_feature_utils.py` | 只从 train fit；缺失≠0；OOV=1；7 天 point-in-time；拒绝 raw `user_id` | PLANNED |
| R006 多目标 DNN | D008, D009, D010 | T5, T6, T9 | `test_labels.py`; `test_rerank_model.py`; import-boundary `rg` | 四个稳定输出；0.95 含边界；masked BCE；`comm` 不 import 具体模型 | PLANNED |
| R007 训练生命周期 | D010 | T7, T9 | `test_checkpoint_agent.py`; `test_reproducibility.py`; `test_training_pipeline.py` | seed 可重放；best checkpoint 恢复；NaN/Inf/内存超限失败；lineage 不兼容拒绝 | PLANNED |
| R008 指标与门禁 | D011 | T8, T9, T10, T11 | `test_metrics_utils.py`; `test_gates.py`; `test_training_pipeline.py`; smoke | 四目标 count/loss/AUC；单类不可评估；hard block；AUC 只 warn；ONNX preflight 失败不 refit | PLANNED |
| R009 模型制品 | D012 | T10, T11 | `test_gen_onnx.py`; `test_artifact_utils.py`; smoke | ONNX 四输出/动态 batch；CPU parity；回读 SHA/reload；Manifest 最后写 | PLANNED |
| R010 CLI 与错误 | D013 | T11 | `test_cli.py`; smoke test | 三个显式参数；validate/train/backtest 同一装配；结构化脱敏错误；非零失败码 | PLANNED |
| NFR001 容量 | D006, D014 | T4, T9, T12 | `test_capacity_report.py`; `make capacity` | 逐 shard/batch；内存预算中止；记录 CPU/RSS/吞吐/耗时；无 SLA 伪结论 | PLANNED |
| NFR002 可复现 | D003, D006, D007, D010, D012 | T2, T4, T5, T7, T10 | reproducibility/checkpoint/artifact tests | 同输入/config/source/lock/seed 的切分和逻辑版本稳定；时间戳不进摘要 | PLANNED |
| NFR003 可测试 | D005, D015 | T1–T12 | `make check` | unit、contract、smoke、schema、format、lint、mypy 全部退出 0 | PLANNED |
| NFR004 安全操作 | D004, D005, D013 | T1, T3, T11, T12 | storage/CLI/security tests | fixture 无真实用户；默认不访问云；prefix 受限；日志/签名无凭据/raw ID | PLANNED |

## 验收条件追踪

| 验收条件 | 证据命令 | 需要保留的结果 | 当前状态 |
| --- | --- | --- | --- |
| AC1 `make check` 纯 CPU 通过 | `make check` | 各阶段退出码与 pytest 汇总 | PLANNED |
| AC2 锁文件无 GPU-only 依赖 | `uv run pytest recommend/train/tests/unit/test_dependency_policy.py -v` | 通过的包策略断言和 `uv.lock` digest | PLANNED |
| AC3 固定 fixture 全链路 | `make smoke` | evaluation/refit epoch、ONNX 路径、Artifact digest | PLANNED |
| AC4 关键失败路径 | `uv run pytest recommend/train/tests/unit recommend/train/tests/contract -v` | 泄漏、SHA、成熟度、OOV、NaN、Checkpoint 的失败断言 | PLANNED |
| AC5 Artifact 可回读 | `uv run pytest recommend/train/tests/unit/test_gen_onnx.py recommend/train/tests/contract/test_artifact_utils.py -v` | 四输出 signature、opset、parity 容差、全部 SHA | PLANNED |
| AC6 失败不产生 Manifest | artifact/pipeline failure tests | 写入顺序及失败后 final Manifest 不存在 | PLANNED |
| AC7 四份 Spec 文档一致 | 文档追踪脚本、`git diff --check` | R/NFR/AC 均映射；无未决占位；diff 无空白错误 | PLANNED |

## 分层验证清单

### Unit

```bash
uv run pytest recommend/train/tests/unit -v
```

必须覆盖配置、日期切分、feature fit/transform、完播边界、模型输出与 loss、Checkpoint、指标、
门禁、pipeline 状态机、CLI 参数和安全策略。

### Contract

```bash
uv run pytest recommend/train/tests/contract -v
```

必须覆盖 Manifest canonical digest、对象位置、Parquet checksum/row count、Feature/Label 版本、Artifact
内容摘要、Manifest-last 与回读校验。

### Smoke

```bash
uv run pytest recommend/train/tests/smoke/test_train_rerank.py -v
```

必须在 CPU、本地 `file://` fixture 上完成：

```text
validate → fit evaluation features → 27-day train → day-28 evaluate
→ hard/soft gates → fit production features → 28-day refit
→ ONNX export/parity → artifact readback/reload
```

测试必须阻止 Boto3 client 构造，证明本地 smoke 没有外部网络依赖。

### Failure semantics

下列失败必须断言稳定错误分类，并断言 final `manifest.json` 不存在：

| 场景 | 错误分类 |
| --- | --- |
| Manifest/schema/label digest 不一致 | `DATA_INTEGRITY_MISMATCH` |
| 分片越过 `as_of_ms - maturity` | `LABEL_NOT_MATURE` |
| 不是 28 个连续成熟自然日 | `SPLIT_INVALID` |
| 四目标全部无有效 mask 或 loss/gradient 非有限 | `NUMERICAL_FAILURE` |
| 目标单类/样本不足导致指标不可评估 | `EVALUATION_GATE_FAILED` |
| RSS 超过显式预算 | `RESOURCE_BUDGET_EXCEEDED` |
| Checkpoint lineage/结构不一致 | `CHECKPOINT_INCOMPATIBLE` |
| ONNX checker 或 CPU parity 失败 | `ONNX_EXPORT_FAILED` / `ONNX_PARITY_FAILED` |
| Artifact 回读 SHA 不一致 | `ARTIFACT_VERIFY_FAILED` |

### Capacity

```bash
make capacity
```

首次生产规模验证建议使用 32 vCPU / 128 GiB，记录但不预设 SLA：

- source revision、dependency lock digest、config digest、Manifest digest；
- row/shard/byte 数；CPU 型号和核数；可用内存；
- download/decode/feature-fit/train/evaluate/refit/export 各阶段耗时；
- samples/s、bytes/s、峰值 RSS、Checkpoint 与 Artifact 大小；
- 是否在调度窗口内完成。该字段是观测结果，不是 Spec 001 的硬门禁。

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
artifact_manifest_digest=<sha256 when applicable>
```

当前证据：仅文档基线存在；训练代码、测试输出、容量数据和模型制品均未生成。
