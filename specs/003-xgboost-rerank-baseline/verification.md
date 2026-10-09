# Spec 003：XGBoost 精排基线验证

## 追踪矩阵

| 需求 | 自动化证据 |
| --- | --- |
| R101 | layout/dependency tests；模型源码无 boto3、Manifest 和输出写入 |
| R102 | XGBoost smoke fixture 复用 DNN Manifest、Feature State 和 27/1/28 split |
| R103 | `test_rerank_train_model.py` 验证四目标 Booster、mask 和 margin 输出 |
| R104 | DataIter/mask row-count unit test；smoke 从 Parquet batch 构建 QuantileDMatrix |
| R105 | smoke metrics 验证每目标 selected rounds 与 fresh refit |
| R106 | 复用 `evaluate_rerank`；smoke 检查四目标 metrics/gates/resource report |
| R107 | artifact round-trip、未知/缺失条目拒绝、create-only output contract test |
| R108 | CLI group help、validate/backtest 与共享结构化失败测试 |
| R109 | dependency/source policy 检查无 SageMaker、GPU、Serving 或 exporter 耦合 |

## 执行命令

```bash
make check
uv run pytest recommend/train/tests/smoke/test_xgboost_train.py -q
git diff --check
```

## 当前证据

2026-10-09 在 macOS arm64、Python 3.12.13、XGBoost 3.4.1 上执行：

- `make check`：锁文件、88 个文件格式、ruff、mypy 均通过；pytest 为 86 passed、1 deselected；
- XGBoost smoke 完成 27/1 evaluation、四目标 fresh 28 日 refit、确定性 ZIP 写入、对象回读和预测；
- `uv.lock` 不含 NVIDIA、CUDA 或 NCCL 包。

本证据仅证明开发 fixture 和仓库合同，不代表约 2800 万生产行的训练 SLA；生产容量仍需单独
执行 `make capacity` 或等价单机 CPU benchmark。
