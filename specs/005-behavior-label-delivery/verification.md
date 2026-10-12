# Spec 005：验证

2026-10-12 本地验证完成。合成 fixture 用于合同与执行验证，不代表线上训练效果。

| 验证 | 命令 / 方法 | 结果 |
| --- | --- | --- |
| 投影失败基线 | offline：pytest tests/unit/test_behavior_labels.py | 实现前 5 项因 unknown labels 失败，确认投影缺口 |
| 五列及发布链路 | offline：pytest tests/unit/test_behavior_labels.py | 7 passed；正例/空数组/NULL/重复 flag、旧列一致、空 schema、Manifest/Parquet |
| 离线场景完整检查 | fyp-rank-samples：make check | 82 passed；format/lint/mypy/wheel 通过 |
| 离线仓库完整检查 | doku-offline：make check | 全部场景通过；monitor 23 passed、1 skipped；Go vet/test 通过 |
| 三模型新任务 | train：pytest recommend/train/tests/smoke/test_behavior_labels.py | 3 passed；DNN/DeepFM/XGBoost 完成 CPU train + test，每个任务保留 7 条已知正负例并排除 1 条 NULL |
| 训练仓库完整检查 | doku-train：make check | 50 passed；lock/format/lint/mypy 通过 |
| 跨仓真实接口 | offline 实际 run_join + run_build_dataset 生成本地合成产物，再由 train PublishedDataset 读取 | Manifest 摘要、版本、分片校验通过；五列均保持 [1, 0, NULL]，无跨仓运行时 import |
| 改动空白检查 | 两仓 git diff --check | 通过 |
| 工作区入口要求 | doku-app-client：python3 tools/repo_lint/lint.py | 5 项既有阻断；该仓本次未修改，不计为本次两仓检查通过 |

生产新数据集尚未构建；既有 v4 Manifest 和历史实验均不修改。
