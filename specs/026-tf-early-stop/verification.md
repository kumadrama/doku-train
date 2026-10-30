# 验证记录

Source: 2026-10-30 本次终端输出；仅本地实现与合成 CPU 验证，未构建云镜像或启动训练。

## 执行结果

- `make lockcheck format lint typecheck test`：锁、格式、lint、mypy 通过；主仓 **318 passed, 3 skipped**（193.93 秒）。
- `env -u UV_PROJECT_ENVIRONMENT uv lock --project recommend_tf --check`：通过。
- `PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 <tf-python> -m pytest recommend_tf/train/tests -q -p no:cacheprovider --disable-warnings`：完整 TF **97 passed**（122.99 秒）。使用已有 Python 3.10 / TF 2.11 环境，该环境所属项目锁文件与本仓逐字节一致。
- `... -m pytest recommend_tf/train/tests/test_early_stop.py -q -p no:cacheprovider --disable-warnings`：新增 **35 passed**（31.89 秒）。
- 最后补齐“全轮指标不可评估、在 max_epochs checkpoint 恢复”的 selection 持久化边界后，重跑 `... -m pytest recommend_tf/train/tests/test_early_stop.py -q -p no:cacheprovider --disable-warnings -k 'minimize_or_unavailable or train_early_stop_disabled'`：**3 passed, 32 deselected**（22.71 秒）。
- 最终 `ruff format --check recommend_tf/train`、`ruff check recommend_tf/train`、`git diff --check` 与 Spec 索引链接检查均通过。

## 行为证据

1. 未配置 early_stop 时跑满 7 个合成 epoch，保持原选模规则且不生成早停记录。
2. AUC 最大化、logloss 最小化；综合指标改善可以在单个任务退化时发生。权重归一化与任务/指标校验通过。
3. patient 边界正确：连续有效评估未改善达到阈值才停，相等值不重置计数；min_delta 只影响计数，仍保存实际最佳 checkpoint。
4. 单项 null/非有限值不重新分配权重，综合分数 null，计数保持；全部不可评估则跑满并记录 unavailable，不执行测试集。
5. 合成图训练在第 5 轮停止、选择第 2 轮；第 5 轮完整 checkpoint 与未启用早停的同轮 checkpoint 逐张量一致。包含 LR 调度及 AdamW optimizer slots。
6. 从第 3 轮恢复得到相同停止轮、选中轮、指标和最终 checkpoint；从已停止 checkpoint 恢复不额外训练；改变配置后恢复被 identity 校验拒绝。
7. 原 selection 配置存在时，early_stop 选模优先；experiment 的测试评估实际读取第 2 轮最佳 checkpoint，而非第 3 轮停训 checkpoint。
8. epoch JSON、最终 early-stop.json、experiment-status.json、TensorBoard 综合分数与 wait 均核验。
