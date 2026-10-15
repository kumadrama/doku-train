# 验证

全部验证使用 fake SDK、本地 file registry、合成 CPU 训练数据，不调用线上任务。

- `uv run pytest recommend/train/tests/unit/test_runtime_policy.py recommend/train/tests/unit/test_scheduled.py -q`：初版 20 passed；后续新增完整 smoke 场景。
- `make check`：已通过：140 passed、3 skipped；依赖锁、格式、lint、类型检查全部通过。
- 训练进行中关闭 publication 或 redis：保留 checkpoint 和诊断对象，registry current 不推进。
- 关闭训练/清理：业务依赖不初始化，跳过不计完成指标。
- 无效 JSON/字段/预算与读取失败：fail closed。
