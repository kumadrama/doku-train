# 验收记录

- 监控仓库执行 `PYTHONPATH=src python -m unittest tests.test_model_metrics` （9 项通过）验证历史事件重放、迟到事件、配方与任务合同冲突；提交前执行 `make test`。
- 生产切换时读取当前 40 轮 run 的监控结果、发布日志与模型目录，条件写后核对两个配方摘要及旧配方的最后截止日。
- 首个 30 轮任务执行后，核对实际训练轮数、选模、部署目标、制品对应关系与正式端点合成请求。

兼容 ZIP 两次打包逐字节一致；与生产当前 ZIP 比较，唯一变化的成员为 `pkg/model_metrics_collector.py`。新 ZIP SHA256 为 `61dc184a5ea155fdfd2e763fefb682d3eebda2616235c470ecce6938de64b2b3`。
