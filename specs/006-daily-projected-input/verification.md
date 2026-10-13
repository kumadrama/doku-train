# 验证

仅使用合成数据，不读取生产记录。

- `uv run pytest recommend/train/tests/unit/test_daily_input.py -q`：解析固定分片、拒绝不兼容投影与并发指针变更，真实 reader 在指针损坏后仍可按冻结输入读取相同数据。
- `make check`：格式、lint、类型、合同、单元和 CPU smoke，执行结果在交付时记录。
- 上游验证包含缺分区、构建失败、源变化、损坏分片和 CAS 冲突时不切换指针。

2026-10-13 实测：`make check` 通过，53 个测试通过（含 CPU smoke）；mypy 33 个源码文件无问题。

配套仓验证：doku-offline 根 `make check` 通过，样本场景 95 测试、monitor 23 测试通过
（1 项依赖隔离 ClickHouse 的集成测试按既有条件跳过）；PROD/DTE `terraform validate` 与变更文件 fmt 通过。
客户端按会话要求运行 repo lint，仍有 5 项不涉及本改造的存量阻断，未修改客户端文件。
