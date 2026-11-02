# 验证

- 2026-11-02：`uv run pytest recommend/train/tests/unit/test_publication.py recommend/train/tests/unit/test_redis_distribution.py -q`：54 passed，覆盖七天边界、pin/current 保护、联合删除及恢复。
- `make check`：运行中；格式、lint、mypy 已通过。
- 生产发布：待执行。
