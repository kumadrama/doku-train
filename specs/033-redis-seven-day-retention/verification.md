# 验证

- 2026-11-02：`uv run pytest recommend/train/tests/unit/test_publication.py recommend/train/tests/unit/test_redis_distribution.py -q`：54 passed，覆盖七天边界、pin/current 保护、联合删除及恢复。
- `make check`：通过。lock、format、lint、mypy 通过；训练测试 326 passed / 3 skipped；TF 测试 149 passed。3 个跳过用例为未配置真实 Redis URL 的集成测试。
- `git diff --check`、AGENTS/Spec 本地索引链接检查：通过。
- `./build/build.sh` 与 `bash -n build/build.sh build/docker-build.sh`：通过；确认打包后的两个清理入口均为七天。
- 生产发布：待执行。
