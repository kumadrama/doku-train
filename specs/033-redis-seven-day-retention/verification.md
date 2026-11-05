# 验证

- 2026-11-02：`uv run pytest recommend/train/tests/unit/test_publication.py recommend/train/tests/unit/test_redis_distribution.py -q`：54 passed，覆盖七天边界、pin/current 保护、联合删除及恢复。
- `make check`：通过。lock、format、lint、mypy 通过；训练测试 326 passed / 3 skipped；TF 测试 149 passed。3 个跳过用例为未配置真实 Redis URL 的集成测试。
- `git diff --check`、AGENTS/Spec 本地索引链接检查：通过。
- `./build/build.sh` 与 `bash -n build/build.sh build/docker-build.sh`：通过；确认打包后的两个清理入口均为七天。
- 生产发布：待执行。

## 2026-11-05 规划冲突重试验证

- `.venv/bin/python -m pytest recommend/train/tests/unit/test_redis_distribution.py -q`：40 passed；覆盖单次冲突恢复、四次耗尽及重新规划遵守最新保留状态。
- `UV_CACHE_DIR=/tmp/doku-uv-cache UV_NO_SYNC=1 make check`：通过。锁文件、格式、lint、mypy 均通过；训练测试 329 passed / 3 skipped；TF 测试 191 passed / 168 warnings。
- 使用既有虚拟环境：默认 uv cache 不可访问，临时 cache 下依赖同步因 PyPI DNS 不可达失败，故上述完整检查禁用同步。
- `git diff --check`：通过。
- 本次仅本地修改与验证，未部署或重跑生产清理。
