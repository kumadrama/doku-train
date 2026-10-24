# Spec 018：验证

Source: 2026-10-24 本地执行证据。

- 实施前 git fetch origin main 成功；git rev-list --left-right --count main...origin/main = 0 0；从 d42eabc 创建 codex/weight-decay-overrides。保留既有 Spec008 未提交记录。
- UV_NO_SYNC=1 UV_CACHE_DIR=/tmp/doku-train-uv-cache make check：锁文件、格式、lint、mypy通过；全量 246 passed, 3 skipped（121.35s）。使用现有虚拟环境，默认 uv sync 因沙箱无法访问 PyPI 构建依赖而失败。
- 新增 smoke 后专项执行：.venv/bin/pytest recommend/train/tests/unit/test_weight_decay_overrides.py recommend/train/tests/smoke/test_declarative_train.py -q -k weight_decay：9 passed, 12 deselected。
- 最终 .venv/bin/ruff format --check .、ruff check .、mypy recommend、git diff --check 均通过。
- 覆盖稳定参数分组、继承与零衰减、三种优化器实际更新一致性、状态恢复后下一步一致性、未知位置/任务和负值拒绝、原单组路径及 DeepFM CPU 训练成功。
- 无云训练、镜像构建或部署。旧镜像不支持新配置。
