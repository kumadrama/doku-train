# 验证记录

2026-10-13 本机执行：

- `UV_CACHE_DIR=/private/tmp/doku-uv-cache make check`：通过。lock 校验、82 文件格式检查、
  lint、33 源文件 mypy 成功；55 tests passed in 55.69s，包含合成 CPU smoke。
- 三份当前本地 YAML 分别执行 `python -m recommend.train.run validate --config <path>`：成功。
- `git diff --check`：通过；AGENTS.md/README.md 的本地文档索引目标检查通过。
- `test_frozen_input.py`：源 Manifest 修改后仍使用冻结输入，冻结合同损坏失败，
  Manifest 摘要不符/身份缺失失败，旧 dataset 参数被 schema 拒绝。
- 既有分片 checksum、resume、评估和三模型训练测试一并通过。

首次默认 `make check` 因沙箱不能访问用户 uv 缓存而中断；改用 /private/tmp 缓存后完整通过。
本次没有真实数据训练、搜参或测试集读取。历史 runs 未修改；本地实验 YAML 保持 Git 忽略。
