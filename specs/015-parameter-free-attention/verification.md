# Spec 015：验证

Source: 2026-10-21 本地隔离分支验证；仅合成 fixture。

- R1/R2/R3/R4：test_attention.py 覆盖手算 softmax、尺度、置换及 padding 不变性、空历史和零长度序列、OOV、query/history 梯度、DNN/FM 与无新参数。
- R5/R7：schema 拒绝非法尺度、未知选项及树模型使用；模型构造拒绝错误组引用、类型、维度和权重混用。
- R1：缺省 attention 的 mean/weighted_mean 数值回归。
- R6：test_declarative_train.py 增加 DNN/DeepFM attention 训练及独立 test，带 dropout 的 attention epoch resume 与不中断训练参数逐项相同。

首次专项：`uv run pytest recommend/train/tests/unit/test_attention.py -q`：21 passed，1.18 秒。完整验证见下。

## 完整检查结果

2026-10-21，隔离分支 codex/nn-history-attention，基于 main 0096494：

- `UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/Users/yujiatang/doku/doku-train/.venv UV_CACHE_DIR=/tmp/doku-train-uv-cache UV_OFFLINE=1 make check`：exit 0；锁文件 69 包、158 文件格式、Ruff、mypy 59 源文件通过；205 passed、3 skipped，98.48 秒。
- 强化 FM 独立输出断言后，`uv run pytest recommend/train/tests/unit/test_attention.py -q`：21 passed，0.83 秒。
- `git diff --check`、AGENTS/architecture/Spec015 的本地 Markdown 链接检查通过。
- 完整测试包含 DNN/DeepFM attention CPU train/test、带 dropout 的 attention 恢复与不中断训练逐参数相等；合成数据，不代表模型效果提升。
- 没有构建/部署云端镜像、启动 SageMaker 训练或发布模型。
