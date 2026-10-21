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

## Finder 启用接口纠正后的最终验证

根据用户反馈，改用组内 hist_attn_query/skip_hist_attn 和顶层 attn_scale；不保留未发布的 hist_attn_configs 自定义接口。

- 相同环境执行 `make check`：exit 0；锁文件、158 文件格式、Ruff、mypy 59 源文件通过；206 passed、3 skipped，98.40 秒。
- 专项 `uv run pytest recommend/train/tests/unit/test_attention.py -q`：22 passed，0.88 秒。新增开关/缺省尺度、显式启用必须声明 query、公共 schema 必填及未知字段约束仍生效、旧自定义接口被拒绝的验证。
- `git diff --check` 与文档本地链接检查通过；仍未推送或部署。

## 字段组合/历史聚合拆分后的最终验收

Source: 用户要求按 Finder 的计算阶段而非字段名实现。以下结果取代前文对旧启用接口的验收结论；历史记录仅供追溯。

- 首轮编码/数值专项：54 passed，4.46 秒。
- 首次完整检查发现多字段 train 成功但独立 test 的 mask 宽度不同：冻结配置排序改变了共享源字段的最后编码组。修复为字段级统一 padding 容量，拒绝冲突的截断策略；新增顺序无关回归。另补齐 masks 类型标注。
- 修复后 `uv run pytest recommend/train/tests/unit/test_attention.py recommend/train/tests/smoke/test_declarative_train.py -q`：48 passed，64.10 秒。
- 最终 `UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/Users/yujiatang/doku/doku-train/.venv UV_CACHE_DIR=/tmp/doku-train-uv-cache UV_OFFLINE=1 make check`：exit 0；69 包锁文件、158 文件格式、Ruff、mypy 59 源文件通过；222 passed、3 skipped，107.01 秒。
- `git diff --check` 通过；AGENTS、architecture、Spec015 本地文档链接检查通过。
- 数值覆盖 concat/sum/avg × mean/weighted_mean/attention/none；真实执行合成 DNN/DeepFM 的单字段/多字段 attention train/test，以及带 dropout 的恢复一致性。包含 padding/OOV/空历史、梯度、FM 使用聚合输出、原 mean/weighted_mean 迁移、长度/mask 对齐、同步尾截断和 schema 拒绝旧接口。
- 无云训练、镜像构建/部署或发布。配置语义变化需显式迁移后新建 Run，不声称旧 checkpoint 可跨配置恢复。
