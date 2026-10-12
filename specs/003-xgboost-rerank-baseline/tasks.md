# Spec 003：XGBoost 精排基线任务

状态：T301-T305 已完成。

## T301 依赖与配置合同

- 状态：完成。

- 依赖：无。
- 文件：`pyproject.toml`、`uv.lock`、`models/rerank_tree/model_params.py`、schema、配置测试。
- 完成条件：XGBoost CPU 依赖锁定；未知参数、非 CPU tree method 和非法范围被拒绝。

## T302 流式矩阵与四目标训练

- 状态：完成。

- 依赖：T301。
- 文件：`models/rerank_tree/data_utils.py`、`rerank_tree_model.py` 及单元测试。
- 完成条件：DataIter 可重置、按 mask 过滤；四个 Booster 可训练、返回 margin、确定性序列化并
  严格回读。

## T303 生命周期与训练输出

- 状态：完成。

- 依赖：T302。
- 文件：`train_rerank.py`、`comm/training_output.py`、smoke/contract tests。
- 完成条件：完成 evaluation、门禁、28 日 fresh refit、create-only 四制品写出和回读预测验证。

## T304 CLI 与文档

- 状态：完成。

- 依赖：T303。
- 文件：`run.py`、`README.md`、architecture 文档、CLI/layout tests。
- 完成条件：`rerank-tree validate|train|backtest` 可发现，README 给出本地对比命令和限制。

## T305 验证

- 状态：完成。

- 依赖：T301-T304。
- 文件：全仓。
- 完成条件：`make check`、XGBoost smoke train、`git diff --check` 通过；记录验证证据。
