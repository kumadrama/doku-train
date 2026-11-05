# Spec 034 验证记录

日期：2026-11-04（Asia/Shanghai）。仅本地合成 fixture，不涉及真实用户样本或云操作。

## 已执行

- TF 全套：`PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 /Users/yujiatang/doku/doku-train/recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests -q -p no:cacheprovider --disable-warnings`：178 passed（134.25 秒）；随后补充预算/字符串处理并执行下列针对性回归。
- 原训练全套：`/Users/yujiatang/doku/doku-train/.venv/bin/python -m pytest recommend/train/tests -q -m 'not performance'`：326 passed、3 skipped（190.33 秒）。
- 类型：`/Users/yujiatang/doku/doku-train/.venv/bin/mypy recommend`：70 个源码文件无错误。
- 格式与 lint：现有环境 `ruff format .` / `ruff check .` 通过。
- `uv lock --check`：通过。统一 make 检查尝试在 format 的 uv 环境同步阶段遇到 PyPI DNS 不可用，无法拉取 hatchling；以上使用已有环境独立运行相应检查，没有绕过代码检查。
- 上游 doku-rec-offline：`PYTHONPATH=tools /Users/yujiatang/doku/doku-train/.venv/bin/python -m unittest discover -s tests -p test_request_consumer.py -q`：9 项通过；其 diff whitespace 检查通过。只修改 projection 白名单、每日配置、测试与说明；保留原配置改动，未部署。

## 覆盖与证据

- test_mixed_integration：双源实际 TF 训练、单源故障降级、全空 skip、验证/测试为空、损坏来源整源回滚、冻结输入重放、checkpoint resume 一致、placement 字符串门控。
- test_mixed_days：多份精确每日 Manifest 直接消费、不依赖 FYP 专属元数据；重复时间窗口拒绝。
- test_mixed_shuffle：固定 seed 可复现、不同 seed 顺序改变、行数与样本集合守恒、sum loss。
- test_filtered_metrics / test_migration：逐指标独立过滤/分母/AUC、空指标、loss 门控与 metric filter 独立。
- test_mixed_input：重复 source ID 拒绝、不可用来源冻结跳过与空清单、所有来源共享 prepared 磁盘预算。

## 限制

未运行生产训练或 28 天容量基准；测试耗时不代表生产 SLA。未推送 PR、未部署、未重产历史每日数据。现有 exporter 固定四输出与旧输入冻结布局，本 Spec 不扩展混训模型的线上导出发布入口；训练 checkpoint/指标/lineage 可本地审查。采样、模型目标集合、loss 权重和指标过滤须按 config-example.md 在实际实验配置显式声明。

## 最后增量验证

- 预算/字符串适配后的 TF 专项与迁移回归：`python -m pytest recommend_tf/train/tests/test_mixed_input.py recommend_tf/train/tests/test_mixed_integration.py recommend_tf/train/tests/test_mixed_days.py recommend_tf/train/tests/test_mixed_shuffle.py recommend_tf/train/tests/test_filtered_metrics.py recommend_tf/train/tests/test_migration.py -q --disable-warnings`（上述 TF 环境）：41 passed，34.72 秒。
- 新增共享预算用例后的 `test_mixed_input.py`：3 passed，1.34 秒。
- 最终 `ruff format --check .`：358 文件无需格式化；`ruff check .`、`git diff --check` 均通过。

## 2026-11-05 每来源单 manifest_uri

新增 source_manifest resolver：精确 Manifest 保留，显式 `{date}` 模板按 split 日期解析每日指针或每日 Manifest，验证文件摘要、不可变运行路径、日期范围及解析期间指针稳定性。多源冻结 resolved_manifest_uris 与 resolution 原文，重放无需访问指针。

使用现有 TF 环境执行：`PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 python -m pytest recommend_tf/train/tests/test_source_manifest.py recommend_tf/train/tests/test_mixed_days.py recommend_tf/train/tests/test_mixed_input.py recommend_tf/train/tests/test_mixed_integration.py -q --disable-warnings`：15 passed。
覆盖两来源单模板实际训练、指针被改坏后的冻结重放、指针摘要不符拒绝、缺 test 日期跳过、精确 Manifest 与列表兼容、混训故障/空数据回归。Ruff lint/format 与 git diff --check 通过。未云部署。

## 2026-11-05 合并前独立审查

修复全部来源跳过时未上传证据、空 test 最终状态误报、失败来源 dtype 污染后续来源三个问题。新增 test_mixed_review.py 三项实际 TF 回归。
最终 TF 全套：186 passed，136.45 秒；Ruff lint/format、git diff --check 通过。
线上只读核对两天输入：UTC 11/02 FYP175603/Idea27768行，11/03 FYP173873/Idea34661行。FYP104特征包含u_account_id；Idea119特征待placement回溯为120。测试配置使用11/02训练、11/03验证、无独立test，1epoch；不会发布模型/Endpoint。

合并前真实合同形状合成检查发现空历史词表初始化失败，已修复为空词表直接映射 OOV，保留 padding 不产生梯度语义；新增整数/字符串回归。空词表及迁移测试26项通过（20.17秒），120列Idea合同形状的合成双源完整实验completed，test skipped/empty。
