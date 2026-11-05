# Spec 034 验证记录

初始验证日期：2026-11-04（Asia/Shanghai）。以下历史记录为本地验证；2026-11-05 已授权执行的云端结果见末节。

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

## 初始本地验证时的限制（云端状态见末节）

当时未运行生产训练或 28 天容量基准；测试耗时不代表生产 SLA。未推送 PR、未部署、未重产历史每日数据。现有 exporter 固定四输出与旧输入冻结布局，本 Spec 不扩展混训模型的线上导出发布入口；训练 checkpoint/指标/lineage 可本地审查。采样、模型目标集合、loss 权重和指标过滤须按 config-example.md 在实际实验配置显式声明。

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


## 2026-11-05 云端部署、历史回溯与两天 smoke

Source: 用户授权分别审查、合并和部署，回溯两天 Idea 样本并在 SageMaker 测试；随后明确授权最小 Idea 只读 IAM 变更及部署。

- doku-rec-offline PR #43 合并 `195e155a498d239f9260ac67144cf955e685bcb7`；两个每日 workflow run `41965639449`、`41965643399` 成功。
- UTC 2026-11-02 重产 27,768 行（REGULAR 21,426、WATCH_NOW 6,342）；11-03 重产 34,661 行（REGULAR 27,135、WATCH_NOW 7,526）。120 个特征包含 placement；校验和、曝光唯一性通过，新旧所有原有列逐值一致，无 placement 缺失。
- doku-train PR #38 合并 `c92d3e43e9392a96394420e76bf16208bec16428`；原生 amd64 CI 的完整 `make check`、训练容器和推理容器检查全部通过。本地 ARM 模拟 amd64 TensorFlow 曾发生段错误，不作为训练成功证据。
- doku-iac PR #708、#709 已合并。已审查并应用 Terraform plan：0 新增、2 策略原地更新、0 删除；随后仅刷新 binding output。`terraform validate` 通过。`aws iam simulate-principal-policy` 验证 Idea 前缀 GetObject allowed、PutObject implicitDeny。
- 训练镜像：`703844615824.dkr.ecr.us-east-2.amazonaws.com/doku-train/doku-reco-train@sha256:5eb805bdddb5cfe3a8113d10dea133c5d2e19c4f9f11c93222c2670f86f79c97`；源码来自 PR #38 head `95a13d0c919bea78d1522085d0a461e78ebffc11`，配置、源码和锁文件摘要由容器入口校验。
- SageMaker job：`doku-exp-idea-fyp-smoke-20261105`，us-east-2，单台 ml.m5.4xlarge CPU、100 GB 磁盘。`aws sagemaker describe-training-job` 返回 Completed；训练/计费时间 244 秒。
- 执行配置：UTC 11-02 train、11-03 validation、无独立 test，1 epoch、学习率 0.001；FYP 使用 `u_account_id % 50 == 1`，Idea 全量。仅实验训练，没有模型或 Endpoint 发布。
- `input/sources.json` 确认 FYP、Idea 均 ready：train 分别 18、27,768 行，validation 分别 22、34,661 行。test 明确 skipped/empty；`COMMITTED.json`、checkpoint、冻结输入和指标已上传。
- 验证指标过滤计数：read_more 34,661、watch_now 7,526、vote 27,135，与 Idea 及 placement 汇总吻合。AUC 分别为 0.519172244966012、0.517921217008511、0.7637566019826636；logloss 分别为 0.08254345534036014、0.1348324622672557、0.21016498079639878。
- 数据准备 150.86 秒，训练流程 53.85 秒，总流程 204.73 秒，采样峰值 RSS 1,313,624,064 字节。单次两天 smoke 不能代表 28 天容量 SLA 或模型质量结论。

产物根目录：`s3://prod-data-lake-20260901071345278900000001/ml/sagemaker-experiments/environment=prod/runs/doku-exp-idea-fyp-smoke-20261105/live/`。
验收通过 `aws sagemaker describe-training-job`、`aws s3api list-objects-v2`、下载源状态和指标 JSON 后断言两个来源 ready、阶段 completed 以及逐目标样本数。

FYP 实际保留比例明显低于预期的约 1/50，需单独核验输入用户 ID 分布；本次证明真实双源流程可运行，不代表预期辅助数据量已达成。


## 稳定哈希采样修正

原始列独立核验：11-02 175,603 行、12,931 个账号，无 NULL/0，其中 175,011 行为偶数；直接取模命中 18 行。11-03 173,873 行、14,583 个账号，无 NULL/0，直接取模命中 22 行。与训练源状态精确一致，问题为 ID 低位分布偏斜。

用户授权稳定哈希后取模及重跑。新增 golden hash 向量，覆盖大于 2^53 的相邻 int64、int64 边界、字符串一致性、跨 batch/graph 不变、结构化偶数 ID 的采样分布以及 float ID 拒绝。
执行 `python -m pytest recommend_tf/train/tests/test_stable_hash.py recommend_tf/train/tests/test_mixed_integration.py recommend_tf/train/tests/test_mixed_review.py -q --disable-warnings`：14 passed，21.84 秒。后补 golden 向量再运行 stable_hash 专项。
