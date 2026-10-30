# Spec 027 验证记录

Source: 2026-10-30 本地实施与实际执行；仅合成数据和 AWS 接口模拟，无云操作。

## 1. 覆盖关系

| 范围 | 可执行证据 |
| --- | --- |
| 滚动窗口、缺日、成熟性/摘要错误、无 test | `test_daily_publication.py` 的 window、ready、daily 测试 |
| 共用训练不改变结果 | 同一合成 train/validation 比较离线/日更选模分数和 checkpoint 全部变量逐元素相等 |
| 导出与制品合同 | EVAL checkpoint → SavedModel reload 最终预测 parity；空历史、默认、OOV、重复/截断；四任务/无标签签名 |
| 重试与完整性 | 同进程/独立进程重复导出摘要相同；损坏状态拒绝；create-only 目标预留；S3 条件写与不可变上传 |
| 平台恢复 | 从封存归档下载→独立导出→发布→Endpoint 更新；任务查询避免重复提交；ECS 接管要求旧执行者停止 |
| 部署故障 | 实际配置与 probe 校验；probe 失败恢复旧配置；模糊提交拒绝盲重试；旧读结果不误判回滚；显式回滚保留自动发布高水位 |

## 2. 已执行命令

```sh
UV_CACHE_DIR=/tmp/doku-tf-uv-cache make check
```

根锁检查、格式、lint、mypy 通过；主仓测试 **318 passed, 3 skipped**（184.06 秒）。首次执行在最后 tf-check 环境同步时因 PyPI DNS 不可达失败，不声明该次 make check 整体成功。

使用已有 Python 3.10 / TF 2.11 环境（其所属项目 uv.lock 与本仓 `cmp` 一致）执行：

```sh
PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  /Users/yujiatang/.codex/worktrees/finder-strict-migration/doku-train/recommend_tf/.venv/bin/python \
  -m pytest recommend_tf/train/tests -q --disable-warnings
```

结果：**111 passed**（126.83 秒）。此后补充了目标目录并发预留与 Endpoint 最终一致性两个用例，补充后的专项 **16 passed**（10.50 秒），包括 create-only 目录预留及 Endpoint 最终一致性用例。最终标准 tf-check 结果见补充记录。

补充检查：`ruff check .`、`ruff format --check .`、`mypy recommend`、根与 TF `uv lock --check` 均通过，未新增依赖或更改锁文件。

### 标准入口最终补跑

在允许下载锁定依赖后执行 `UV_CACHE_DIR=/tmp/doku-tf-uv-cache make tf-check`：退出码 0，TF 锁检查通过，最终全量 **113 passed**（159.06 秒）。包括最后的公共准备参数注入、16 项日更/导出/平台测试；当前仓库本地 TF 环境已完成同步。该结果补齐先前 make check 因 DNS 中断的最后阶段，根检查及主仓测试无需重复。

最终 `git diff --check`、格式、lint、mypy 检查通过；文档链接检查覆盖 9 份文件，见本次终端记录。实现未修改依赖锁文件。

## 3. 可重复的合成吞吐测试

```sh
PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  /Users/yujiatang/.codex/worktrees/finder-strict-migration/doku-train/recommend_tf/.venv/bin/python \
  -m recommend_tf.train.tests.benchmark_export \
  --output /tmp/spec027-repeatable-benchmark.json --iterations 100 --candidates 1 32 128
```

macOS 26.5.1 arm64、TF 2.11.0、单推理线程；24 行合成训练 fixture，10 次预热，各 batch 100 次测量：

| 候选/batch | 候选/秒 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: |
| 1 | 2790 | 0.345 | 0.405 |
| 32 | 74304 | 0.426 | 0.461 |
| 128 | 184106 | 0.686 | 0.763 |

整个进程（含训练和导出）峰值 RSS 541,261,824 bytes。只测合成小模型的 SavedModel session 调用，不含原始请求预处理、网络、SageMaker 或线上真实模型，不是生产吞吐/SLA，也不是 30 天容量验证。

## 4. 未执行与限制

- 未下载/导出用户指定真实实验的 epoch 21；远端完整性需后续核验。
- 未提交云训练、上传模型、创建/更新 Endpoint、修改 IAM/调度或 PROD。
- 未交付独立 serving 仓库的实现或镜像；当前包和签名需要其显式适配。
- 未实施 iac 上游 ready 接线、告警资源或消费方改造；本地 AWS 模拟不能代替真实资源验收。
- 没有自动历史清理，保留全部发布引用；保留期与外部清理联动待确认。
- 未知 API 结果保留发布认领并要求对账，不保证无人介入即可从所有 AWS 控制面模糊状态恢复。

## 5. 实施复审与最终全量验收（2026-10-30）

本次复审修复输出目录竞争时误写其他 run 状态、内存预算启动失败资源泄漏、导出配置与拟合状态身份校验、导出器升级版本碰撞、同逻辑模型不同运行元数据的制品地址冲突、恢复预算改变 operation 身份、输入失败占用 Endpoint 发布槽位、发布/回滚期间外部 Endpoint 配置变更，以及平台自动回滚失败时提前释放占用的问题。训练归档的 validation 截止日还必须与发布 cutoff 一致。

最终重新执行 `UV_CACHE_DIR=/tmp/doku-tf-uv-cache make check`，**退出码 0**：

- 根/TF 锁检查、格式、lint、mypy 全部通过。
- 主仓 **318 passed, 3 skipped**，169.59 秒。
- TF 全量 **124 passed**，140.31 秒。
- 另执行仓库 TF 环境的 `python -m pytest recommend_tf/train/tests/test_daily_publication.py -q --disable-warnings`，**27 passed**，39.32 秒。
- `git diff --check` 通过；9 份相关文档的 **139 个本地链接**全部存在。

其中离线/日更相同输入的 checkpoint 全变量逐元素比较、SavedModel reload parity、独立进程重复导出、输出目录竞争、缺日失败释放、预算变化恢复、Endpoint 漂移与回滚恢复均由自动化用例覆盖。未改变依赖锁文件，未执行 AWS 云操作。该完整检查结果取代此前因 DNS 分阶段补跑的验收结果。
