# 验证记录

Source: 2026-11-20 告警现场只读查询和本次本地实施。

现场 `logs filter-log-events` 确认 CreateTrainingJob ValidationException；`ecs describe-tasks` 确认失败 executor STOPPED/exit 1；`sagemaker describe-training-job` 返回 Requested resource not found。读取 journal 确认 active cutoff 2026-11-19/stage training；Endpoint InService，与 journal 的 cutoff 2026-11-18 online 配置一致。

本地与生产验证结果在实际执行后追加，不以实现完成代表生产恢复。

## 本地定向验证

在独立分支 `codex/tf-monitoring-context-file` 执行（使用主工作区已锁定的 Python 3.10 环境，PYTHONPATH 指向本分支）：

```sh
PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  /Users/yujiatang/doku/doku-train/recommend_tf/.venv/bin/python -m pytest \
  recommend_tf/train/tests/test_monitoring_context.py \
  recommend_tf/train/tests/test_daily_publication.py -q --disable-warnings
```

结果：46 passed，17.79s。覆盖完整大 JSON 往返、摘要篡改、协议冲突、512/513 长度边界、只读计划、原请求留档、冻结输入不变、已有作业拒绝、旧执行器仍运行/未知、权限错误、配置摘要损坏以及 journal CAS 竞争。

## 恢复操作

先构建并验证支持文件协议的训练镜像与编排镜像，取得不可变镜像 digest 和镜像内真实训练源码 SHA256。不得把工作区摘要误当作旧基底镜像打补丁后的摘要；生产现有训练摘要与当前仓库 HEAD 摘要不同。

```sh
AWS_PROFILE=EXPLICIT_PROFILE python -m recommend_tf.platforms.sagemaker.repair_monitoring_context \
  --binding /path/to/original-binding.json \
  --operation ORIGINAL_OPERATION \
  --training-image REGISTRY/IMAGE@sha256:VERIFIED_DIGEST \
  --source-sha256 VERIFIED_TRAIN_SOURCE_SHA256
```

输出原请求与候选请求。核对只包含监控传递字段、训练镜像及源码摘要变化后，同命令加 `--apply`。工具只修复 journal，不创建训练任务。随后通过原 deferred envelope 恢复：它会读取原绑定并计算原 operation，已修复 journal 提供实际训练请求。新日期使用新绑定/训练镜像并同时升级 worker；保留原 deferred envelope 的旧绑定，不改变原 operation。

当前未执行生产构建、部署、journal 写入或训练启动；上线验收仍需验证新请求被接受、上下文摘要正确、成功密封监控事件及既有发布流程完成。

## 最终本地检查

独立 worktree 临时将 `recommend_tf/.venv` 链接到主工作区已安装的锁定环境，并使用 `UV_NO_SYNC=1` 复用依赖；验证后移除链接。命令前缀：

```sh
export UV_CACHE_DIR=/tmp/doku-tf-monitoring-uv-cache
export UV_OFFLINE=1 UV_NO_SYNC=1
export UV_PROJECT_ENVIRONMENT=/Users/yujiatang/doku/doku-train/.venv
make check
make tf-check inference-check
python3 idea-mapped-daily/build/package_source.py --pin
make inference-check
```

- 根锁文件与 TF 锁文件检查通过；Ruff 格式与 lint 通过；mypy 70 个文件通过。
- 公共训练测试：329 passed、3 skipped，181.67s。
- 最终 TF 全量：241 passed，227.94s（包含本次 13 项传输/恢复测试）。
- 最终推理合同：29 tests，OK，0.292s。
- `git diff --check`、文档索引链接、恢复 CLI `--help` 均通过。

首轮 TF 测试加载了修正前的校验顺序，重复修复的错误消息断言失败；最终版本独立重跑 TF 全量通过。推理检查发现共享运行时代码变更导致 Idea 本地源码包 pin 过期，已使用仓库打包工具同步 ARTIFACT_KEY/ARTIFACT_SHA，再次完整执行推理合同检查通过。仅同步本地源码 pin，未上传源码包或部署 Idea；原 binding 和角色未改。

本次共 599 项测试通过、3 项跳过。没有声称首轮 `make check` 单次全绿；以上是各检查目标在最终相关版本上的执行证据。

## 发布阻塞（已解除）

自动审批拒绝向已有 origin `github.com/kumadrama/doku-train` 推送 `codex/tf-monitoring-context-file`，理由是授权未明确覆盖向该目的地发布代码。未更换工具绕过拒绝。修复和恢复计划已保留本地，生产构建、部署、journal 修复及训练启动尚未执行，等待目的地发布授权。


## 2026-11-20 生产合并、部署与恢复

Source: 用户明确要求“合并 并部署 恢复本次日更”；AWS 回读、保存的 Terraform 计划和恢复回执。

- 训练仓 [PR #68](https://github.com/kumadrama/doku-train/pull/68) 三项 CI 全部通过，已 squash 合并为 `6e7e2707938f95dec19bb0f28219385d3d61f90d`；IaC [PR #1017](https://github.com/kumadrama/doku-iac/pull/1017) CI 通过，合并为 `12b2de972113a822cf80e7832d0d6e595b35c6c7`。
- CodeBuild `aca0f4be-70b8-4717-8f27-41e20f298a57` 成功。基于原固定镜像仅覆盖本次运行时补丁，旧文件摘要比对通过；CPU TensorFlow 2.11、文件/inline 往返和损坏拒绝检查通过。第一次构建因仓库 Docker 忽略规则排除补丁而失败，使用独立补丁上下文后完成；没有更换训练配方。
- 新训练镜像 digest `2b2ffecbc3b819fde8601c96f8c2e5d66376d9539c558b844f2318dbff01adbe`；镜像内训练源码 SHA256 `e933b4dae165be6ea5f83b996316529a846ca1a862167db8c92e0c980f9817cb`；锁摘要仍为 `b2763acbc6cb48f18bd9326b605e9bf7634a6d3a4ec05355023c8f8c983d0bd5`。
- 新编排镜像 digest `18e2f42f97e32d2e1363a94a79da637be3953f4a7ff2036d0e01ea9e328b3229`；两个镜像均在 `703844615824.dkr.ecr.us-east-2.amazonaws.com/doku-train/doku-reco-train`。
- 新 worker 为 `doku-prod-tf-rerank-daily:22`。新 binding SHA256 `e22fa7c884d6d3e7ad3f908d4b70e6a971f33c75d1319fcd81a67e622f5d0083`，ready ZIP SHA256 `888fff45faaf75727e7b1183fada23ed0118a3741f7673ce7e6d767d9b35a390`。Lambda LastUpdateStatus 为 Successful；EventBridge target、任务镜像和 binding 回读均一致。
- 保存的完整 Terraform plan 仅该模型的 ready Lambda 和 worker target 两项 update，0 add / 0 destroy。apply 成功；随后 `terraform plan -detailed-exitcode` 返回 0，零漂移。
- 实际修复前，自动恢复更换过 executor。工具在最新 executor 仍运行时正确拒绝，确认 STOPPED 后才执行 CAS。原请求保存在 `monitoring_context_repair`，只改变监控传输、新训练镜像及源码摘要；operation、job 名、冻结输入、配置摘要、run URI 和在线高水位不变。
- 通过原 deferred envelope 恢复，EventBridge event ID `c9cd2b90-44fe-5dd0-b54b-bff8eae5c052`。新 ECS executor `6c916026787b41558caa2ab617adfc00` 使用 revision 22。
- 原 job `doku-daily-565d9f282f8f3aedf4b34e015a911c141aeb1438` 于 **2026-11-20 16:05:55 UTC+8** 启动，实测 InProgress / Training；请求与修复后的 journal 对账一致。上下文 SHA256 `4f544de223b23b99f1ea06189d69c241c0404c38b4ce9531b80dea0bd7e90fc1`，内容与原 inline JSON 完全相同；最长环境变量值为 192 字符。
- 训练日志流 `doku-daily-565d9f282f8f3aedf4b34e015a911c141aeb1438/algo-1-1795161954` 已记录 `prepare_split: train`，证明通过训练入口中的源码、配置及上下文摘要校验。

当前验收边界：已合并、部署并恢复原日更任务。训练、导出和新模型发布尚未完成，线上 Endpoint 仍 InService，保持 cutoff 2026-11-18 成功版本。未重新生成样本、未创建 retry-id、未重训已有成功任务，也未人工清除/ACK 告警。前一轮同模型训练耗时 16790 秒，不作为本轮耗时 SLA。

可执行证据和回执保留在本地 `runs/tf-context-repair-20261120/`：`deployment-verified.json`、`repair-plan.json`、`repair-applied.json`、`resume-event.json`、`training-accepted.json`、`status.py`、`status.json`、`plan.json`、`apply.log` 和 `post-plan.log`。可重跑 `status.py` 查看真实终态，不把启动成功声明为发布完成。
