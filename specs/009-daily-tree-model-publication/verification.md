# Spec 009 验证记录

Source: 2026-10-13 本次实现后实际运行；仅本机合成数据和 SDK stub，不代表生产验证。

## 已执行命令

```sh
UV_CACHE_DIR=/tmp/doku-train-uv-cache UV_OFFLINE=1 make check
```

结果：exit 0；锁文件解析 60 packages；94 files 格式检查通过；ruff 通过；mypy 38 个源码文件
通过；65 tests passed，pytest 耗时 60.56 秒。测试包含既有 CPU DNN/DeepFM/XGBoost smoke。
缓存路径显式放到可写临时目录，避免默认 uv 缓存权限问题；未修改依赖或锁文件。

```sh
UV_CACHE_DIR=/tmp/doku-train-uv-cache uv run --offline pytest \
  recommend/train/tests/unit/test_publication.py \
  recommend/train/tests/smoke/test_tree_publication.py -q
```

专项结果：10 passed，5.99 秒（最终 make check 已再次覆盖这组测试及之后的改动）。

## 证据对应

| 需求 | 自动化证据 |
| --- | --- |
| 字段迁移与原训练回归 | test_declarative.py、原三模型 smoke 和 schema 验证 |
| 缺日/30 自然日/至少三段 | test_publication.py::test_window_has_gaps_and_natural_day_bound；test_tree_publication.py |
| 回归同序/逆序/平局 | test_concordance_matches_bruteforce：固定种子与小样本穷举比对 |
| 0.60 边界/null 拒绝 | test_gate_null_boundary_and_regression；端到端 rejected 保留训练成功 |
| 原生导出与完整发布 | test_daily_export_publish_and_repeat：真实 CPU XGBoost、独立进程、file CAS、包归档 |
| checkpoint 不变、评估重跑 | 同上；重复执行 test 并比对 checkpoint bytes |
| 状态损坏检测 | 同上；篡改 feature state 后拒绝导出 |
| 去重/同名锁/CAS | test_registry_duplicate_and_stale_cas；端到端 duplicate_input |
| pin/回滚/current 保护/30 天与24小时 | test_pins_drain_rollback_and_deletion |
| 注册与清理竞争 | test_concurrent_cas_recomputes_after_pin |
| 删除失败可恢复、禁止半删回滚 | test_delete_failure_keeps_fence_and_retries |
| S3条件写与错误分类 | test_s3_conditional_write_and_missing_are_not_permissions：botocore Stubber，未调用 AWS |

## 限制与跨仓接入

- 未执行 AWS 上传、生产日更、实际删除或故障演练；S3 SDK 合同测试不能替代目标环境验证。
- 未创建 iac 后继触发或 IAM；未修改 mono RPC。消费者未实现 pin 协议时清理保持 dry-run。
- Go/C API 与 Python 预测一致性、多个预测实例的就绪协调由 mono 后续集成验证；本次只是
  Python 原生库导出重载的合成输入 parity，不是 Go 验收。
- 失败/拒绝产物登记归档；未知未登记对象保守保留，不靠目录年龄推断可删。共享冻结输入
  原件和上游数据不在模型清理的删除范围；本地清理只跟随远端 deleted 墓碑。
- 本轮最初发现原生库混用导致测试进程崩溃，已将导出也放到独立进程；最终完整检查通过。
- 日常阈值应按已确认方案在完整 YAML 填 0.60；端到端成功路径测试显式用 0.0，验证发布流程，
  不能将合成模型的通过解释为真实业务效果。

文档另外执行 git diff --check 与本地链接检查；未提交生成模型、真实数据、缓存或凭据。

## 第二轮审查：故障注入与竞态修复

Source: 2026-10-13 用户要求复审后修复；全程 file 存储或 SDK stub，无云端写入。

新增验证：

- 成功写入后丢失响应：CAS revision 与不可变内容读回确认；覆盖超时和 SDK 重试后 412。
- 日更上传在训练归档处中断：current 不变、manifest 不提前出现；显式原 run 目录恢复完整发布。
- 同一天更旧模型包以及显式回滚后的迟到发布：拒绝覆盖。
- 两个版本共享 run：删除旧版本不删除仍被当前版本引用的本地 checkpoint；退休后拒绝新引用。
- 活跃本地读者：清理不删除也不提前退休，读者退出后才可删除。
- 两个清理进程交错完成同一版本：删除完成状态幂等。
- reserved manifest 路径：登记前拒绝。

专项命令：

```sh
UV_CACHE_DIR=/tmp/doku-train-uv-cache uv run --offline pytest \
  recommend/train/tests/unit/test_publication.py \
  recommend/train/tests/unit/test_publication_recovery.py -q
```

结果：15 passed，0.93 秒。完整检查结果见本节后续记录。

第二轮最终完整命令：`UV_CACHE_DIR=/tmp/doku-train-uv-cache UV_OFFLINE=1 make check`。
结果 exit 0：锁文件、97 文件格式检查、ruff、mypy 39 源码文件均通过；
**73 tests passed，61.93 秒**。另运行 `publish --help` 确认 --run-dir 参数可用；
`git diff --check` 通过，49 个本地文档链接有效。

## 2026-10-13 PROD 编排接入验证

- `UV_CACHE_DIR=/tmp/doku-train-uv-cache UV_OFFLINE=1 make check`：exit 0，99 文件格式、
  Ruff、mypy 40 源文件通过；**81 passed，62.07 秒**。
- 新增 8 项容器边界测试：SSE-KMS 头保留 IfMatch/IfNoneMatch、配置 SHA256/model identity、
  原 cutoff 传递且不 force、门禁拒绝作为正常完成、异常日志、清理消费者 dry-run、无 Git 的
  镜像 lineage，以及 S3 GET 403 下的存在/不存在区分。已有拒绝权限测试补覆盖 LIST 也被拒绝。
- `./build/build.sh` 和 `bash -n build/build.sh build/docker-build.sh` 通过；从 output 源码包
  （在 /tmp 启动，确认实际导入 output 路径）执行 source_identity 成功，未读取仓库 .git。
- JSONata 2.0.6 原调度日期/跨日重试与容器退出码表达式测试通过。
- 本地迁移配置通过 validate_config；SHA256 为
  `9a392ef411c54acfbae151d674dbe4fe7b75a7a7cd63eb07e6ed50ddd9f2b6a1`，与 iac PROD 配置一致。
- iac `terraform -chdir=modules/ecs_scheduled_task test -no-color`：3 passed；覆盖旧调用方、
  训练位于上游重试范围之外、静态清理命令无重试；PROD 和 DTE `terraform validate` 均通过。
- iac `test-tree-publication-contract.py`、`test-fyp-rank-samples-contract.py`、
  `test-scheduler-time-input.py` 通过；修改文件 terraform fmt 检查、两仓 git diff --check 通过。

未执行 Docker 镜像构建/发布、配置上传、Terraform apply、binding 生成或生产调度验收。
首次发布须先以 tree_publication_enabled=false 建立资源，注册 TD 并上传配置后再启用。
AWS provider 的 mock 测试使用本机已有 provider（沙箱外本地进程通信），没有连接 AWS。

## 接入 Review 最终验证（2026-10-13）

修复 dkd 两段式 PSM 无法发布、首次 apply 默认提前启用、关闭开关改变清理工作流结构、
训练角色列举其他模型和同步 Scheduler 保留无用 ECS 权限的问题。
新增部署合同及首次训练前清理回归测试，清理不创建空发布清单。

- `UV_CACHE_DIR=/tmp/doku-train-uv-cache UV_OFFLINE=1 make check`：exit 0，99 文件格式、
  Ruff、mypy 40 源文件均通过；**83 passed，62.07 秒**。
- `./build/build.sh`、`bash -n build/build.sh build/docker-build.sh`、`git diff --check` 通过。
- iac 从最新 main 的独立工作树验证：mock Terraform **4 passed**；PROD/DTE validate、
  fmt、接线合同、上游合同、Scheduler 输入与 JSONata 2.0.6 表达式测试全部通过。
- Docker 镜像构建、真实 IAM 调用、S3 上传、apply 和生产调度均未执行；上述为本地证据。
