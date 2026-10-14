# Spec 010 验证

Source: 2026-10-14 本地实现验证。测试使用合成样本、file:// S3 替身及回环地址 Redis；未访问 PROD。

## 自动化覆盖

| 范围 | 可执行证据 |
| --- | --- |
| R1/R2/R5 | unit/test_redis_distribution.py 校验 serving 子集、摘要损坏、文件失败不切目录；integration/test_redis_publication.py 用真实 Redis 原字节加载 XGBoost UBJ。 |
| R6/R7/R15 | 重复输入不重训；pending 分发明确失败后独立恢复；拒绝 run 不同步；同名回滚与迟到重试、S3 确认失败、CAS 响应丢失、stage/cancel。 |
| R8/R9/R10 | 两侧 current、双方 pins、替换保护、drain 起点、pin 竞态、Redis 部分删除、S3 部分删除、删除目录提交确认丢失。 |
| R11 | Redis 丢失/损坏时停止切换/删除；显式新 epoch 重建。真实 failover 仍待 iac 验收。 |
| R13/R14 | 环境隔离、危险 TLS 参数/超时拒绝、包和总预算、v2 禁止旧清理/回滚路径、首次同步前清理不写状态。 |
| CLI/调度 | unit/test_scheduled.py；publish_model 和 run redis-sync --help；CI 运行完整 make check。 |

R3/R4 的实际 RPC、Go 加载、实例路由及失联排空属于 mono；本仓只以同版本请求上下文和
两 owner pins 的状态机测试验证协议基础，不能替代服务端集成验收。noeviction/ACL/集群同槽及
容量 RSS 仍需部署环境验证。所有真实删除门禁保持默认 false。

## 执行记录

- `DOKU_TEST_REDIS_URL=redis://127.0.0.1:16389/0 .venv/bin/pytest recommend/train/tests/integration -q`：2 passed；本地 Redis 7.4.11 容器，测试结束已停止。
- `.venv/bin/pytest recommend/train/tests/unit/test_redis_distribution.py -q`：23 passed，含重复输入分发失败后恢复，且显式断言不调用训练。
- `DOKU_TEST_REDIS_URL=redis://127.0.0.1:16389/0 make check`：通过；lock 校验、108 文件格式检查、lint、43 源文件 mypy 均通过，**108 passed in 69.52s**（无 Redis 测试跳过）。
- CI 已配置 Redis service 和 DOKU_TEST_REDIS_URL；这里报告本地证据，未声称 GitHub CI 已运行。

- `./build/build.sh`、`bash -n build/build.sh build/docker-build.sh`：通过。
- `.venv/bin/python -m recommend.train.publish_model --help`、`run redis-sync --help`：退出 0。
- Markdown 相对链接检查：8 个变更/新增文档全部可解析；`git diff --check` 通过。

首次全仓运行在 macOS PyTorch/XGBoost 的 OpenMP 等待中停滞（本地 sample 证据），已终止该次
运行。新增真实 UBJ 加载测试改为独立子进程，避免向后续 PyTorch 单进程测试引入另一原生运行时；
不修改产品训练行为，不用首次未完成运行冒充通过。

## 2026-10-14 追加 Review

本轮按版本一致性、失败恢复及误删风险检查，先复现再修复：

1. **删除饥饿**：每次先 drain 且只执行一批，持续日更会让已到期 delete 永久等待。
   改为 delete 优先，仍遵守双 current、pin、替换和 drain 保护；三版本连续日更测试覆盖。
2. **pending 与 Redis 丢失互锁**：普通 resume 只接受原目录，rebuild 又拒绝 pending，导致无法恢复。
   新增 `redis-resume --rebuild-redis --confirmed-stopped`，持久保留原操作目标、新增 epoch/generation，
   从 S3 恢复全部目标引用后完成原操作；覆盖上传失败、Redis 提交后确认失败、回滚、排空和 shadow。
3. **删除中目录回退**：原实现只在开始核对 Redis，随后目录回退仍继续删除 S3。测试确认原代码
   删除了本应暂停处理的 S3 文件。现在每项删除前核对目录及引用，Redis 删除结束后再次检查；
   发现变化立刻停止，恢复删除前先提交不引用待删版本的安全目录。另测跨版本键清单拒绝。
4. **损坏目录无法恢复**：持久操作此前以 UTF-8 保存原目录，非 UTF-8 损坏值会让重建报解码错误。
   改为 base64 保留精确比较字节，普通操作仍 fail closed；显式恢复测试覆盖二进制损坏目录。

控制写入和删除仍要求旧执行者及其未决请求已停止，不能靠超时推断。S3/Redis 不具备跨存储
事务：两服务必须 pin-before-load、按精确摘要校验，并遵守 draining/deleting，不能仅相信
恢复备份中的 Redis ready。上述检查不能替代 mono 协议或 iac 的 writer 隔离。

最终验证（本轮刚运行）：

- `DOKU_TEST_REDIS_URL=redis://127.0.0.1:16389/0 make check`：**120 passed in 69.58s**；
  lock、格式、lint、43 源文件类型检查全部通过，含 3 项真实 Redis 集成测试，没有 skip。
- `.venv/bin/pytest recommend/train/tests/unit/test_redis_distribution.py -q`：34 passed。
- `./build/build.sh`、`bash -n build/build.sh build/docker-build.sh`、`run redis-resume --help`：通过。
- 8 个变更/新增文档相对链接检查、`git diff --check`：通过。
- 仅本机 Redis 7.4.11 / file:// 测试，没有执行线上 Redis/S3 操作或更改部署/门禁。

## PROD Secret 连接适配

`DOKU_TRAIN_REDIS_SECRET_ID` 运行时读取平台 JSON，和 URL 二选一；强制 TLS、db=0、
密码 URL 编码，错误不包含 Secret 内容。新增 doku-redis-model store 身份供 dkd setup。
2026-10-14：带本地 Redis 的 `make check` **122 passed in 71.24s**，格式/lint/mypy/lock 全部通过。
本次测试未读取生产凭据；云端资源、ACL 与任务绑定由 iac 后续接线验收。
