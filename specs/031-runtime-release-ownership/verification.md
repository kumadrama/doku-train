# Spec 031 验证记录

日期：2026-10-31。生产变更仅限用户要求先执行的 PR #30；Spec 031 只实现代码与只读验证。

## PR #30 生产验收

- doku-train 源码 823bcbf94b6077392412826301b46e73075b5999。
- 原生 CodeBuild `doku-train:5e3b866e-b819-42e6-a0fe-c21993b68779`：149 项测试通过。
- pipeline sha256:b1cab3463859f71fad2092fd3eebeda4e7f0e8742bca31f96ce666af574deec5；training sha256:2d3b2fa9645760a1937ae5582ef7a2d012cfbdcc05b2f50b074b16375a9ddae4。
- [IaC PR #655](https://github.com/kumadrama/doku-iac/pull/655) 已合并并应用；EventBridge 指向 `doku-prod-tf-rerank-daily:6`，发布后 plan 无变化。
- `doku-prod-tf-rerank` 保持 InService 和原 EndpointConfig；没有手动重训或切换模型。

## Spec 031 本地验收

- train `make check`：锁文件、ruff、mypy、既有训练测试通过（320 passed / 3 skipped），该次 TF 159 passed；随后最终 TF 全套 162 passed，新增发布文件生成、恢复和回滚测试通过。最终 ruff check/format 通过。
- monitoring `GOTOOLCHAIN=go1.26.6 make test PYTHON=... PROMTOOL=/opt/homebrew/bin/promtool` 通过：671 个 Python 用例（1 个已有 expected failure）、35 个 pattern、契约以及 notifier/investigator/release-observer/dsh-agent 检查。另执行包含新 SQS 持久化 ACK 测试的 10 个定向用例通过。首次默认 Go 1.27.1 与仓库 pin 不符，改用要求版本后完成。
- tools `go test ./deploy-task/... ./deploy-api/biz/deployjob/...` 通过；最后的版本审计记录修改后重跑 ops 与 deployjob 通过；补齐受信 Python Lambda runtime/architecture 后全部 deploy-task/deployjob 再次通过，另验证 Python code-only apply 的回读。
- IaC：TF mock 4 passed、monitoring mock 1 passed；独立 policy 模块、TF 环境和 prod 根环境 validate 通过。
- TF 迁移只读 plan：4 create / 2 update / 1 forget，无 delete；forget 是保留现有 task 的无删除脱管。真实 target revision 与 InputTransformer 均未改变。恢复 schedule 默认 DISABLED。没有 apply 此计划。
- ready ZIP 两次构建相同 SHA256 `4d0234781fdc42e32a5c8c1e448226e808d7b4ad1d8c1460b31061750359c66f`；monitoring ZIP 两次相同 `aa12d6debe488aa195e76292b503c78116f3411934b2e75d325429acef2f853d`，两者均独立导入通过。这些是本地测试包，不是生产已发布版本。

## 未执行的生产门槛

没有把 Spec 031 镜像/ZIP 上传、切换 AppConfig、启用恢复调度、更新 Endpoint 或发送测试告警。需要在平台扩展先上线后，按 operations.md 完成发布权限、兼容程序、启动参数、ready 生产者和策略的顺序迁移，再验收一次自然日更。
