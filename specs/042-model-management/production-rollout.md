# 生产部署与接管记录

Source: 2026-11-26 用户明确授权“部署并接管线上”；AWS STS、EventBridge、ECS、SageMaker、S3 只读快照。

## 当前状态

**生产共享管理平台、FYP/Idea worker、监控接线和两组控制器轮询均已部署；DTE 端到端验收通过。Idea 和 FYP LHUC 两个 pilot 均已暂停旧入口并等待重试排空，尚未完成管理权切换。** 线上模型 Endpoint、权重和调用方配置未因接管改变。Idea 最早可在北京时间 2026-11-27 21:06:06、FYP LHUC 在 21:33:05 后继续各自原接管计划；轮询不会自动完成这个接管步骤。官方 base 日更已按用户明确要求停止，不再纳入后续日更接管。

身份核验：`doku-prod-developer`，AWS account `703844615824`，region `us-east-2`。
本地审计快照目录：`/private/tmp/model-management-rollout-20261126/`，包含规则/target、Task Definition、冻结配置、binding、发布 journal。快照不作为可变线上配置的长期权威。

| 模型 | 旧 TD revision | max_epochs | 快照 high_water |
| --- | --- | --- | --- |
| inner_watch_lhuc_m512_e16_t64 | tf-rerank-daily:40 | 15 | 2026-11-24 |
| inner_watch_lr003 | tf-rerank-daily:32 | 30 | 2026-11-24 |
| inner_watch_w02_lr003 | tf-rerank-daily:33 | 30 | 2026-11-24 |
| tf_rerank_ln_lr002_3feat_us_or_unknown | tf-rerank-daily:30 | 30 | 2026-11-24 |
| tf_rerank_ln_lr002_3feat | tf-rerank-daily:34 | 30 | 2026-11-24 |
| idea_mapped_rm32_drop10 | idea-mapped-daily:9 | 20 | 2026-11-25 |

快照时 FYP 任务族有 5 个运行中的 ECS task；不能停止或将它们记为已排空。Idea 无运行中 task，但同时存在 scheduled 和 published 两个入口，必须共同排空。

## 真实资源兼容修正

- FYP 各存量模型使用不同 inference role，official Endpoint 名称也不属于新 `doku-prod-fyp-` 前缀。新增 `adoption_services` 精确登记 Endpoint→role；只接管已核验旧资源，不扩大新服务命名空间。
- Idea 旧 binding 没有顶层 `model_name`，已有 `monitoring.model_name` 与训练 config 一致。接管接受这个既有身份字段，仍拒绝身份不一致或完全缺失。
- Idea 两个旧 target 没有显式 RetryPolicy。IaC 声明 AWS 现有默认值：86400 秒、185 次；不缩短排空时间。Source: [AWS EventBridge retry policy](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-rule-retry-policy.html)。
- FYP 五个 worker target 已显式配置 86400 秒、3 次。共享 upstream/recovery rule 只能移除相应模型 target，保留其他生产者及消费者。

## 发布阻塞（首次尝试）

本机 GitHub 网络不可用，已观察：

- SSH 22：`Connection closed ... port 22`。
- HTTPS fetch：`LibreSSL SSL_connect: SSL_ERROR_SYSCALL`。
- GitHub 官方 SSH 443：`Connection closed ... port 443`。
- GitHub CLI：`Get https://api.github.com/...: EOF`。
- 规范 `dkd deploy plan iac envs/prod --env prod --remote-path envs/prod/model_monitoring --repo kumadrama/doku-iac` 在解析 main 时失败：`resolve kumadrama/doku-iac branch main ... EOF`。没有 job id，未执行 apply。

GitHub connector 可创建功能分支和 PR；它不能代替 DKD 的构建、计划与部署控制面依赖。不使用临时脚本更新 Lambda 程序或本地 Terraform apply 绕过发布链路。

## 网络重试与部署准备进展

用户要求重试后，GitHub SSH/API 一度恢复。已合并管理 CLI PR #3、监控 PR #438 和制品校验修正 PR #439。训练 PR #80、IaC PR #1143 尚未合并；不能将源码合并或开发镜像构建视为生产部署。

- 开发运行时 CodeBuild `doku-train_tf-pipeline:b2a9b6bc-7957-4c14-8a02-606dd412cbeb` 成功，离线导入与锁定 SDK 检查通过；尚未构建 main 正式镜像。
- 监控基线 DKD plan `dj_249f8185-ace7-452f-bdec-a53a5adf4f73` 成功，无资源变更。
- Lambda plan `dj_6aef5d87-b79b-4fbd-8c4c-52b5fb6df637` 因 S3 缺少服务端 ChecksumSHA256 失败，未执行 apply。原始 ZIP 字节校验通过后，补传具备服务端校验的新版本并更新发布引用。
- FYP/Idea 与隔离 DTE 装配 `terraform validate` 通过；模块 `terraform test` 2 passed；应用输出 binding 合并单测 1 passed。平台调度保持关闭，尚未创建云资源。
- 重试 Lambda plan 时 GitHub HTTPS API 再次持续 EOF，未创建部署任务。SSH ls-remote 与 push 可用，但不能替代 DKD 的 API 依赖。当前等待 API 网络恢复，未执行任何 production apply、Endpoint 更新或旧日更入口切换。

## 恢复后执行顺序

1. 更新远端主线并复核四仓 PR/CI，按 PR 合入；平台代码构建固定 release image 和 controller/resource ZIP。旧线上任务继续使用原冻结运行时。
2. 完成 FYP/Idea 平台环境装配：组专用资源模板、运行时镜像/worker revision、显式 legacy Endpoint/role、read/write prefixes、legacy ingress targets、监控 registry snapshot 来源。模板不能写业务训练配方和调用方路由。
3. 通过 DKD plan 审查共享资源与 collector 接线，只 apply 本次范围。新 controller tick 保持关闭，输出生成 client/code bindings。
4. DTE 合成训练→导出→服务探针→重复触发/配置修改/恢复验收；核验实际镜像、IAM 和日志。新组的监控来源/权限先就绪，再启用 tick。
5. 重新抓取线上快照，等待旧任务完成；按管理 CLI 先 FYP LHUC 与 Idea 各一例 plan/apply，移除精确旧入口后等待全部投递窗口排空。排空中同一 pin 查询/恢复，不提前激活 managed ownership。
6. 验证 pilot 登记状态、配置摘要、原 EndpointConfig/权重及调用方 AppConfig 未变后，逐个接管剩余 FYP。验证下一次 managed 日更的固定 Endpoint 更新及监控身份。
7. 记录实际 deployment/job/registry receipts 和业务探针结果后，才能标记“已部署并接管”。

## Worker release declaration validation

DKD requires three-segment PSM names. Worker release units use `doku-modelworker-smoke`, `doku-modelworker-fyp` and `doku-modelworker-idea`; ECS family and log-group identities remain unchanged. The old four-segment declaration was rejected locally before any deployment job was created.

## Worker image publication boundary

DKD apply requires the release unit's own ECR repository even when plan accepts a shared repository URI. The DTE worker apply was rejected before task definition registration. Each fixed task group now has a small release wrapper that derives from the same immutable shared runtime image and publishes via its standard main CodeBuild flow. This is platform setup per task group, not per-model packaging; new model recipes still require no source/image changes.

## 2026-11-26 DTE 实际验收进展

- 网络根因已定位：系统代理为 127.0.0.1:17891，命令行未继承。显式 HTTPS_PROXY/HTTP_PROXY 及 SSH ProxyCommand 后，GitHub API、Git fetch 和 DKD 恢复；未修改系统代理配置。
- 生产监控 Lambda 已通过 DKD job `dj_37a534c5-77d3-481b-97e6-ee54094354df` 更新，回读 CodeSha256 符合预期、LastUpdateStatus=Successful。
- DTE 41 项资源通过 job `dj_9b1b9f9e-a0e2-4708-9556-4b2c4913dcab` 创建，调度关闭。
- 模型登记实测发现 AppConfig ListDeployments 需要应用 ARN，已补齐精确权限；三项 mock 测试通过，job `dj_96a3a1ba-3ab3-4c3f-8d9b-d2e17ac4cd38` 成功应用单策略更新。
- `managed_smoke` 登记 plan `d07c6c994fe9c926e9e08e1e7667ad57434b98402154e94c742bdc55fadec895` 经正常 10 分钟烘焙后 complete，revision=1，daily_enabled=false。七天纯合成输入已在独立 DTE 前缀上传并通过实际 S3 freeze。
- DTE worker apply job `dj_a149aaa0-27dc-459e-ad62-33913d73dbc9` 在注册 TD 前被镜像仓库校验拒绝；未启动训练。修复 PR #82 已合入，独立 worker 开发构建及完整 CI 通过，DTE 正式构建已成功，镜像 `doku-train/model-worker-smoke:20261126.1-89322d3`，摘要 `sha256:94e42f95f790d7c43ee5f8ff4d7a0c0ec4f0d00cc6fe677c2ce11c03d22387a0`。用户明确确认重试后，job `dj_d98cde8e-842e-4598-874d-62ced8b6383a` 成功，实际 TD 为 `doku-dte-model-worker-smoke:1`。
- DTE 已应用客户端绑定通过生成器写入 doku-iac 的 `envs/dte/bindings/model-management/smoke.json`，PR #1152 已合入。
- 仍需：DTE 单轮训练/导出/版本探针及冻结/恢复验证；之后生产平台装配、FYP/Idea pilot 接管及完整旧重试排空。当前不声明已完成上线或接管。

## 2026-11-26 DTE 首次训练提交与权限修复

- 首次 `train managed_smoke --cutoff 2026-10-07` 已冻结请求，但控制器读取新数据目录下的 release state 被 S3 拒绝，尚未启动 ECS。请求摘要 `a97c7da3f06aa21e5bde2c02cdb61dd9a163657e721c62863eaa68b7c0d08ef7`、配置摘要 `8d9982afa73f2ffbcb3065cb3b473b662e4ab3749a1d7948b74ea73cc2a59d3d` 保留，修复后继续原请求。
- IaC PR #1153 将各任务组自己的 data prefix 纳入 controller read 权限；三组装配回归测试通过。
- 控制面现要求 leaf stack 名称符合 `[a-z0-9-]+`，旧 `model_management` 路径在创建部署任务前被拒绝。DTE/PROD 部署根目录改为 `envs/<env>/model-management`；backend key、资源地址及云资源名称不变。
- 新目录 `terraform validate` 两组通过。DTE plan `dj_71263d4a-8ece-4df4-a106-b39fcc25fa46` 成功：1 add、4 change、1 destroy，仅 IAM、controller 环境接线和按摘要命名的平台 JSON 替换；冻结请求 runtime 位于独立 `groups/`，不会删除。PR #1153 合入后，apply `dj_b48ec79c-1ca5-4867-b211-281e991d3247` 成功。恢复原请求后 ECS task `51f35deee7a344909bade6d20d0376d9` 启动，request pin 未变。

- SageMaker `doku-exp-mm-smoke-22fdfe4c901cd105199b68515b4e6915f8331d74` Completed，云端 resolved-config 为 max_epochs=1、metrics 仅 epoch 1，COMMITTED 存在；模型导出与上传成功。重复 train 返回同一请求与 ECS task，没有新建训练。
- 首次 CreateEndpoint 在发布前被 EndpointConfig IAM 检查拒绝，原 executor 已 STOPPED。IaC PR #1155 补充 group-scoped EndpointConfig 上的 CreateEndpoint/UpdateEndpoint，与既有 tf_daily_pipeline 一致；4 项 Terraform mock 测试通过。不得将成功训练/导出声明为 Endpoint 上线完成。
- DTE 后续配方计划 `b9f837d0510766b91e72b3483d663dd9f97a772398468594aab3f4859b964c83` 仅将两处 max_epochs 从 1 改为 2，affects=future_requests_only；12:01 UTC 开始 AppConfig deployment 3 烘焙。旧请求仍固定原单轮配置。

## DTE 端到端验收通过

- PR #1155 的单策略权限更新通过 DKD `dj_479cb250-9a8f-457f-9b22-94938edb7f10` 成功应用；管理平台专用 CI 已启用并通过。
- 原请求 recover 启动 executor `5ba4e5ebb826413d85385a40a24f0ba0`，继续使用原 runtime/config/request pin 和成功封存训练 run，无重训。
- Endpoint `doku-dte-smoke-managed-smoke-44fac78e` InService；版本 `5a1a91ba5d865d8553f11ada91a9a99f361e4883422629a9f16118164d995ccc`，release journal 和请求均为 deployed，版本/输出探针通过，active 为空。
- AppConfig deployment 3 完成，revision 2、后续 max_epochs=2；原请求发布记录仍为旧配置摘要，训练 metrics 只有 epoch 1。证据断言通过，记录在本地 `dte-e2e-verification.json`。
- 生产平台 main plan `dj_66ef1bce-8b7f-4e7e-be7a-52ccddddaa4c` 通过：74 add、0 change、0 destroy；生产 worker FYP/Idea plan 分别为 `dj_0c16eef4-10a2-439e-b891-1a870b241ccb`、`dj_7c2489f6-e32a-4249-852d-096d2b8821dc`，仅注册 Task Definition，Desired=0。
- 生产平台 apply `dj_19aca97d-dd39-4347-8f43-7561a4f69142` 已创建，尚待结果。调度默认关闭，未接管任何线上模型。

## 生产平台部署与监控接线阻塞

- 生产共享平台 apply `dj_19aca97d-dd39-4347-8f43-7561a4f69142` 成功。FYP worker apply `dj_39d3dc14-6581-4024-b444-7d65168cc017` 与 Idea worker apply `dj_04536a73-bc47-46d4-a0b5-b500700517da` 成功，各为 revision 1，仅注册 TD，无 ECS service。
- 监控接线 apply `dj_95bc9908-f901-4e39-9d0b-7a8e9bb59af7` 失败：完整绑定使 Lambda 环境变量达到 4199 字节，超过 4096 上限。回读函数仍 Active / LastUpdateStatus=Successful，保留此前 2262 字节旧环境。新增 IAM 策略可能已写入，后续必须重新 plan，不能假定完全回滚。
- 已停止该 apply 的重试。修复目标为内容摘要固定的 S3 资源绑定及小型 URI/SHA256 环境变量，保留全部源与 AppConfig 运行开关。先发布兼容 inline 的代码，再切换接线。修复测试和新计划准备后，由用户决定是否重试。
- 原线上 Endpoint、权重、业务路由、旧日更入口均未切换。Idea 显式 RetryPolicy 的计划已通过但尚未 apply。

## 2026-11-26 监控修复重试验收与 pilot 接管

Source: 用户确认监控修复重试并继续接管；DKD 部署回执、Lambda/Events/ECS/SageMaker/S3 实际回读，审计目录同上。

- 监控 PR #440 完成兼容代码发布，DKD `dj_00a3801e-c439-41d2-b0a0-c1876336b0fa` 成功。线上 CodeSha256 为 `U2AKWuz5SUonielkazl0wJ3J9sL8YXDqK2VBtn4HP5k=`，Active / Successful。
- IaC PR #1158 的主线计划 `dj_ea1921e2-a1db-433d-9426-4c3dd23287ac` 为 1 add、1 change、0 destroy；重试 apply `dj_df22e60f-b9ed-4ce2-b6a0-c16e2eafc8f3` 成功。完整 binding 位于按内容摘要固定的 S3 对象；环境变量仅 293 字节，回读原文 SHA256 校验通过。原业务运行开关和全部资源字段均保留。
- 使用 `boto3 Lambda.invoke(FunctionName="doku-prod-model-monitoring", InvocationType="RequestResponse", Payload=b"{}")` 执行实际协调，StatusCode=200、无 FunctionError、Grafana 同步 status=success、dashboard version=151。证据为 `monitoring-pointer-readback.json` 和 `monitoring-pointer-reconciliation.json`。
- 已运行 `scripts/gen-runtime-release-binding.py` 从 applied outputs 重生成发布 binding，与已提交绑定内容完全一致，无新增绑定差异。监控修复本地验证为 769 项 Python 测试（1 项 expected failure）、35 个 promtool 变体、指定 Go 1.26.6 检查及 3 项 Terraform mock 测试通过；对应 PR CI 全部通过。
- Idea 旧 daily/published 两个 target 的默认重试策略已显式固定为 86400 秒、185 次。计划 `dj_92387c39-ad05-44c3-8de1-f47228207463` 为 0 add、2 change、0 destroy；apply `dj_4420c26d-49b2-4850-98af-e500574e31a4` 成功，实际回读一致。
- IaC PR #1160 完整 CI 通过。共享轮询计划 `dj_ff426d85-35ed-4ec9-b609-0a749edb9611` 为 0 add、2 change、0 destroy；apply `dj_5f1e4918-06cb-4aed-9c46-9b522639fdd7` 成功。FYP/Idea reconcile rules 均 ENABLED / rate(5 minutes)。模型登记仍为空，未完成接管的模型不会被新调度准入。
- FYP LHUC pilot 计划 `005a547e9e19af50ee2c1742856a61aef9c2cb6841bfff85a28b9303c68d17e6` 已生成并核验：max_epochs=15、原 Endpoint、daily_enabled=true、first_cutoff=2026-11-26。尚未 apply，因为同旧任务族的官方 base `doku-daily-142861bcc0e54a702bf0ca0ca434d04eb1f48251` 仍 Training（核验时 epoch 25/30），executor `87507e99639442a1854604bafa94ac0b` 仍 RUNNING。未停止该任务，也未关闭 FYP 旧入口。
- Idea pilot 计划 `0f118815f8f941046874881cb4be1c15ae2b0f32dcdba0b70a98f92bac167d91` 保留 max_epochs=20、原 Endpoint、daily_enabled=true、first_cutoff=2026-11-26。首次 apply 回执为 legacy_paused，drain_after=1795784765.878168，即北京时间 2026-11-27 21:06:05.878。daily rule 已 DISABLED；published rule 只移除 `idea-mapped` target。原 Endpoint InService、EndpointConfig 与冻结证据一致。核验证据为 `pilot-handover-verification.json`。

### 待完成事项与继续条件

1. Idea 不得跳过上述完整 24 小时窗口。到期后使用 `idea-adopt-plan.json` 的同一 pin 继续 `manage.py ... apply --plan ...`，pending 时继续同一计划，直至 managed；每次回执使用新本地文件名。controller tick 不会代替这一步。
2. FYP LHUC 已满足旧任务族退出条件并开始排空，使用 `fyp-lhuc-adopt-plan.json` 的同一 pin 在北京时间 2026-11-27 21:33:05 后继续。不得重建计划或缩短窗口绕过 CAS 和旧投递等待。
3. FYP 与 Idea 两个 pilot 均 managed 且实际服务/监控验收通过后，才接管剩余三个启用中的 FYP 模型；官方 base 已按用户要求停止日更，应排除。批量接管和下一次 managed 日更验收尚未执行，不声明全部接管完成。


## 官方 base 日更停止与 FYP LHUC 开始排空

Source: 用户明确要求 kill `tf_rerank_ln_lr002_3feat` 日更流程，并说明线上已没有流量；随后要求核验其他模型影响。此指令授权停止该任务，取代本次操作中不能强停该特定旧任务的限制；不授权停止其他模型。

- 核验 SageMaker `ModelName` 标签及 ECS TD `doku-prod-tf-rerank-daily:34` 后，关闭官方 base 独立 rule，并从两个共享 upstream/recovery rules 中仅移除指向 `doku-prod-tf-ready` 的 target；保留共享 rules。
- `doku-daily-142861bcc0e54a702bf0ca0ca434d04eb1f48251` 已 Stopped；executor `87507e99639442a1854604bafa94ac0b` 在北京时间 2026-11-26 21:32:31.789 已 STOPPED，pipeline exit 137。保存停止回执，不把被终止训练标记为成功，不篡改旧冻结请求或发布 journal。
- 随后核验 FYP LHUC 原 plan 的 state digest 仍完全一致、state.active 为空、旧任务族无运行任务。按此前接管授权继续同一 plan，回执为 legacy_paused，drain_after=1795786384.5876858（北京时间 2026-11-27 21:33:04.588）。其旧独立入口暂停，仅移除 LHUC 对应共享 target。
- 21:34 回读六个存量 Endpoint 均 InService、EndpointConfig 与接管前快照一致。`inner_watch_lr003`、`inner_watch_w02_lr003`、`tf_rerank_ln_lr002_3feat_us_or_unknown` 的独立日更规则及两类共享 target 保持启用；Idea 仍处于原排空阶段。
- 证据：`official-base-stop-before.json`、`official-base-stop-receipt.json`、`fyp-lhuc-adopt-apply-1.json`、`post-stop-model-isolation-verification.json`。官方 base 的服务资源保留；不得因后续接管或旧 IaC 重放重新开启其日更。
