# Spec 028 验证记录

Source: 本次本地实际执行。云端验证未执行；不声明真实指定实验或 Endpoint 已上线。

## 已执行

- 本地新增审核/执行者/推理专项首次运行：7 passed；随后补充 weighted history 默认值、去重排序及异常用例，纳入最终全量检查。
- 本机 Docker（arm64 主机运行 linux/amd64 CPU 镜像），官方 `tensorflow/serving:2.11.1` 摘要 `sha256:ceb17c529b59a3f0ffa3ef87404e0f2cf0e155cdec4a723ed49b6be487e4c728`：成功构建薄适配镜像并加载合成 SavedModel。
- `python -m recommend_tf.inference.tests.container_smoke verify --root /tmp/spec028-container-fixture`：实际 HTTP /ping 通过；3 行批量四任务输出与本地 SavedModel 在 rtol=1e-6、atol=1e-7 下相等；实际模型版本一致；非法请求 HTTP 400。
- doku-iac `modules/sagemaker_inference_access`：terraform validate 通过，mock test 1 passed。
- doku-iac `modules/tf_daily_pipeline`：terraform validate 通过，mock test 2 passed，覆盖默认禁用与可变镜像拒绝。
- `python3 doku-iac/tests/test_tf_daily_binding.py`：2 tests passed，覆盖身份装配、输入不变、环境和预算错误拒绝。

首次直接使用仓库根作为 Docker 上下文被根 .dockerignore 排除源码；已改为独立 build_context 工具并重新构建/验收成功。沙箱不允许本机 Docker socket、localhost 和 Terraform provider 进程时，使用授权的本地验证执行；未执行云资源操作。

## 验证范围

- 真实训练来源核验工具通过合成 AWS 响应测试；尚未下载指定真实实验。CPU exporter 的模型/词表/配置交叉校验复用已有实现。
- 推理数值用例同时与训练侧独立张量准备结果比较；容器测试再验证 TensorFlow Serving 而非仅 Python SavedModel。
- 生产 inference 不导入 train；测试 fixture 为唯一训练依赖。
- CI 新增独立 tf-inference-container job，首次 CI 结果需 PR 运行后确认。
- IaC 测试是 mock provider；没有 environment 实例或远端 state 变更。上游真实信号、Endpoint 告警、数据截止日新鲜度监控与容量验收尚未装配。

## 全量与镜像补充

`UV_CACHE_DIR=/tmp/doku-tf-uv-cache make check` 初次完整退出码 0：主仓 318 passed, 3 skipped（175.82 秒），TF 132 passed（178.07 秒）；格式、lint、类型检查与锁检查通过。随后为干净构建上下文增加两项测试，专项 test_rollout 为 7 passed（1.71 秒）。

编排镜像基于 Python CPU 基础镜像 `sha256:6ff506466f8b1e981719b468cc0023b8ef000b983cdec0eafe119d01f379e31d` 构建成功，依赖来自未修改的 uv.lock。ARM Mac 的 amd64 QEMU 运行检查在导入 PyArrow 时触发 signal 11，限制 SIMD 探测也未解决；未据此声明运行通过。CI 已增加原生 Ubuntu amd64 的编排镜像 CLI/exporter 导入检查，尚待 CI 执行。

最后一次共享工作区 tf-check 为 2 failed, 132 passed，均为恢复时 checkpoint identity mismatch；此期间 Spec 029 正在修改 training_pipeline 等文件（修改时间与失败窗口一致），源码身份发生变化。没有修改该并行工作或放宽身份验证，改用 HEAD 基线加显式 Spec 028 文件的独立 /tmp 快照复测。最终快照结果在下方记录。

生产 inference 导入边界与两份 JSON 合同检查通过。相关 9 份文档 136 个本地链接检查通过。新增 IaC 两模块最终 validate/mock 测试仍为 1+2 passed。


### 最终冻结快照结果

在 `/tmp/spec028-validation` 中，以 Spec 027 基线加显式列出的 Spec 028 文件构造隔离快照，从该目录执行：

```sh
PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  /Users/yujiatang/doku/doku-train/recommend_tf/.venv/bin/python \
  -m pytest recommend_tf/train/tests -q --disable-warnings
```

**134 passed，145 warnings，123.95 秒，退出码 0。** 首次快照尝试仍从共享工作区启动，导出子进程继承该 cwd，导致一项跨进程一致性失败；最终命令同时固定 cwd 与 PYTHONPATH，全部通过。未放宽源码/Checkpoint 身份校验。

最终代码范围的 ruff 与 git diff --check 通过；生产 inference 的 AST 导入边界检查通过。原生 amd64 编排镜像运行仍待新增 CI job 执行，不能将“镜像构建成功”写成“运行验收通过”。本次创建的本地诊断与推理容器已清理，未停止用户其他容器。

## 2026-10-30 云端执行中

- 原生 CodeBuild `doku-train:f75ef755-1c4a-4734-b9b7-8a645d403c33` SUCCEEDED：固定摘要的 pipeline CLI、exporter 导入和 TensorFlow Serving 版本检查通过。
- 历史实验实际 epoch 21 最优 checkpoint 已审计并通过导出 parity，版本 `4636811a0f4d230350f42a2165620e0c120225e460c993fd109c50cdc696f909`。本地真实模型容器合成批量请求和错误请求通过。
- 28+1 实际冻结输入通过：2026-10-01..28 train、2026-10-29 validation，29 个成功发布日；ready 字节摘要 `b1224663b2b65c6cb0089170546091ec826d860643ec152d258e507f2a8916d7`。只读取发布元数据，没有将真实用户样本写入仓库。
- 首次 ECS executor 成功导出和上传模型，但 SageMaker 拒绝 OCI manifest。转换为 Docker V2，保留原镜像 config/layer 摘要；inference digest `5009a4ef2391fc60af8545c5c5e93070b31d450baf46dfed56954d9fa69302c3`，pipeline digest `231be2b36090c4da58d21e09add685f5f7793c7950039596bf355467ae925eaf`。
- 确认失败 executor STOPPED、Endpoint 尚不存在后，使用 ReleaseState 条件写将 OCI 尝试归档为 publication_failed；修正 binding 开启新尝试，未改 high_water。第二次运行的 Linux 导出仍得到同一模型版本。
- 第二次在 CreateEndpoint 被 IAM 拦截，缺少对 endpoint-config 资源的 CreateEndpoint 权限。正修复限定 tf-* EndpointConfig 的 CreateEndpoint/UpdateEndpoint 权限，保留部署状态以原 operation 恢复。
- `python -m pytest recommend_tf/train/tests/test_ready_event.py -q`：2 passed，覆盖 upstream 成功身份、小时任务忽略、29 天窗口和 Manifest 摘要异常。

以上不代表 Endpoint / 日更 / 回滚已验收完成；后续记录实际结果。

恢复修复验证：隔离快照运行 `python -m pytest recommend_tf/train/tests/test_daily_publication.py recommend_tf/train/tests/test_ready_event.py --disable-warnings -q`，31 passed（11.53 秒）。新增明确 AccessDenied/ValidationException 拒绝后允许重试，原网络超时不盲目重试测试保持通过。恢复镜像固定 `sha256:593bd256ceee0ac9b3eab3f0119ed6abb65cbb953d216760f2fb6c36d79f7108`；仅替换 publisher，训练源码与锁摘要不变。

首次 Endpoint 已于实际执行中进入 Creating；等待 InService 与远端数值验收。自动触发两个规则保持 DISABLED。

## 首次 Endpoint 验收通过

`AWS_PROFILE=doku-prod-developer recommend_tf/.venv/bin/python /tmp/spec028-cloud-probe.py` 退出码 0。

- Endpoint `doku-prod-tf-rerank`，us-east-2，InService，单台 ml.m5.large。
- 模型版本 `4636811a0f4d230350f42a2165620e0c120225e460c993fd109c50cdc696f909`，与指定实验 epoch 21 对应。
- 3 行合成输入，四任务名称、行数、模型版本完全一致；远端与本地 SavedModel 预测 `rtol=1e-6, atol=1e-7` 通过。
- 批量 1/32/128，分别预热 2 次和测量 10 次，本地客户端含公网延迟 p50 为 305/508/1335ms，串行候选吞吐约 3.1/61.7/100.1 每秒。这不是生产容量 SLA，容量仍待调用方确认。
- 实际证据：忽略目录 `runs/spec028-rollout-20261030/endpoint-acceptance.json`、`endpoint-response.json`。合成请求 `request.json`；expected.json 为本地模型预期。
- 调用授权策略已创建：`arn:aws:iam::703844615824:policy/doku-prod-role-tf-rerank-inference-invoke`，尚未自动绑定服务端角色。

28+1 日更执行已提交，尚未以“提交成功”替代完整验收。自动触发仍关闭。

日更首次执行在训练创建前遇到 SageMaker `DescribeTrainingJob` 的 `ValidationException: Requested resource not found.`。已补充精确识别该资源不存在消息，不吞掉其他 ValidationException，并通过回归测试（4 passed）。修复镜像 `sha256:13c8f56c5570f99136d3a28be497d4167ff986e0a8def9c07dbb75da1386bda6`，ECS definition revision 4；仍使用已冻结的同一 operation/请求接续，无重新解析日期或扩大窗口。

## 28+1 日更实际训练已开始

任务 `doku-daily-3328789f5e6a355d3b6158477afe8f0178a6370f`，SageMaker 状态 InProgress / Training，CloudWatch 已记录 `prepare_split: train`。云端训练复用已冻结的 operation `3328789f5e6a355d3b6158477afe8f0178a6370fcc0449a570e360b7cc9d8095`；在线仍为 epoch 21 历史模型，数据 high_water 为 2026-10-24，尚未提前改成新数据日期。

最终专项验证：隔离快照 `pytest test_daily_publication.py test_ready_event.py test_rollout.py test_inference_contract.py -q --disable-warnings`，42 passed（14.24 秒）；IaC `terraform -chdir=modules/tf_daily_pipeline test -no-color`，2 passed；rollout stack `terraform validate` 通过，`ruff check`、`git diff --check` 通过。

剩余一次性验收使用忽略目录中的 `finish_acceptance.py`：等待日更 ECS 成功退出且新版本实际可调用 → 重复执行验证无新增训练/发布历史 → 回滚到保留旧版本并校验数值 → 恢复新版本 → 检查 high_water 不回退 → 仅允许两个既有 EventBridge rule 与发布心跳 alarm 的 update plan → 开启自动触发。任一步失败停止，不提前开启；最终结果写入 `remaining-acceptance.json`。该进程在本机执行，不是生产日更调度器；生产任务由上游成功事件触发云端 ECS。

实际编排中断恢复验收通过：SageMaker 训练进行中时停止旧 executor `1f438095113e462baff4ff57afe6dd32`，AWS 确认 STOPPED 后启动新 executor `73a04996b7f3421e8b73ad83c9c9fe90`。发布状态中的 executor 已切换为新 ARN，stage 仍为 training，TrainingJobName 仍是 `doku-daily-3328789f5e6a355d3b6158477afe8f0178a6370f`。训练未停止或重复创建。证据 `executor-recovery-drill.json`。

一次性验收接续进程已启动，日志 `runs/spec028-rollout-20261030/remaining-acceptance.log`。当前训练进行中，不能声明日更 Endpoint 更新/回滚/自动启用已完成；最终以 `remaining-acceptance.json.status` 为准。

上游适配器读取真实 29 天发布元数据的验收通过：`ready_event.build_ready` 生成的 ready 与本次训练已冻结 ready 逐字节相同，摘要仍为 `b1224663b2b65c6cb0089170546091ec826d860643ec152d258e507f2a8916d7`，证据 `ready-adapter-acceptance.json`。本地真实模型验收临时容器已删除，临时 Docker ECR 登录已退出，不影响云端 Endpoint/训练。

## 2026-10-31 复用完成训练的模型重新发布

用户明确授权补齐推理角色权限、复用完成训练产物，不重训；完成更新、回滚和恢复验收后启用自动触发。

- 原训练实际于北京时间 2026-10-31 04:14:24 Completed，20264 秒，导出模型版本 `410f512b49800352e265deb24f3563b8ff7aa521e1af59833be59d6d6301fcc3`。
- 首次日更部署已回滚；AWS Endpoint FailureReason 明确指出推理角色缺少 `cloudwatch:DescribeAlarms`，不是训练失败或模型指标问题。
- IaC 推理角色模块新增只读 DescribeAlarms 权限及回归断言，mock test 1 passed。审核 plan 只有 `module.inference["tf-rerank"].aws_iam_role_policy.load` 的 update，已 apply。
- 使用既有 `--bootstrap-run` 入口消费原日更 run 的 COMMITTED 清单，跳过训练，仅导出/发布；新 executor `6b4f2e611dfe454d81c5a446928b8f05`。复用原 SageMaker job `doku-daily-3328789f5e6a355d3b6158477afe8f0178a6370f`。
- 接续验收日志 `runs/spec028-rollout-20261030/remaining-acceptance-20261031.log`，等待实际结果后记录最终状态。

重新发布实际成功：Endpoint InService，新模型版本 `410f512b49800352e265deb24f3563b8ff7aa521e1af59833be59d6d6301fcc3`，四任务/3 行合成请求验证通过。重新导出的 manifest SHA256 仍为 `75590df3dd00f657d282a07f24385bd3b897f79a34c2e4a8d4e4fd5c7354f1de`，与原训练产物完全一致。`list-training-jobs --name-contains doku-daily-` 只有 2026-10-30 22:35:58 创建的原任务，未新建训练。

实际重复执行验收通过：成功发布后再次运行同一 operation，history 长度、online 记录与 high_water 均不变，无新增训练。

实际回滚验收通过：恢复保留版本 `4636811a0f4d230350f42a2165620e0c120225e460c993fd109c50cdc696f909`，Endpoint InService；3 行四任务输出与原参考值满足 rtol=1e-6、atol=1e-7；high_water 仍是 2026-10-29。已发起恢复新版本的最后一次切换，自动触发继续等待恢复验收。

## 2026-10-31 最终验收：通过

`remaining-acceptance.json.status=passed`，实际完成以下链路：

1. 不重训，复用已完成 run 发布版本 `410f512b49800352e265deb24f3563b8ff7aa521e1af59833be59d6d6301fcc3`。
2. 相同发布 operation 重复执行成功，发布历史及 online/high_water 不变。
3. 切回保留旧版本，实际调用与旧参考数值一致。
4. 恢复新版本，Endpoint InService，实际调用四任务/版本/行数和有限数值检查通过；high_water 仍为 2026-10-29。
5. Terraform enable plan 严格只允许既有 `upstream_success`、`daily.ready` 两个 EventBridge rule 和 `daily.freshness` 告警 update，apply 成功；AWS 查询两个规则均 ENABLED。

当前 Endpoint `doku-prod-tf-rerank`，us-east-2，最新已发布数据截止日 2026-10-29。首次由上游自然事件触发的下一次日更尚未发生，本次确认的是触发接线与开关启用，不能将其写成已跑完下一天训练。

可执行证据：`runs/spec028-rollout-20261030/remaining-acceptance.json`、`remaining-acceptance-20261031.log`、`enable-auto-plan.log`、`enable-auto-apply.log`、`final-acceptance-20261031.json`（环境运行文件按规则忽略，不提交模型或真实数据）。本次 IAM 模块 mock test 1 passed，`terraform fmt -check` 与 `git diff --check` 通过。
