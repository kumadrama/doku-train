# 验证

2026-11-17，本分支仅准备代码和测试，没有发布 worker、AppConfig 或 Endpoint。

- 只读确认 FYP 现有独立业务版本发布材料及已完成回执。提供的 TD17 binding 实际指向 `doku-prod-fyp-inner-watch-w02-lr003`，无业务模型列表/定义接线；不能把日训 Endpoint 更新误称为业务自动切流。
- 完整 inference 测试 20 项通过，日训 runtime-policy/Idea-window 测试 9 项通过；Ruff 通过。
- 新增测试覆盖独立版本命名、禁止更新/回滚已有 Endpoint、原 FYP 默认完成行为、配置模板与旧定义保留、models→list 顺序、重复恢复不重复发布、签名变化拒绝、外部更改保护、自动回滚不触旧路由，以及 CAS 的响应丢失和最后时刻竞争。
- 发布窗口已用现有 Idea TD5 做纯只读 SDK 检查：boto3/botocore 1.43.120、Python3.10.22、LatestDeploymentNumber=true、退出0。因此无需升级基础运行环境。原镜像仅有实验 tag，生产门禁要求从 main 构建：新增现有 Idea worker 的标准发布目录，FROM 原 digest 并复制本次源码；对应专用 CodeBuild/ECR 仅作为发布配置，不新增训练 worker 或调度。
- SDK 官方 wheel SHA256 与 recommend_tf/uv.lock 匹配，botocore 1.43.120 的 StartDeployment shape 含 LatestDeploymentNumber；运行时不允许降级无 CAS。
- 完整 TensorFlow 训练测试需要仓库锁定 Python3.10/TF 环境，后续由现有 CI 执行。此次没有部署新执行服务，没有共包多模型，也没有运行新 observer。

当前 5d9e 独立 Endpoint 稳定化和首次实际 RPC 由发布窗口单独验收。此处不将配置 COMPLETE 当作每个 AI 副本已采纳，也不宣称下一次自然日训跨版本链路已经通过。

## Existing worker publication manifest

`idea-mapped-daily` retains production revision 5's family, task role, resources, logs, and SHA-checked
S3 bootstrap. Its Dockerfile inherits the exact original runtime and copies the current reviewed
source. Standard CodeBuild issues a main release tag; no experimental tag is relabeled as a release.
There is no task-target manifest, so registration cannot update the FYP family or activate a trigger.
The source archive is deterministic and the inference suite verifies its deployment SHA pin.
The binding is generated from actual applied Idea outputs and adds only `idea_appconfig`; SHA256
`b2ec157fe3223aa37ca8e2961cc4fa61158a56f64ef961e8d3799b15eceabae8`. The release state was idle
at generation. Immutable S3 publication and a fresh idle check still precede activation.

## Idea 默认 Endpoint 与配置版本

- `python -m unittest discover -s recommend_tf/inference/tests` 32 项通过。覆盖默认模型探测、显式版本旧定义兼容、省略版本之后继续生成下一次发布、拒绝重复候选，以及 worker 源码归档摘要。
- AI Engine 执行 `GOWORK=off go test ./...` 和 `GOWORK=off go vet ./...` 均通过。Idea 合同测试校验省略配置版本后的 manifest 身份，打分测试校验调用只发 Endpoint 和标准推理输入。
- 生产当前 Idea Endpoint 合成探测为 2 条样本、3 个输出。带版本头与默认模型两次调用均返回 HTTP 200，每个输出值相等。

## 日更发布恢复

运行 `python -m pytest -q recommend_tf/inference/tests`。测试覆盖 AppConfig COMPLETE 后数据面短暂读到旧配置、外部改动拒绝、ECS 原任务详情过期后的接管，以及错误任务、集群、任务族和状态的拒绝。重新打包后运行 `python3 idea-mapped-daily/build/package_source.py` 校验启动清单的归档摘要。

生产验收回读 Idea 专用日志权限、两个触发器的同一新任务版本，以及现有发布水位。下一次自然日训结束后对账 AppConfig 的模型定义与路由、发布水位和新端点的实际评分日志。
