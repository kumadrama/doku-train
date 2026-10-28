# Spec 008：验证记录

状态：本地实现及验证已进行；未部署、未启动云训练。下方早期记录为历史状态。
Source: 2026-10-19 当前会话工具返回结果。

## 只读环境证据

使用显式 profile doku-prod-developer、region us-east-2：

- `aws sts get-caller-identity`：成功，确认 PROD SSO Developer 身份。
- `aws service-quotas list-service-quotas --service-code sagemaker`：成功；
  ml.m5.2xlarge training 配额和全部训练实例配额均 30。
- `aws sagemaker list-training-jobs --status-equals InProgress`：成功，空列表。
- EC2 `describe-vpcs`、`describe-subnets`、`describe-route-tables`、
  `describe-vpc-endpoints`、`describe-network-acls`、`describe-security-groups`、
  `describe-vpc-attribute`：成功；结果摘要见 requirements。
- IAM `get-role`、`list-attached-role-policies`、`list-role-policies`、`get-role-policy`、
  `list-roles`：成功；未查到 SageMaker 服务信任角色。
- `aws iam simulate-principal-policy`：CreateTrainingJob、DescribeTrainingJob、StopTrainingJob、
  CreateDomain、PassRole、CreateRole、PutRolePolicy 均 allowed，无 missing context；
  不代表 SCP 与具体资源策略端到端验证。
- `aws sagemaker list-domains`：成功，Ohio 空列表。
- Python urllib 只读获取 AWS Ohio 官方价格 JSON：发布于 2026-10-16，
  Training ml.m5.2xlarge 为 $0.461/小时。

以上为快照，不是实际提交成功、网络吞吐或 15 并发的验收证据。
未执行 Create/Update/Delete、配额申请、角色修改、镜像上传或训练 API。

## 文档验证

- `git diff --check`：通过。
- Python 标准库链接检查：扫描 AGENTS.md、docs/architecture/components.md 和 Spec008
  的 Markdown 相对路径链接，全部目标存在；不把网页外链检查当成本地索引验证。
- 本轮修改仅为 Spec008 文档、AGENTS 索引及 components 目标状态说明。
  工作区原有代码与其他 Spec 变更保留；未运行 make check，因为未修改代码或配置。

## 设计确认后的验收计划（未执行）

| 需求 | 必需证据 |
| --- | --- |
| R1 配置与冻结 | 提交后修改原 YAML/Manifest 不影响 Run；源码/镜像身份一致性测试 |
| R2 执行 | 合成数据容器 train+test；失败非零退出；单机 CPU 配置检查 |
| R3 结果 | AUC null 原因、独立 test 结果、产物清单、归档失败不报成功、重复输出拒绝 |
| R4 界面 | 无 Studio Domain 时的 CLI 提交与网页查看；训练期间 AUC/loss 事件及提取规则；任务/split 不串线；null 不伪造为 0；compare 与最终产物一致 |
| 网络权限 | 授权后真实单任务冷启动读取与归档，确认无业务数据库/Redis 访问需求 |
| 性能并发 | 授权后代表性单任务和 15 并发记录总耗时、阶段耗时、RSS、实际计费 |

2026-10-20 需求设计已确认并生成 tasks.md；本表和任务清单均不表示实现已完成。

## 首期界面决策同步

2026-10-19：已同步首期不使用 Studio、本地提交与 AWS 网页 AUC/loss 曲线需求；
补充后期 Studio 接入边界和指标频率/保留策略待确认项。未创建任何云端资源或修改实现。
本次重新运行 git diff --check 和 5 份文档的本地链接检查，均通过。

## 2026-10-20 决策同步

记录 step 横轴硬需求、产物/日志保留、手动停止及不建并发队列；
step 网页承载方式与 AWS 服务时限尚待确认。新增验收要求：global step 跨 epoch 连续、
validation/test 绑定正确 step、手动停止不声明完整成功、无业务侧两小时超时。
本次文档 diff 与本地索引链接检查通过；未改实现或云资源。

## 2026-10-20 Q20/Q21 确认

用户确认本地 TensorBoard step 曲线与 SageMaker 86400 秒服务时限。
待实施验收增加：Training Job 请求与提交预览的时限一致；关闭本地 TensorBoard 不影响训练；
训练期间指标可同步并按 global step 查看。仅同步文档，未执行这些实现验收。
本轮 git diff --check 与 49 个本地链接检查通过。

## 2026-10-20 Q22 确认

训练 loss 每 100 global step 记录、TensorBoard 每 30 秒同步，均可配置；
validation 每次评估完成记录，不新增 AUC 评估。后续测试验证记录频率、周期配置与最终同步，
不将同步周期作为网页实时性承诺。本轮仅更新文档，git diff --check 与 49 个本地链接检查通过。

## 2026-10-20 存储只读核查

S3 get-bucket-location/encryption/lifecycle-configuration/policy/versioning，KMS describe-key/
get-key-policy，实验候选前缀 list-objects-v2 和 17 个分片 head-object 均成功；未读取样本正文。
本地 Python 按文件 stat().st_size 汇总缓存和 Run，各分片总大小与云上 HeadObject 一致。
具体结果见 [存储规划](storage-plan.md)。未做任何云端写入或训练。
文档 git diff --check 与本地链接检查通过。

## 2026-10-20 方案确认与任务生成

用户确认存储、100 GiB 磁盘和 $250–300/月规划预算，并确认需求与设计。
已生成 [实施任务](tasks.md)，分离 T01–T06 本地改动与 C01–C03 后续另行授权的云端阶段。
本轮只修改文档，未运行代码测试、terraform 或 AWS 命令；原有工作区变更保留。
文档 diff --check 与所有本地索引链接检查通过。

## 2026-10-20 本地实现验证

- `make check`：锁文件、145 文件格式、lint、57 源文件 mypy 全部通过；176 passed、3 skipped（72.22 秒）。
- 新增 SageMaker 测试 8 项：只读冻结、输入越界、SSE-KMS/create-only、丢失创建响应重试、step/null 事件、冻结输入 train/test/归档、test 失败、归档失败。均使用独立合成数据与内存 AWS stub。
- 本地 TensorBoard 回环 HTTP smoke：启动并返回 200，随后关闭；事件读取测试检查 step=[2,4]、split/task、null 不填 0。
- iac `terraform fmt -check modules/sagemaker_training_access envs/prod/sagemaker_training.auto.tf` 通过。
- iac 新模块 `init -backend=false`、`validate` 通过；`terraform test` 使用 mock_provider，1 passed，0 failed。无 AWS API 操作。
- `python3 -m unittest discover -s tests -p test_sagemaker_binding.py`：1 passed，验证生成内容及禁止覆盖。
- Docker amd64 构建失败：本机 QEMU 在 `uv sync --frozen --no-dev --no-install-project` SIGSEGV/139。未声称镜像构建或容器 smoke 通过；需原生 amd64 环境继续验证。
- 资源报告包含阶段耗时、进程峰值 RSS、每 5 秒采样的目录峰值/最低可用磁盘；采样值不代表瞬时绝对峰值或真实生产容量证据。
- 未运行 terraform apply、云端 plan、ECR push、S3 写入或 CreateTrainingJob。实际 KMS 服务上传、网络、日志组与 15 并发仍待后续授权验收。
- 最后补充原始 YAML 冻结后，针对性重跑 `pytest recommend/train/tests/unit/test_sagemaker_platform.py -q`：8 passed（4.53 秒），ruff 通过。
- 两仓 `git diff --check` 通过；本仓 README/AGENTS/components/Spec008 共 69 个本地链接均可解析。

## 2026-10-20 原生 amd64 与独立 PR

[doku-train PR #12](https://github.com/kumadrama/doku-train/pull/12) 从 origin/main 独立整理，未混入 Spec013 等工作区改动。
[CI run 39392803434](https://github.com/kumadrama/doku-train/actions/runs/39392803434) 全部通过：make check、现有构建、amd64 镜像构建；容器在 `--network none` 下运行合成 DNN/DeepFM 的 train/test/归档/TensorBoard 事件验证。
此证据取代之前 ARM/QEMU 的容器验证阻塞；不代表真实 AWS 网络/KMS/训练性能通过。
只读 AWS 核查复用现有 `doku-train/doku-reco-train` ECR repository；TrainingJobs 日志组不存在，iac 新增 30 天保留声明。未上传镜像、apply 或启动训练。

后续部署审阅：iac PR [#518](https://github.com/kumadrama/doku-iac/pull/518)；完整 PROD validate 通过，定向只读 plan 为 8 create / 0 update / 0 delete，无依赖资源修改。云端 apply、镜像上传及训练仍未执行。原生 amd64 CI 的完整测试为 179 passed（包含 Redis 场景）。

## 2026-10-20 C01 获授权部署完成

用户明确授权前述 8 项 PROD 资源新增。以 iac 合入提交 cbeb81ca3806bc25322db112fed307422c885d8b 刷新计划，严格核验 8 create / 0 update / 0 delete 后部署。
首次 Terraform backend 写入因 S3 multipart EOF 失败；使用自动保存的恢复文件执行 state push 成功，远端 lineage/serial 与 8 项资源记录核对通过。未 force 或重复 apply。
部署后定向 plan 为 No changes（exit 0），不是全 PROD 漂移审计。角色信任、日志 30 天和 HTTPS 出站均经只读 API 核验。
真实 binding 从 Terraform 模块输出生成并通过本仓 Binding schema；本地副本为 deploy/prod/sagemaker-binding.json。根级 output 在定向部署中未持久化，采用同一 Terraform 输出表达式求值，没有手填物理 ID。
未上传镜像、启动训练、附加人员权限或发布模型；C02/C03 尚未执行。

## 首次单任务云端验收（进行中）

2026-10-20 用户明确授权上传镜像和运行单任务，不运行 15 并发。镜像通过现有 doku-train CodeBuild 原生 amd64 构建，双模型容器 smoke 成功，上传 ECR 并固定 digest；构建 ID 0c25f4f0-70e6-48bd-b9c0-b326ba743539。
真实 DNN 验收采用现有七任务配置，2026-10-15 train / 16 validation / 17 test，1 epoch，loss 每 10 step。
提交区 Manifest/合同支持已通过 [train PR #13](https://github.com/kumadrama/doku-train/pull/13) 合入，数据分片范围未扩展。
首次任务 smoke-dnn-20261020-01 在训练前因 CreateNetworkInterface 缺少 ec2:Vpc 条件上下文失败。CloudTrail/STS 解码确证，修复 [iac PR #520](https://github.com/kumadrama/doku-iac/pull/520) 保持精确子网/SG 范围；两项 Terraform 测试、IAM 正反向模拟通过，仅 1 policy update，无新增/删除。
串行重试 smoke-dnn-20261020-02 已进入 Training，尚待完整结果验收。浏览器控制在当前 Work 设置中禁用，页面可做 HTTP/图表数据接口验证，不能声称已人工视觉验证。


## 2026-10-20 C02 单任务结果（数据验收通过，视觉验收待完成）

Source: 用户授权“继续上传镜像，先跑一个训练任务，验证云端训练和网页上的 AUC、loss，先不要跑 15 个并发”。

- 真实成功任务：`doku-exp-smoke-dnn-20261020-02`，AWS `Completed`；首次尝试在训练前失败，终止后才串行重试，没有并行运行两个训练任务。
- 镜像 digest：`sha256:94299a3928b0e5f572f38a6052ce1a1b450a818fdf4983e7ee2c5a1f3740841d`；构建源码 `64335efb478994e6ad1cdd6d896603e696c2cf4f`。原生 CodeBuild 双模型合成 smoke 通过；未额外启动云端合成训练。
- 实例为 `ml.m5.2xlarge` ×1，8 vCPU / 32 GiB，API 请求卷容量 100 GiB，On-Demand，86400 秒服务时限。
- 2026-10-15 train / 16 validation / 17 test，1 epoch；这是代表性小规模验收，不是 28 天容量或 15 并发 SLA。
- AWS 创建 09:28:46 UTC，TrainingStartTime 09:29:27，TrainingEndTime 09:51:28；TrainingTimeInSeconds 与 BillableTimeInSeconds 均为 1321（22 分 1 秒）。
- 容器 train 用例 878.96 秒、独立 test 395.42 秒；fit_features 131.04 秒，encode_train 348.35 秒，encode_validation 378.67 秒，fit_model 9.50 秒。当前耗时主要在特征拟合/编码。
- 进程峰值 RSS 2,168,283,136 bytes（2.02 GiB）；work 目录 5 秒采样峰值 2,644,382,467 bytes（2.46 GiB）。采样不是瞬时峰值；`statvfs(/opt/ml)` 可用空间属于容器文件系统视图，不用它证明请求的 100 GiB 卷实际大小。
- 正式 `compare` CLI exit 0：校验归档必需文件、所有文件 SHA256、submission 身份、独立 test 证据；结果为 complete。S3 输出加密及日志权限在真实运行中验证。
- 日志恢复修复见 [iac #521](https://github.com/kumadrama/doku-iac/pull/521)：仅补充专用日志组 CreateLogGroup/DescribeLogStreams，Terraform 3 个 mock 测试通过，仅 1 policy update；在同一活动任务中恢复日志，无重启。真实 binding 见已合入 [iac #519](https://github.com/kumadrama/doku-iac/pull/519)。
- TensorBoard HTTP 200，33 条 loss/AUC 序列检查通过：7 条 train loss 的 step 为 `[10,20,30,40,50,60]`，validation/test 的 26 条 loss/AUC 在 step 63。各值有限，序列不串 split/task，与归档及 SageMaker FinalMetricDataList 在浮点容差内一致。单 epoch 的 AUC 为单点，不声称已有跨 epoch 曲线。
- 浏览器工具多次返回 `Local browser use is disabled in Work settings`；用户表示会开启，但复查仍被禁用。页面数据接口验收通过，截图/实际曲线画面尚未验收。
- 本地证据位于忽略目录 `runs/sagemaker-smoke-20261020/`：`job-status.json`、`verified-result.json`、`compare.json`、`resources.json`、`tensorboard-api-evidence.json`、`cloudwatch.log` 和 TensorBoard events；不提交实验 YAML、指标报告或模型权重。

可复现读取命令（需显式配置身份，命令只读云端结果）：

```sh
.venv/bin/python -m recommend.train.platforms.sagemaker \
  --runtime-config runs/sagemaker-smoke-20261020/runtime.yaml \
  --profile doku-prod-developer \
  compare --job-names doku-exp-smoke-dnn-20261020-02
```

本轮实际执行该命令 exit 0；另用 TensorBoard `/data/plugin/scalars/tags` 与
`/data/plugin/scalars/scalars` API 对 33 条序列逐条断言，全部通过。
未运行 15 并发、每日调度或模型发布。真实 DeepFM 和更长日期容量仍未验收。

## 2026-10-28：单机 16 vCPU 校验扩展

在从 main b625f61 创建的独立工作树验证；未带入原工作区其他改动。
- `pytest recommend/train/tests/unit/test_sagemaker_platform.py -q`：25 passed，16.59 秒。
- `ruff check .`：通过；`mypy recommend`：69 源文件通过；`git diff --check`：通过。
- 新增两种允许实例的请求生成与包反序列化测试，以及其他实例/GPU/多机拒绝测试。
