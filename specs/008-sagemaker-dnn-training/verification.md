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
