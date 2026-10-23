# Spec 008：实施任务

状态：2026-10-20 本地/原生容器验证与 C01 完成，C02 单任务云端数据验收通过；视觉验收及 C03 未完成。
Source: 用户先授权代码实施，后授权云端部署、上传镜像和单任务验证；15 并发仍未授权。
本文件不构成 apply、上传镜像/配置、提交训练或改变 PROD 权限的授权。

## 依赖顺序

```text
T01 train/iac 合同
  → T02 iac 本地资源声明与校验（不 apply）
  → T03 train 配置/冻结/提交与查询
  → T04 train 容器入口与归档
  → T05 train 指标与本地 TensorBoard
  → T06 跨仓本地集成验证与 PR
  → C01 iac 云端部署与 binding 生成（另行授权）
  → C02 train 镜像上传与单任务 smoke（另行授权）
  → C03 15 并发及预算验收（另行授权）
```

T02 的云端 apply 不阻塞 T03–T06：本地使用独立合成 binding fixture。
先实现并验证可审阅代码，再进入云端执行；各仓使用短生命周期分支和 PR，不直接更新远端 main。
已有未提交变更须保留，实施前识别与本任务的交叉文件，不把其他 Spec 的工作一并提交。

## T01 — 固定跨仓合同（doku-train 主导，doku-iac 配合）

- [x] 在 Spec008 定义版本化运行配置及 binding 字段：environment、region、execution role、
  subnet/SG、输入允许范围、实验桶/前缀、S3 KMS key；物理 ARN/ID 来自 binding。
- [x] 固定运行值：ml.m5.2xlarge、instance_count=1、volume_size=100、max_runtime=86400、
  On-Demand；不启用 warm pool、Spot、队列或业务侧并发上限。
- [x] 固定 Run 提交包、归档清单与指标事件版本，区分原始 YAML、解析配置和运行配置。
- [x] CLI 采用 `python -m recommend.train.platforms.sagemaker`，提供
  `submit --dry-run`、`submit`、`status`、`stop`、`compare`、`tensorboard`；
  平台 __main__.py 仅参数解析和装配，既有 run.py train/test 保持兼容。
- [x] 记录绑定来源和镜像 digest 的解析规则：代码变更需新镜像，只有 YAML 改动可复用镜像。

完成证据：schema/合成 fixture 测试拒绝缺失身份、冲突预算与未批准输入范围；
dry-run 不调用任何写 API、不上传 S3、不提交任务。

## T02 — 基础设施声明与权限（doku-iac，仅本地）

建议文件：`modules/sagemaker_training_access/{main,variables,outputs}.tf`，
`envs/prod/sagemaker_training.auto.tf`，相应 outputs、binding 生成脚本与测试。
按 iac 的逻辑名/for_each 约束落地，不在业务仓硬编码 ARN。

- [x] 新增专用执行角色及 SageMaker 信任条件，受限 S3、KMS、ECR、日志、ENI 权限。
- [x] 输入仅 `ml/fyp-rank-train/daily/environment=prod/*`；实验读写仅已确认实验前缀。
  按需增加 multipart 操作，不授予数据删除或线上发布权限。
- [x] KMS 数据访问限定现有 key、S3 ViaService，兼容 Bucket Key 的桶级 context；
  不给 key 管理权限，不把该 S3 策略直接用于 EBS 加密。
- [x] 专用 SG 只放行所需路径，复用现有 svc 子网与 S3 endpoint；不放开数据库/Redis。
- [x] 核查现有 SageMaker 日志组及保留期限；若影响其他任务，记录影响并在 apply 前审阅，
  目标为本项目日志 30 天，不暗改共享保留行为。
- [x] 输出 T01 binding；caller policy 仅定义需要的权限，不擅自给人员 attach。
- [x] 不创建 Studio、MLflow、NAT、桶、key、定时任务、预算提醒或清理任务。
- [x] 使用合成输出测试 binding 生成；真实绑定只在 C01 apply 后生成，不伪造资源 ID。

完成证据：terraform fmt -check、无后端的 validate（环境允许时）及权限/binding 合同测试；
检查输入无写权限、输出范围受限、KMS key 精确匹配。不运行 apply 或云端 plan。

## T03 — 平台配置与提交管理（doku-train）

文件：`recommend/train/platforms/sagemaker/{__init__,__main__,config,submit,jobs}.py`、
相应 schema、`deploy/prod/sagemaker.yaml`（显式环境与逻辑 binding 引用）、unit/contract tests。

- [x] 实现 T01 配置校验，检查内存/缓存与实例和磁盘的兼容性；不自动改线程、日期或超参数。
- [x] 提交时冻结精确 Manifest/合同；公共 reader 增加独立的 frozen input 消费接口，
  不借用 resume，不再次读取可变地址，不接受越界分片引用。
- [x] 唯一 Run/job/submission 身份，配置包 create-only；所有 PutObject 显式 SSE-KMS/key ARN。
- [x] 固定镜像 digest/源码与依赖身份；本地未进入镜像的代码不能被标为本次训练源码。
- [x] 创建单实例请求，明确 OutputDataConfig.KmsKeyId、86400 秒和 100 GiB。
- [x] 处理提交响应丢失与重复请求，使用确定 job 身份查询恢复；禁止盲目重复创建付费任务。
- [x] 查询、手动停止、控制台链接、实际 AWS 状态/计费用量；返回配额错误，不建队列。

完成证据：AWS stub 测试请求字段、配置冻结、错误传播、幂等与无写 dry-run；
测试不访问 PROD，manifest 修改后已准备的提交包保持一致。

## T04 — 容器、评估与完整产物（doku-train）

文件：平台 `entrypoint.py`、`artifacts.py`，`build/Dockerfile.sagemaker`，公共输入/报告窄接口。

- [x] CPU amd64 镜像按锁文件构建，专用 SageMaker train 入口；运行时不安装依赖。
- [x] 依次调用既有 train/test 公共 Python 用例，异常传递到入口非零退出；不拼 shell，路径映射不改变逻辑配置。
- [x] 归档包含 checkpoint、指标、feature state、lineage、冻结输入、配置及 test 证据；
  排除原始分片、编码缓存和无用途的中间文件。
- [x] 阶段日志与耗时、RSS、输入/编码/临时目录字节量、卷剩余与峰值证据。
- [x] 训练/评估/平台归档状态分开；仅 AWS Completed 且完整性校验通过才报完整实验成功。
- [x] 失败非零退出，手动停止不伪报成功；保留可获取诊断，不承诺强杀续训。

完成证据：合成数据本地容器 smoke；train 成功但 test/归档失败的合同测试；
验证恢复本地路径不受影响，运行 make check。

## T05 — step 指标与本地 TensorBoard（doku-train）

文件：公共指标事件接口、平台事件上传/查看模块、模型 runner 的事件回调、pyproject/lock 和测试。

- [x] 检查既有依赖；需要 TensorBoard 时仅新增所需 CPU 依赖并锁定，不引入 TensorFlow/GPU runtime。
- [x] global step 跨 epoch 连续，loss 每 100 step，validation 指标在评估完成后输出；
  test 绑定所评估 checkpoint step；task/split 标签明确，AUC null 保留原因，不填 0。
- [x] 事件 flush，30 秒周期上传 S3，本地周期同步，正常完成时最终同步；频率显式可配置。
- [x] 上传显式 SSE-KMS，采用唯一事件文件/片段避免覆盖不可变归档；不依赖未核实的服务隐式上传头。
- [x] 本地 TensorBoard 仅监听回环地址，命令输出浏览器链接；退出查看器不停止云训练。
- [x] CloudWatch 日志与可支持的 MetricDefinitions 接入；校验数量限制，不静默丢指标。
- [x] compare 从完整产物汇总 AUC 与身份差异，不加跨实验排名。

完成证据：事件读取器自动核对 step、task/split、频率及 null 行为；
mock S3 验证加密参数、重复同步与完整事件读取；本地 TensorBoard smoke 可绘制 step 曲线。

## T06 — 本地集成与交付（两仓）

- [x] 组合合成 binding、实验 YAML、dry-run 请求、容器与事件输出做端到端本地验证。
- [x] doku-train 执行 make check；doku-iac 执行 T02 校验与仓库相关测试；保存真实命令结果。
- [x] 更新 README 的构建、提交、停止、查看、结果下载流程与成本说明。
- [x] 同步 Spec008/architecture 状态，分别形成 train 和 iac PR，标明依赖及实际部署状态。
- [x] 输出待云端执行清单与资源变更范围；本阶段不得 apply、push 镜像或提交训练。

## 云端阶段（C01/C02 已获单独授权，C03 未授权）

- [x] C01（iac）：另行授权后检查 plan、部署角色/SG/日志等，生成真实 binding；
  不更改原始数据生命周期或扩展发布权限。
- [ ] C02（train）：另行授权构建发布镜像/配置，单任务先用合成数据验证 KMS 上传与归档，
  再用代表性冻结数据验证 train/test、TensorBoard 和 100 GiB 峰值占用。
- [ ] C03（train 主导）：另行授权 15 并发，记录成功/失败、总耗时、实例用量、磁盘、日志/指标费用。
  核对 $250–300/月预算的用量假设；性能未达一小时如实记录，不假设线性加速。

第二阶段每日自动训练、神经模型导出与线上消费合同另行设计，不加入本实施清单。

## 本地交付历史限制（以下被后续验收记录逐项更新）

- T01–T05 的主要实现已落地，T06 的 Python 合成端到端、失败路径和 TensorBoard HTTP smoke 已通过。
- Docker amd64 构建在本机 QEMU 执行 uv sync 时 SIGSEGV（139），未形成容器 smoke 通过证据；需原生 amd64 构建机验证。
- iac 仅验证新增模块（mock provider）与 binding 脚本，未运行 PROD plan。PROD 模块开关为空，待明确 ECR repository ARN 后启用。
- 日志组是否已存在、共享影响及 30 天保留仍为 C01 前置核查；模块不暗改共享日志组。
- 未提交/推送或创建 PR；两仓原有未提交变更保留，混合工作区须分别整理审阅后交付 PR。

2026-10-20 更新：原生 amd64 CI 容器构建及双模型 smoke 已通过，解除 QEMU 阻塞；独立 train PR #12 已创建。日志组只读核查为空，已补充 iac 30 天保留声明，未部署。

2026-10-20 C01 更新：用户单独授权 8 项新增，部署及 state 恢复成功，定向 post-plan 无变化，真实 binding 已生成并校验。C02/C03 保持未执行。

2026-10-20 C02 更新：原生 CodeBuild 镜像构建/上传成功；首次启动因 ENI 权限失败，修复后串行重试成功。真实 DNN 单任务完成 train/test/归档，正式 compare CLI 成功；33 条网页指标与产物/AWS 一致。C02 保持未完全勾选：实际浏览器视觉验收受 Work 开关阻塞；没有额外启动云端合成训练，合成双模型验证在原生构建容器完成。C03 未执行。详见 verification.md。
