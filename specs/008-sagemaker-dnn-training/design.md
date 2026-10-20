# Spec 008：SageMaker CPU 实验训练设计

状态：2026-10-20 用户确认需求与设计；实施计划见 tasks.md，尚未实现或部署。
Source: 2026-10-19 会话确认、根 AGENTS.md、Spec 007，以及当前训练代码和只读 AWS 核查。
以下模块划分与平台映射源自会话设计推导 [INFERRED]，已随本方案获用户确认；不代表现有能力。

## 1. 当前状态与目标状态

当前状态：`run.py` 提供 train/test，`train_rerank.py` 装配模型，
`comm/training_pipeline.py` 读取 Manifest、拟合与编码、训练并写出本地 Run。
DNN 与 DeepFM 共用 rerank 实现。现有 build/Dockerfile 默认启动 scheduled，
尚无 SageMaker 目录映射、提交或结果归档合同。

目标状态：新增薄平台适配层调用现有用例，保持本地入口和模型不变。
平台适配依赖公共合同和用例入口；comm、models 不反向依赖 SageMaker。
不建立独立仓库，不提前建设多平台通用框架。

## 2. 目录与职责

```text
recommend/train/platforms/sagemaker/
  config.py       # 校验独立云端运行配置
  submit.py       # 冻结提交记录，构建并提交单实例 Training Job 请求
  entrypoint.py   # 平台目录适配，依次调用既有 train/test 用例
  artifacts.py    # 成功/失败产物清单和归档映射
  jobs.py         # 查询 AWS 状态、日志/结果位置，汇总各任务 AUC
build/Dockerfile.sagemaker
deploy/prod/sagemaker.yaml      # 实现时新增；引用 iac binding，不含凭据
specs/008-sagemaker-dnn-training/
```

上图 deploy 路径位于仓库根目录。现有统一 CLI 增加薄命令分发，或提供平台模块入口；
具体 CLI 入口见 tasks.md。操作示意：

```text
submit --config <实验.yaml> --runtime-config <云端.yaml>
status --job-name <name>
stop --job-name <name>
compare --job-names <name-a> <name-b>
```

不在 run.py 放特征算法，不在模型中访问 AWS。外部用例调用使用参数数组并检查退出码。
既有 boto3 等依赖能满足时不额外引入 SageMaker SDK。

## 3. 两类配置

| 实验 YAML（用户维护） | 云端运行配置（独立维护） |
| --- | --- |
| manifest_uri、日期切分、任务/标签/指标、特征、网络、loss/optimizer、seed、epoch、batch、线程与内存/缓存预算 | environment、region、固定镜像 digest、执行角色引用、子网/SG 引用、实例类型/数量、磁盘、硬超时、提交与产物位置 |

初始规划为 PROD/us-east-2/ml.m5.2xlarge/单实例/On-Demand。
没有隐式 PROD 默认值；每次提交明确指定运行配置。主机 32 GiB 不等于进程可用预算，
保留系统余量；平台校验预算与资源是否冲突，不能静默更改线程或超参数。
本地路径（缓存与输出）只做平台执行位置映射，记录原配置和解析后配置，算法语义保持不变。

## 4. 提交与输入冻结

1. 本地校验实验与云端配置，生成唯一 Run/job 身份。
2. 使用固定镜像 digest；镜像内源码身份与提交记录关联。未进入镜像的本地修改不得声称被执行。
   首期不必每次改 YAML 重建镜像；代码或依赖改变才构建新镜像。
3. 在提交阶段读取并校验精确 Manifest 与版本合同，冻结原文、摘要和分区引用。
   保存原始 YAML、解析配置、运行配置摘要、源码/依赖/镜像身份；不上传凭据。
4. 将提交包写入唯一、create-only 的实验前缀，再调用 CreateTrainingJob。
   此步骤属于未来执行能力，本次文档工作不执行。
5. 容器必须使用提交包中的冻结输入，不重新解析原地址上的可变内容；分片仍按冻结引用读取并校验。
   dataset 的用户配置仍只有 manifest_uri，不新增用户维护的摘要字段。
6. 提交后返回 job name/ARN、Run ID 与控制台链接；本地命令结束不影响任务运行。

须在公共输入层提供窄的冻结输入消费接口，不能借用 resume 参数假装续训，
也不能仅在提交端记录摘要却让训练重新读取可变地址。
重复提交的幂等性按提交身份处理；改配置产生新 Run，不覆盖已有提交包或输出。
提交包已写入但任务创建失败时记录失败状态，保留可追踪记录。

## 5. 容器运行与产物

- 专用训练镜像适配 SageMaker train 启动约定，使用 /opt/ml 输入/输出目录。
- 依赖在镜像构建阶段固定，运行时不在线安装包；先验证 amd64 镜像，不能直接假设 Mac ARM 产物兼容。
- 首期在一个任务内顺序执行 train、独立 test；test 复用 Run 的冻结输入与 feature state。
  epoch/checkpoint 选择仍按 YAML，跨实验只展示指标。
- stdout/stderr 输出结构化阶段日志及 task/split AUC；不输出原始样本或敏感特征。
- `/opt/ml/model/` 仅放可交付的实验产物集合，排除编码缓存和数据分片；归档名不构成线上推理合同。
  训练 checkpoint、metrics、feature state、lineage、冻结输入、配置和 test 结果均列入完整性清单。
- `/opt/ml/output/data/` 保存运行诊断与性能报告；失败原因写入平台约定位置并非零退出。
- 不使用 /opt/ml/checkpoints 的自动续训或 Spot；现有本地 resume 不受此变更影响。

状态分离：提交成功 → AWS 运行 → 训练完成 → 独立评估完成 → 平台归档完成。
只有 AWS Completed 且最终产物清单/摘要/必要文件检查通过，客户端才标记完整实验成功。
单独的训练 status=complete 不代表上传完成；test/归档失败不得伪报完整实验成功。
失败时保留能够取得的诊断与已完成训练结果，不保证强杀后所有本地文件可恢复。

## 6. 指标展示与性能

首期不创建 Studio Domain，使用 SageMaker 控制台 / CloudWatch 网页查看任务及日志。
2026-10-20 用户要求 AUC/loss 横轴为 step，原仅使用 CloudWatch 的可视化方案不足，
用户随后确认本地 TensorBoard 作为 step 可视化界面，训练任务不依赖本地查看进程。
Source: 2026-10-19 用户确认本地提交、网页看 AUC/loss、后期接 Studio。

[INFERRED] 建议：公共训练用例通过与 AWS 无关的指标事件接口输出任务名、split、epoch/step、
指标名和值；平台适配负责日志格式与 SageMaker MetricDefinitions，不把 AWS SDK 放入模型。
loss 按显式配置的 step 间隔上报，初始值为 100 global step，validation 各任务 AUC/loss 在评估完成后上报，
独立 test 仅完成后输出最终值。不为绘图隐式新增全量训练集 AUC 评估。
训练期间实时 flush；事件经日志采集后成为网页时间序列，允许平台采集延迟。
指标缺失/不可计算写原因与统计，不向数值曲线填入 0 或 NaN。
CloudWatch 原生横轴为时间，不能替代已确认的 step 横轴要求。
已确认补充 TensorBoard 事件输出，训练期间按可配置周期同步到 S3，初始周期 30 秒；
本地查看端周期拉取后用浏览器展示，初始拉取周期同为 30 秒；不能把单段同步周期当作端到端延迟保证。
Source: 2026-10-20 用户确认 Q22；[INFERRED] 本地拉取采用同周期，结束时补做最终同步。
训练仍在云端独立运行，本地只负责查看。该方案无需 Studio，但需要新增/确认 TensorBoard 依赖。
首期不部署 AWS 托管 TensorBoard 或独立云端 TensorBoard 服务。本地查看服务仅绑定回环地址。
step 定义为跨 epoch 单调递增的 optimizer update 次数，validation 结果绑定评估时的 global step；
test 结果绑定所评估 checkpoint 的 step，明确标记 split。global step 横轴及每 100 step 记录、每 30 秒同步的初始配置已确认。

MetricDefinitions 根据 YAML 的任务和指标生成，校验 AWS 数量与命名限制；
超限时明确报告，不静默丢弃任务指标。最终 metrics/test 产物仍为完整结果依据，
网页采样/聚合不得用于重写原始结果。跨任务 compare 读取归档结果，生成各任务 AUC 对比表，
同时显示 split、配置摘要、数据摘要、样本量、耗时及不可计算原因；不增加综合排名。
不假设 AWS 会自动读取 metrics.json，也不默认引入 MLflow 或自定义网页。

后期 Studio 接入只增加 Domain/用户/权限及可选实验跟踪适配，复用既有训练接口与产物。
历史任务和 S3 产物保留；若要纳入 MLflow 等实验表，另外执行导入，不假定自动同步。

同批数据一致性由用户配置，汇总结果展示差异，不强制拒绝比较。
性能报告区分本地提交耗时、AWS 排队/准备、容器阶段、最终上传与总时长；
计费以 AWS 返回用量与账单为准，不拿模型 fit 时间替代完整任务成本。
一小时目标先在代表性冻结输入上验证，再测并发；增加 vCPU 不自动改变 YAML 线程数。

## 7. doku-iac 的改动范围

Source: doku-iac/AGENTS.md 和 2026-10-19 PROD 网络/权限只读核查。

1. 声明训练执行角色，信任 sagemaker.amazonaws.com；限定输入读取、实验前缀写入、
   ECR 镜像读取、日志、所需 KMS 和 VPC ENI 权限。具体资源范围按已确认 storage-plan.md 实现。
2. 声明实验训练专用 SG，允许必要的 S3 HTTPS 及实际需要的服务路径；
   不直接复用可访问业务 PostgreSQL/Redis 的离线 SG。
3. 复用现有 PROD VPC、两个 svc 子网及 S3 Gateway Endpoint；核实桶策略、KMS 和端点策略。
   当前 IP 与训练配额充足，不计划扩大子网、提高配额或另建 NAT。
4. 输出角色、子网、SG、S3/KMS 等版本化 binding，doku-train 的运行配置引用 binding；
   不复制 Terraform 内部实现或硬编码资源 ID。
5. 首期不创建 Studio Domain、用户工作空间或开发应用。后期接入 Studio 时单独配置这些资源，
   不改变已有 Training Job 训练实现；相关计算/存储费用独立管理。
6. 后续每日触发与调度资源归 iac，第一阶段不创建定时任务。

当前用户身份模拟已具备提交与 PassRole 权限，但不把个人 SSO 身份当作训练执行角色。
若团队需要为其他使用者开权限，由 iac 定义受限 caller policy，按现有流程绑定。

网络隔离不能机械开启：现有 reader 在容器内直接访问 S3；若开启 SageMaker
EnableNetworkIsolation，会影响该路径。建议首期使用 VPC + 专用 SG/IAM 限制访问，
不启用该开关；若必须启用，则另行设计平台预下载及输入映射，不能假装兼容。

## 8. doku-train 的改动范围

- 新增第 2 节的平台目录、镜像入口和独立运行配置 schema；保持既有训练 YAML schema 的职责。
- 给冻结输入提供明确的公共消费接口；补充分阶段性能记录和最终产物清单。
- 新增提交/查询/比较命令、任务链接、失败处理和只读 dry-run 请求预览。
- 合成 fixture 验证平台合同；AWS API 用 mock/stub 验证，单元测试不访问 PROD。
- 更新 Spec008、architecture 与开发者入口文档；不改变树模型发布链路。

## 9. 实施顺序与审阅边界

需求与设计已确认，tasks.md 已生成。先固定跨仓 binding 合同，再实施本地适配与合同测试。
随后准备 iac 变更供审阅；云端资源变更、镜像上传与付费 smoke 各按明确授权执行。
云端先验证一任务的冷启动和完整结果，再验证 15 并发；此后才进入第二阶段设计。

运行参数和预算见 requirements 第 6 节；角色物理 ID、镜像 digest 等通过实施核查与 binding 解析，不能隐式填成生产默认。

## 10. 官方参考

- [训练容器启动约定](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html)
- [训练输出处理](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-output.html)
- [VPC 训练网络](https://docs.aws.amazon.com/sagemaker/latest/dg/train-vpc.html)
- [网络隔离与 AWS 服务访问](https://docs.aws.amazon.com/sagemaker/latest/dg/interface-vpc-endpoint.html)
- [Studio 费用](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-cost.html)
- [Ohio 官方价目表](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-2/index.json)

- [自定义训练指标](https://docs.aws.amazon.com/sagemaker/latest/dg/define-train-metrics.html)
- [网页查看训练指标](https://docs.aws.amazon.com/sagemaker/latest/dg/view-train-metrics.html)

## 11. 2026-10-20 运行与保留决策

Source: 用户对 Q17–Q19 的确认。

- 复用合适的现有 PROD 桶、实验专用前缀；首期不自动删除实验产物，日志保留 30 天。
  iac 需核查现有桶生命周期是否覆盖新前缀，避免继承非预期过期策略。
- 不实现业务侧并发限制或队列，15 仅为支持与验证目标；不把本地任务计数当作原子限流。
- 不添加业务侧两小时超时或卡顿自动停止，提供明确的手动 stop 命令和阶段/step 日志。
  loss 持续变化只是进展线索，读取、编码、评估、上传期间没有 loss 不代表卡死。
- AWS StoppingCondition.MaxRuntimeInSeconds 官方默认一天；省略并不表示无限运行。
  2026-10-20 用户 Q21 已确认采用 86400 秒服务时限，运行配置与请求显式声明，提交预览显示该值；
  不自行改成两小时或声称无限运行。
- 每日费用提醒首期后置；磁盘大小仍需容量核查后确定。

参考：[AWS StoppingCondition](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StoppingCondition.html)、
[托管 TensorBoard](https://docs.aws.amazon.com/sagemaker/latest/dg/tensorboard-on-sagemaker.html)、
[本地或独立托管 TensorBoard](https://aws.amazon.com/blogs/machine-learning/visualizing-tensorflow-training-jobs-with-tensorboard/)。

## 12. 存储与容量具体提案

Source: [存储授权与磁盘规划](storage-plan.md)，2026-10-20 只读证据。
建议复用既有数据湖桶/key，使用 ml/sagemaker-experiments/environment=prod/ 前缀，
避免继承训练输入 180 天过期规则。每任务初始磁盘 100 GiB；存储方案与月预算 $250–300 已于 2026-10-20 获用户确认。
所有上传显式 SSE-KMS，KMS 条件兼容 Bucket Key；现有输入过期后不承诺历史重跑。

## 2026-10-20 实现说明

平台实现位于 `recommend/train/platforms/sagemaker/`；公共指标 context observer 位于 comm，
模型只发事件，不依赖 AWS。容器直接调用既有 train/test Python 用例，异常使入口非零退出。
提交包包含原始 YAML（CLI 提交时）、校验后的配置、运行配置、binding 和冻结合同原文。
PROD 示例有明确占位值，不可直接使用。IaC 模块默认不启用，binding 由已授权部署输出生成。
本地 step 网页已做 HTTP smoke，容器与生产资源验收状态以 verification.md 为准。
