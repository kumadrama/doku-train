# Spec 028：TF Endpoint 首次部署与自动日更设计

状态：需求与设计已获用户确认，已授权本地代码实施；实现/验证进度见 tasks.md 和 verification.md，尚未授权云部署。

Source: [requirements.md](requirements.md)；Spec 027 当前实现和运行合同；2026-10-30 用户确认以可测试 Endpoint 为交付终点。下述新增模块结构为 [INFERRED] 实施设计。

## 1. 架构与当前差距

```text
首次：历史成功 run → audit_run → evidence → bootstrap ───────────┐
                                                              ↓
日更：上游 published → ready adapter → 一次性 ECS executor       │
                                  ↓                           │
                            冻结 28+1 输入                     │
                                  ↓                           │
                       SageMaker CPU Training Job             │
                                  ↓                           │
                         validation 最优 checkpoint ───────────┤
                                                              ↓
                                             export_model + parity
                                                              ↓
                                                 不可变 S3 模型包
                                                              ↓
                                      模型包 + 固定推理镜像 digest
                                                              ↓
                                    Model / EndpointConfig / Endpoint
                                                              ↓
                                           配置对账 + 实际请求 probe
                                                              ↓
                                                记录实际在线版本
```

当前已有：daily_train_model、export_model、daily_pipeline、publish_model、封存校验及模拟测试。缺少真实 run 核验工具、ECS 实际启动适配、独立编排镜像、可运行推理镜像、环境装配和云端验收。保留已有状态机，不复制训练/选模/发布逻辑。

## 2. doku-train 代码改动

| 路径 | 动作与职责 |
| --- | --- |
| `recommend_tf/platforms/sagemaker/audit_run.py` | 新增：接收精确 run URI、来源信息和本地输出目录；列出/获取预期产物，核对 run 与 selection/checkpoint，生成审核报告及 evidence；不改远端历史 run |
| `recommend_tf/platforms/sagemaker/executor.py` | 新增：ECS 启动适配，下载显式输入文件、从 ECS task metadata 读取实际 task ARN、验证环境绑定并调用已有编排；不允许固定伪造执行者身份 |
| `recommend_tf/Dockerfile.pipeline` | 新增：编排和 CPU TF 导出依赖；不内置真实数据、凭据或环境默认配置 |
| `recommend_tf/Dockerfile.train` | 按需修改构建装配，支持锁定配方和 daily 冻结输入；保持离线实验路径 |
| `recommend_tf/platforms/sagemaker/daily_pipeline.py` | 提取可供 executor 调用的窄入口；保持 CLI、bootstrap、日更阶段及幂等语义；按真实容器合同集成 probe |
| `recommend_tf/platforms/sagemaker/publish_model.py` | 按实际 AWS 集成测试修正参数/错误对账，复用不可变上传、ALL_AT_ONCE 和回滚，不新建第二个发布器 |
| `recommend_tf/train/export_model.py`、`models/rerank/export_adapter.py` | 原则上复用；仅在真实模型核验发现不支持项时修复，并补合成回归测试，禁止静默改变预测语义 |
| `recommend_tf/train/tests/test_run_audit.py` | 新增历史证据缺失、错配、路径逃逸和审核拒绝测试 |
| `recommend_tf/train/tests/test_executor.py` | 新增 metadata/配置失败及恢复身份测试 |
| `recommend_tf/train/tests/test_daily_publication.py` | 扩展真实 wire 合同的模拟测试，保持离线隔离与发布恢复覆盖 |
| `recommend_tf/README.md`、本 Spec 运行/验证文档 | 增加首次、日更、恢复与服务端交付命令 |

环境绑定和真实实验配置保留在部署配置交付链路，不提交真实样本/模型/权重。审核工具不得以“现场计算一遍 SHA256”代替来源验证；只有身份与交叉证据验证成功才允许生成可用于导出的 evidence。审核报告记录输入 URI、实际选中 epoch、截止日、来源、校验项及失败原因。

## 3. recommend_tf/inference 代码改动

Source: 本次实施中用户明确指定先放 `doku-train/recommend_tf/inference/`，不放入 train。以下推理路径以 recommend_tf 为基准，Dockerfile、README 和 contracts 均位于 inference 内。

| 拟新增路径 | 职责 |
| --- | --- |
| `inference/artifact_loader.py` | 校验 manifest 文件清单、合同版本与签名；读取实际模型版本；定位 SavedModel |
| `inference/request_adapter.py` | 校验原始请求，按 feature-config 执行默认、历史整理/截断/padding、dtype 和张量组装；不重新拟合词表 |
| `inference/response_adapter.py` | 只输出四任务及实际加载模型版本，保持候选顺序，拒绝维度错配 |
| `inference/app.py`（按需） | SageMaker 健康/推理接口及 TensorFlow Serving 调用桥接；官方镜像扩展钩子足够时不实现独立 HTTP 层 |
| `Dockerfile`、依赖锁定文件 | 固定 CPU TensorFlow Serving 及适配层；启动与退出正确管理子进程 |
| `tests/test_contract.py`、`tests/test_parity.py`、`tests/test_container.py` | 请求边界、数值一致性、健康检查/加载失败和容器实测 |
| `contracts/request-v1.json`、`contracts/response-v1.json`、`README.md` | 版本化 wire 合同、合成样例、启动与压测说明 |

先做镜像兼容性验证，再选择官方镜像扩展钩子或薄适配层；不自研 C++ 推理引擎，不同时实现两套服务框架。

### 请求/响应提案

wire 合同采用版本化 JSON 批量候选结构（待设计确认后固定）：请求含 `schema_version` 和有序 `instances`；每行是有类型的具名特征。响应含 `model_version` 和等长有序 `predictions`，每行恰好四个主任务。具体原始字段、类型和历史结构以真实模型 feature-config/signature 导出清单确定，不能用训练标签字段冒充请求特征。

在读取模型前校验包摘要，读模型后检查签名及四任务映射。加载失败/未就绪健康检查失败；输入格式错误返回 4xx，运行故障返回 5xx；不打印完整请求。批量/体积上限显式配置。输出单位只声明已验证的实际预测含义。

导出器 parity 只覆盖 checkpoint→SavedModel；新增容器 parity 必须覆盖原始请求→适配→TensorFlow Serving→响应的全路径，防止两端历史整理或默认值分歧。测试预期来自冻结训练/导出结果，不只比较两个共享同一适配错误的路径。

## 4. doku-iac 与上游代码改动

已存在 `modules/sagemaker_training_access`、ECS 相关模块和 `scripts/gen-sagemaker-binding.py`；训练访问模块明确不应混入生产发布权限。实施时先读 iac 自身协作约束，复用适用模块，不复制其实现。

| 拟新增/修改路径 | 职责 |
| --- | --- |
| `modules/sagemaker_inference_access/` | 推理加载角色及 S3/KMS/ECR 权限；调用权限单独交付 |
| `modules/tf_daily_pipeline/` | 一次性 ECS task definition、发布身份权限、日志及输入装配；必要的触发权限与现有模块组合 |
| `scripts/gen-tf-daily-binding.py` | 从明确 Terraform 输出生成 Spec 027 binding，校验完整性，不触发 AWS 写操作 |
| `envs/<environment>/tf_daily_pipeline.tf` | 显式装配网络、角色、资源和镜像；实际环境确认后落位 |
| 对应模块的 `main.tftest.hcl` 与脚本测试 | 权限边界、无默认生产目标、绑定合同、触发参数验证 |

Terraform 管理支撑资源；首次 Endpoint 创建与每日 Model/EndpointConfig 更新统一由发布器执行，不用 Terraform 持续维护在线 EndpointConfig。IAM PassRole 限定明确角色；编排器需要训练提交/查询、模型部署/查询、probe、S3 条件状态写入及 ECS 旧任务状态读取等最小权限，推理角色不获得发布权限。

[NEEDS CLARIFICATION: 实际上游生产者和事件编排入口需定位。] ready adapter 放在上游发布/编排所有者仓库，输出 Spec 027 `doku-tf-dataset-ready/v1`；不在 Terraform 中重新实现数据成熟性判断。若当前生产者没有该合同，这是一项外部必需代码改动，不能仅靠新增 EventBridge 规则完成。

上游交付 ready URI/摘要；executor 获取并验证后交由现有 freeze 校验连续窗口。触发失败通过明确的重试/失败告警处理，补齐数据后可重新触发，不能依赖日更 timer 猜测就绪。

监控与 doku-monitoring 对接训练/部署终态、新鲜度、Endpoint 错误/延迟和部署回滚所需告警。具体告警资源归属按两仓现有规范确定；启用自动更新前必须有可引用的告警名称及接收路径。

## 5. 首次、手动日更与自动日更

首次：audit_run → exporter/container parity → 构建并固定镜像 → 环境绑定 → daily_pipeline bootstrap → Endpoint 对账/probe → 交付调用说明。bootstrap ready/cutoff 使用历史 validation 实际日期；旧 evidence 仅在审核成功后使用。

手动日更：获取真实连续 29 天 ready，手动启动同一 ECS task，完整执行 Training Job→导出→更新。训练参数不因增加日更能力改变，保持 offline experiment 行为。部署探测失败应恢复旧 TF 配置；演练显式回滚后，自动数据高水位保持不下降。

自动日更：通过手动全链路及回滚验证后启用上游成功触发。相同事件复用 operation；预算调整不改变 operation；业务服务继续调用同一 Endpoint 名称。后续镜像/合同升级是明确发布变更，不作为日更中的隐式升级。

## 6. 验证证据与实施阶段

| 阶段 | 要求证据 |
| --- | --- |
| 文档确认 | requirements/design 确认后生成 tasks；未确认的仓库与云参数不得猜测 |
| 本地实现 | doku-train make check、推理合同/容器测试、iac validate/test 与绑定测试 |
| 真实模型审核 | 来源核对报告、完整性 inventory、实际 selection 与导出 parity；产物仅存受控位置 |
| 首次 Endpoint | 实际镜像 digest/模型版本/EndpointConfig、健康状态和合成请求结果 |
| 手动全链路 | 冻结输入摘要、Training Job、选择/导出/发布记录、实际在线版本 |
| 恢复与回滚 | 重复事件、执行者终止后恢复、探测失败、旧版本恢复和高水位保持 |
| 自动运行 | 上游真实成功触发、实际日更成功记录及失败/新鲜度监控证据 |

记录压测机器、实例/线程、候选批量、并发、预热、吞吐、延迟分位数和 RSS；不以合成小模型吞吐代替真实模型容量。云端核验、上传、部署和触发启用在相应明确授权后执行。本次文档不代表任何云操作已执行或验收通过。

## 7. 参考

- [Spec 027 运行合同](../027-tf-daily-serving/operations.md)
- [Spec 027 设计](../027-tf-daily-serving/design.md)
- [AWS 模型部署组成](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model-get-started.html)
- [AWS 推理容器合同](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html)

## 本地实施补充

Source: 本次用户仓库位置修订及实际代码。生产 inference 仅使用标准库适配并调用官方 CPU TensorFlow Serving 2.11.1，不导入 train。采用 HTTP 薄桥接承载原始特征协议，实际容器已加入 CI。IaC 仅新增隔离模块与生成器，未实例化环境；现有仓库未提交改动保持不动。EventBridge 触发默认关闭，新增 DLQ、任务退出失败与发布心跳告警；心跳不等于模型数据截止日新鲜度。后者及实际上游成熟性证据仍需跨仓装配。

Source: 2026-10-30 用户最新指示：第④⑤步继续实施，窗口改为 28 天训练 + 1 天 validation，取代此前等待 31 天的安排。

### 实际上游成功事件适配

现有上游没有发出 TF ready 自定义事件。新增 `platforms/sagemaker/ready_event.py` Lambda 适配器接收指定 PROD 样本任务的 ECS STOPPED 事件，并通过 DescribeTasks 二次核验任务身份、成功退出、daily join/project-training 参数及明确日期。小时任务忽略；不能凭 S3 写入事件认为数据成熟。按配置 28+1 读取日级 `_SUCCESS`，核验其版本、不可变 Manifest 摘要、24 个源 marker 和特征合同，写入摘要寻址 ready 并发出既定 ready 事件。训练入口仍再次完整验证窗口与合同。Lambda、IAM 和事件接线由独立 IaC rollout stack 管理，自动触发验收后启用。

SageMaker 部署镜像使用 Docker distribution manifest v2，拒绝仅通过 ECS 验证的 OCI manifest。运行时镜像 config/layers 均保持摘要固定。
