# Spec 027：TF 每日训练与 SageMaker 发布设计

状态：2026-10-30 用户确认设计并授权本地代码实施；实施状态与证据见任务/验证。未执行云操作。

Source: 2026-10-30 用户确认 Q1–Q12、最小复用结构和“本仓更新 Endpoint、serving 独立仓库”边界；[需求](requirements.md)。以下具体文件划分、状态与接口细化为 [INFERRED] 设计提案，不代表全部细节已经验收。

## 1. 实施前状态（历史基线）

- `recommend_tf/train/train_rerank.py::train` 当前准备 train/validation/test，冻结输入并调用公共训练流水线。
- `comm/training_pipeline.py::run` 已执行 train-only 词表拟合、逐 epoch 训练/validation、checkpoint 和选模；`evaluate_checkpoint` 提供独立评估。
- `experiment.py` 选模后执行 test；`cloud_train.py` 目前装配 experiment 并上传训练产物。
- `models/rerank/rerank_model.py` 已有无标签 PREDICT，不等于已交付 SavedModel exporter。
- 旧 `recommend/train/scheduled.py` / `run.py::daily_run` 是树模型路径，有其独立 test、发布和可选 Redis 合同，不能直接用作新版 TF 日更。
- 本仓未交付本 Spec 的日更入口、推理包合同和 Endpoint 发布实现。首次候选 epoch 21 来源于本地实验快照，实际远端权重尚待读取核验。

Source: 上述源码及 Spec 024；本会话只读代码核查。实验配置、云快照和真实结果仍保留在本地 runs，不加入源码。

## 2. 目标链路与运行位置

```text
上游每日样本发布成功
  ├─ 原树模型流程（保留）
  └─ 一次性 ECS：platforms/sagemaker/daily_pipeline.py
       ├─ 校验 ready / 日期窗口 / 幂等认领
       ├─ 固化精确多日 Manifest 与运行配置
       ├─ SageMaker CPU Training Job
       │    └─ daily_train_model.py → 共用准备/训练/validation/选模
       ├─ 独立进程 export_model.py → 校验与导出
       ├─ S3 不可变版本完整上传
       └─ publish_model.py → Model → EndpointConfig → UpdateEndpoint
                                     ↓
                          就绪后一次性切换、验证、观察
                                     ↓
                              确认上线或回滚
```

首次模型发布从成功 run 引用进入 export 阶段，不重新训练。导出在具备固定 TF 依赖的受控一次性任务环境执行；训练不持有 Endpoint 管理权限。ECS 可以等待 Training Job，但必须有持久状态和可恢复入口；没有必要常驻 cron。

TF Endpoint 镜像和在线执行代码来自独立 serving 仓库。本仓只指定确切镜像 digest 与制品 URI，不自行实现请求服务或业务特征获取。

## 3. 最小代码变更计划

路径相对仓库根，均为目标，禁止据此声明已经存在。

| 文件 | 计划 |
| --- | --- |
| `recommend_tf/train/daily_train_model.py` | 新增薄入口：解析 online_config/冻结日期证据，执行两阶段准备和公共训练，保存日更状态 |
| `recommend_tf/train/export_model.py` | 新增独立导出用例：校验成功产物、恢复选定模型、导出签名和包、输出一致性验证 |
| `recommend_tf/train/comm/datasvr/prepared.py` | 复用现有准备实现；必要时承接从 train_rerank 提取的公共准备函数，避免复制 |
| `recommend_tf/train/comm/datasvr/daily_dataset.py` | 必要辅助能力：日期窗口、逐日交付验证和合并 Manifest；无调度器、无模型算法 |
| `recommend_tf/train/train_rerank.py` | 小范围提取数据准备和运行身份/预算装配，离线三阶段默认行为不变 |
| `recommend_tf/train/experiment.py` | 保持训练后 test 语义；仅适配公共函数提取 |
| `recommend_tf/train/cloud_train.py` | 明确选择 experiment/daily；复用摘要检查与上传，不引入生产默认值 |
| `recommend_tf/train/models/rerank/` | 复用 PREDICT；必要时提取窄的模型导出适配/纯推理定义，不复制网络 |
| `recommend_tf/platforms/sagemaker/daily_pipeline.py` | 新增一次性平台编排、训练提交跟踪、步骤恢复；已有可用提交能力优先复用 |
| `recommend_tf/platforms/sagemaker/publish_model.py` | 新增推理包发布与部署适配，状态对账、更新 Endpoint 和回滚 |

不预建通用插件框架，不新增 `serving/`、Dockerfile.serve、自研 HTTP/C++ server。持久状态和制品验证可先以内聚函数实现；达到明确复用/复杂度需求时才拆文件。依赖仍由 pyproject/锁文件管理，不把纯文档计划变为新增依赖授权。

### 3.1 共用调用边界

```text
experiment.py → train_rerank.py ── splits=(train, validation, test)
                                   │
daily_train_model.py ──────────── splits=(train, validation)
                                   ↓
              公共输入冻结、数据准备、预算和身份生成
                                   ↓
                    training_pipeline.run()
          词表拟合 → 每轮训练/validation → checkpoint/selection
                                   ↓
        experiment 执行 test；daily 返回，由平台接 export/publish
```

公共函数名称最终随实施确认，不要求改造 train_rerank 成所有模式的统一控制器。`training_pipeline` 继续通过现有模型边界装配；不新增按模型名分支。

### 3.2 导出与训练解耦

导出入口只读取成功训练产物，不调用训练用例、不重新读取日更指针、不访问上游数据服务。模型专属适配负责恢复结构和权重；纯推理构图若需复用，应从训练专属逻辑中提取稳定窄接口，由训练和导出共同依赖，不让 exporter import 整条训练流水线。任何提取须保持变量名、shape 和现有 checkpoint 恢复一致。

## 4. 配置与数据 ready 合同

固定配方包含 online_config，其值只被 daily 消费。离线忽略窗口，仍按实验显式日期。日更不静默继承实验 test 日期，也不意外启用后来新增的 early_stop。

通用解析：validation 为截止 D 的连续 V 天，train 为 validation 之前的连续 T 天；本次 T=30、V=1。日分区时区来自版本化上游合同。31 天不齐全时停止本轮提交并记录缺失日期。

平台输入至少包括：环境绑定、配方 URI/digest、截止日期、上游发布引用和摘要。ready 证据必须代表完整发布，而不是任意文件事件。入口根据截止日枚举窗口，校验逐日发布合同、成熟性/完整性证据，冻结来源清单和合并 Manifest。训练仅接受冻结 Manifest；分片内容验证继续由训练 reader 完成。数据列、标签、mask 和过滤仍由配置声明。

[NEEDS CLARIFICATION: 上游现有成功触发是否已交付全部字段，实施前需核对接口；缺项优先扩展既有版本化发布结果，不额外构建 ready 数据库。]

同一配方、输入和模型身份得到稳定 operation key；训练 run 的时间戳不参与逻辑模型版本。重复同日同输入为幂等重试；同日不同输入必须作为显式修订，不能自动当作相同任务或无条件覆盖。

## 5. 推理交付合同

本仓交付的逻辑内容如下；具体 tar 目录必须与 serving 仓库确认，不能把本节示意当成官方容器目录：

- SavedModel：无标签预测图及模型权重；不包含训练 optimizer/recovery 作为加载依赖。
- Feature State：训练集拟合状态及其摘要。
- Feature Config：默认值、类型、变换、历史规则、字段映射；不要求推理消费完整训练配置。
- Signature：批量具名输入及 shape，按候选顺序的四任务输出，业务单位/转换及模型版本。
- Manifest：合同版本、逻辑模型版本、来源 run/epoch、配置/源码/锁文件/数据摘要、各文件大小和校验和。

编码在 serving 执行，但其语义与状态由发布包完整声明。图内编码随 SavedModel 执行；图外编码由外部镜像依据合同执行。不得导出依赖 Python 回调的图却声称可独立由 TensorFlow Serving 加载。TF 2.11 checkpoint → SavedModel → 指定 CPU 镜像必须实际验证，不能假设版本兼容。

外部镜像需提供 `/ping`、`/invocations` 等 SageMaker 合同适配和实际版本响应。精确 wire schema、格式、模型目录及图内/图外编码分界在 serving 合同联调前确认。本仓提供导出验证和黑盒镜像测试，不将外部服务源码复制进来。

指标含义须校验，尤其 watch_seconds 的最终输出单位/变换；不把 logits 当作业务结果。四任务列表来自签名和配方约束，辅助任务不对外暴露。

## 6. 发布与恢复设计

建议状态：`input_frozen → training → trained → exported → uploaded → deploying → deployed`，另记录失败、无可选 checkpoint、回滚中/回滚完成等结果。每个步骤保存产物摘要和确切资源名称；训练成功不因后续失败改写。

- 训练：确定性任务名与持久记录关联。提交超时先查询任务再重试，不创建第二个任务。
- 导出：验证 checkpoint 与 Feature State 同源；历史 selection 中的容器绝对路径必须解析为 run 内经校验的相对成员，不直接使用过期路径或允许路径逃逸。输出 create-only，临时目录完整后提交。
- 上传：先完整 payload，再提交 manifest/发布记录；已存在对象必须校验内容一致，禁止覆盖旧权重。
- 部署：新建不可变 Model/EndpointConfig，使用固定镜像 digest 和精确制品 URI。API 返回只代表已受理；DescribeEndpoint 对账并核验目标配置，随后实际无敏感样本调用与版本检查。
- 切换：蓝绿 all-at-once，新资源就绪后切流量；配置观察期和明确告警才能启用相应自动回滚。首次创建无旧 TF 版本，失败只能记录未上线，不能声称已回滚到树模型。
- 恢复：部署请求超时或 ECS 重启必须查询实际状态，避免把仍在执行的部署误判为失败；观察期结束并确认实际配置后记录 deployed。
- 发布互斥：持久认领采用条件写或等价机制；不能仅用进程内锁。平台 API 不接受 fencing token 时，禁止仅因租约到期就自动并行接管部署；须确认旧执行者已结束并完成状态对账。
- 回滚：用保留的旧配置恢复 Endpoint；显式回滚不降低自动发布的数据高水位，防止旧任务再次上线。
- 清理：保护在线、部署中、回滚目标及其模型包、镜像/资源引用；保留时长待配置确认，不直接套用旧树/Redis 清理器。

S3 发布版本与 Endpoint 在线版本分别记录；切换期间请求可能命中不同版本，响应版本为分析依据。只有模型包上传成功不能推进到 deployed。

## 7. 跨仓部署职责

| 责任方 | 实施边界 |
| --- | --- |
| doku-train | train/export/platforms 与制品、部署状态；负责每天更新 Endpoint |
| serving 仓库 | 固定可部署推理镜像及输入输出/制品合同；不由本仓实现在线逻辑 |
| doku-iac | 独立训练/部署/推理角色、网络、ECS 和上游成功触发；部署身份与训练身份分离 |
| doku-monitoring | 实际发布结果、失败、数据新鲜度、Endpoint 执行健康；日更无 test，不伪造 test AUC |
| 消费方 | 获取特征、实验分流、业务规则；本轮不改 mono |

IaC 和每日发布器不得同时争夺 Endpoint 当前配置的写权限/状态权威；具体 Terraform 资源管理策略在 iac 设计中明确。实验训练访问角色不直接扩权为生产发布角色。

## 8. 验证计划与实施顺序

先确认 requirements，再确认 design，然后生成 tasks/verification 并实施。当前仅提供验收计划，不声称测试已通过。

1. 日期和入口：合成逐日 Manifest 覆盖跨月/年、缺日、时区、重复事件；离线不应用 online_config；日更不准备 test。
2. 公共训练：相同冻结 train/validation、配方、种子和环境比较离线/日更的词表、逐轮指标、权重和选模；覆盖不可计算指标状态。
3. 导出：合成 checkpoint 的最终输出一致性、缺失/OOV/序列边界、批量顺序；校验损坏、错配、失败 run 和过期路径拒绝。
4. 平台：模拟提交超时、重复触发、重启恢复、旧任务晚完成、Endpoint 部署失败、回滚和清理引用竞态。
5. 外部推理镜像：验证声明合同、CPU 加载、模型版本及输出一致性，测吞吐/延迟/RSS；依赖外部镜像交付。
6. 云端：另行授权后核验指定 epoch 21 产物，首次发布及完整每日链路。实例、告警和容量目标未确认前不声明生产容量通过。

## 9. 官方机制参考

- [SageMaker Pipeline 调度](https://docs.aws.amazon.com/sagemaker/latest/dg/pipeline-eventbridge.html)：支持 EventBridge，但本期沿用现有上游成功触发 ECS。
- [蓝绿部署](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green.html)：all-at-once、观察及告警回滚机制。
- [自定义推理容器合同](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html)：由外部 serving 仓库承担。


## 10. 本地实施细化（2026-10-30）

Source: 用户确认设计并授权实施；对应 [任务](tasks.md)、[运行合同](operations.md)、[验证](verification.md)。

- 公共装配提取到 `comm/training_input.py`，train_rerank 保留三阶段薄包装，daily 调用同一函数指定两阶段。没有复制模型或训练循环。
- `comm/artifact_integrity.py` 提供成功 run 文件清单，上传最后提交 COMMITTED；历史 run 导出需外部审核清单。模型专属导出实现位于 `models/rerank/export_adapter.py`，不依赖训练流水线。
- 制品签名为组织/padding 后的无标签张量，外部 serving 负责原始请求编码；图内变换/词表/model 已随 SavedModel 导出。没有实现本仓 serving。
- 新 ready adapter 接口明确为 `doku-tf-dataset-ready/v1`，当前只支持既有 UTC 日分区；不宣称上游已接入。
- 平台条件状态与持久任务认领内聚于 publish_model；ECS 接管需确认旧 task 已停止，API 提交结果模糊时保持锁并要求对账，不用租约猜测。
- 历史引用全部保留，不实现未经确认保留期的自动删除；因此清理保护采取保守保留，不能声明已提供自动清理。
- 外部镜像/输入 wire 合同、IaC、告警资源及云端联调仍为外部依赖，不因本地实现完成而标记完成。

## 实施复审补充（2026-10-30）

Source: 本次实现复审与回归测试。

- 离线与日更入口只在成功创建输出目录后取得所有权；目录竞争失败不得修改另一运行的状态或封存记录。内存预算启动失败也必须清理采样器。
- 发布认领前检查 ready 数据；认领后的输入错误在调用训练 API 前记录 `input_failed` 并释放占用。已完成或已冻结训练请求的恢复不重新读取上游。
- operation 标识不包含轮询、总等待、下载和导出执行预算，允许增加这些预算恢复同一次操作；训练与发布配置仍参与标识。
- 导出校验冻结 Manifest、合同、Feature State 的 prepared identity 与配置/lineage 一致。导出器源码摘要参与逻辑模型版本，防止升级导出实现后碰撞旧制品。
- 不可变上传地址为 `<model_prefix>/<model_version>/<manifest_sha256>/model.tar.gz`；模型资源名同时绑定制品 URI，允许相同逻辑权重对应不同运行元数据。
- Endpoint 更新前重新核对捕获的旧配置；回滚只允许操作本次目标或捕获的旧配置，遇到外部修改保留占用并失败。平台自动回滚失败时继续尝试恢复旧配置，恢复完成前不释放占用。
