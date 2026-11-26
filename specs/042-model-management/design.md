# 设计

Source: Spec 042 requirements及2026-11-26用户确认；现有Spec027/036/038/039/040/041和实际实现。

## 当前状态

现有pipeline已提供冻结输入、持久化单模型互斥、确定性训练任务、导出、Endpoint探测及失败恢复。
但每模型ready ZIP内嵌配方，ECS重复固定配置引用；FYP发布器包含模型白名单；监控目录把模型名绑定唯一配方。Idea发布器假定唯一主路由。

## 目标与归属

模型登记及管理CLI由doku-rec-deploy负责；执行合同和TF runtime由doku-train负责；共享调度、权限和AppConfig基础设施由doku-iac一次性建设；多配置版本监控由doku-monitoring支持。跨仓只交换版本化JSON合同，不import另一仓实现。

稳定model_name独立于config_revision/run_id/model_version。配置版本为完整声明的摘要；配方算法摘要沿用recipe_sha256。登记包含任务组、完整配方、日更开关、模型服务配置和准入所有者。AppConfig是生效配置权威，S3保存不可变快照、执行请求与操作日志。发布使用比较并交换和操作ID，旧请求不解析新配置。

共享任务组调度器读取登记配置，为每个模型解析数据窗口，并将ready/config/binding及摘要冻结。创建请求时同时固定准入身份；重复数据截止日不因配置更新重复产生训练。worker仅接受可信任务组范围内的冻结请求。记录持久执行状态，ECS响应不明确时先查原任务，不盲目启动。恢复继续原版本；显式取消只能针对未开始请求，重建使用新的显式请求ID。

训练来源沿用ready_manifest与idea_mapped两种适配。首发可使用成功封存run或正常冻结输入训练。新模型服务发布仅更新其Endpoint和模型目录；不写消费者路由/AB。存量固定Endpoint可沿用现有特征合同、调用协议和合成探针。

## 配置与安全

管理合同先声明schema与允许操作。登记不接受任意IAM角色或任意任务定义；由受信任务组binding提供角色、镜像digest、CPU资源、网络、KMS、可读数据/制品范围及Endpoint命名范围。已有Endpoint接管是显式验证后的例外，不扩大前缀权限。配置更新不得改变model_name、组、Endpoint和管理权。

公共runtime发布与业务配方发布分开；本次不修改调用方。已有发布路径保留用于未接管模型，托管模型使用独立的通用固定Endpoint合同守卫，不走带业务路由副作用的旧publisher。

同名部署比较实际导出signature（忽略模型权重版本）及feature-config。不能以recipe_sha变化拒绝同名配置；服务合同不兼容则在更新Endpoint前失败。导出不可变产物及版本签名探针复用现有实现。

## 版本与恢复

registry配置发布和训练执行分离。每次请求记录registry/config revision、recipe摘要、固定数据、runtime和binding摘要。监控事件携带登记与配置证据，collector允许同名多个合法配方，但保留环境、任务组、服务身份和指标任务合同校验。历史事件不得把最新配置身份回退。

技术发布失败沿用旧EndpointConfig恢复；手动回滚同名版本不改下游路由。回滚不隐式停止日更：工具显示原开关，明确下一次成功日更仍更新权重。恢复历史训练配置是新配置版本，影响未来请求。

存量迁移分为计划、停止旧准入、空闲/待执行请求核验、登记、激活新准入、验证。每阶段持久化，可恢复。先FYP/Idea各一例；允许批量但每模型独立结果。若旧请求尚存在则等待或显式处理，不能丢失。旧入口未确认停止时不得激活新入口。不中途抢占在跑任务，不更新权重。

## 平台与监控

每任务组一个共享dispatcher和worker runtime；现有业务资源不自动替换。IaC新增模块默认不实例化生产。权限限定组内Endpoint/S3前缀和指定角色，不使用全账户无边界权限。代码所有权交给既有发布系统；不将模型epoch等参数写入Terraform。

新模型通过登记进入监控，不逐模型手写IaC。保留现有任务组失败告警与按model_name发现的指标，Endpoint资源监控使用可自动登记的受控机制。新增任务组须独立完成告警接线。

## 验证与发布边界

使用内存AWS仿真及SDK契约测试验证CAS、配置冻结、并发、重试及迁移；真实CPU smoke验证训练到导出合同。运行仓库必需检查。生产验收另行记录：不以本地测试代替AWS部署、网络探针或线上无变化证明。

## 操作合同与流程

```mermaid
flowchart LR
  U[用户自然语言] --> S[doku-rec-model-manage skill]
  S --> C[统一 CLI / JSON API]
  C --> P[配置差异 plan → apply]
  P --> R[AppConfig 模型登记]
  R --> Q[不可变请求：配置 / 数据 / runtime]
  Q --> W[任务组共享 ECS worker]
  W --> T[CPU 训练或成功封存 run]
  T --> E[导出与同名服务合同检查]
  E --> D[独立固定 Endpoint + 版本探针]
  D --> M[监控模型目录]
```

| 操作 | 持久结果 | 线上影响 |
| --- | --- | --- |
| register → apply | 独立名字、配方版本、合成探针；日更初始关闭 | 登记日志/告警，不训练、不替换权重 |
| configure / restore-config → apply | 新完整配置摘要、前后差异、发布回执 | 只影响新请求 |
| publish | 冻结源 run 校验和清单、注册快照及运行时 | 从源 run 选择并导出到该名字；未来日更使用注册配方 |
| train / 日更触发 | 固定 cutoff 的唯一请求 | 首次创建独立 Endpoint；后续仅更新同名 Endpoint |
| enable/disable-daily → apply | 日更准入开关 | 禁用后不接受新日期；原请求保留 |
| cancel / rebuild | 未启动请求取消；新请求替换原槽位 | 重建是显式操作，不覆盖旧快照 |
| recover | 已确认停止的 worker 对应原请求再次执行 | 配置、数据、运行时不变 |
| rollback | 原名字内历史成功版本的恢复记录 | 不改变日更开关和训练配方 |
| adopt / abort-adoption | 所有权日志与原入口快照 | 不更新 Endpoint 权重；未激活前可恢复旧入口 |

CLI plan 是可审查的完整差异。已授权操作可直接 apply，无二次审批要求。AppConfig发布未结束返回pending，重试同一个plan；并发/过期计划被拒绝。权重回滚后下一次成功日更仍可正常更新，工具明确展示此语义。

## 冻结、队列与恢复细节

- `registry-snapshots/`、`frozen/config|binding|ready/`、`groups/`、`requests/`、`run-inventories/`按内容摘要create-only保存。首发源run即使之后被错误覆盖，也不能悄悄换成另一个产物。
- 日更槽位为`slots/<model_name>/<cutoff>.json`；首发/回滚还要求显式operation_id。配置更新不导致同日重跑。`pending/<model_name>.json`在创建槽位前写入准入意图；调度只扫描待处理索引，不逐轮下载全部历史。
- 请求先CAS进入starting，再调用ECS clientToken。响应丢失继续原token；超过保守重试窗口则按startedBy查证，无法确认时报告需人工对账，不盲目重提。一个模型最多一个已提交请求，多个模型各自独立。
- 正常STOPPED的submitted/running或优雅deferred请求可由共享调度恢复；程序明确失败须显式recover/rebuild。worker、pipeline各自保留请求与发布日志互斥。查询不到旧执行器状态不能声称已经停止；优雅暂停已持久化释放执行权时，可依据该证据恢复，不依赖过期的ECS STOPPED记录。
- 请求同时冻结整个组runtime，升级控制器后仍用原worker revision、原镜像和原binding。平台升级须保留旧revision及权限，直到其请求全部终态；业务配置更新不涉及这一步。
- 源run首发的训练参数属于源run自身；不能把它的指标伪称为未来日更配方训练结果。导出头、日期和实际服务合同仍校验；首发源run可与未来日更窗口不同。

## 存量接管细节

计划读取旧任务定义中固定config/binding摘要，验证release journal、实际EndpointConfig、ModelDataUrl及制品manifest中的signature/feature-config。保持原Endpoint、部署参数、合成探针、state/work/model路径；推理角色必须符合预先登记的组边界。

只暂停专用旧worker rule，并移除旧模型在共享上游/恢复规则上的专属target，保留其他模型target。旧target必须有明确投递最大重试窗口。暂停后等待这些投递耗尽，再查旧family没有活动执行器、没有未消费请求、journal与计划一致及Endpoint仍InService；其间不激活新准入。若journal发生变化，撤销后重新计划，不忽略变化。旧入口没有完成停用，即使登记已写入，ownership门禁仍拒绝新调度。

默认先验收FYP与Idea各一个再批量，批量按模型串行并保留每项计划，pending时停止推进后续项。中断恢复重试原计划。abort在新所有者激活前恢复原target和开关；已激活则不能借abort绕过新请求排空。

## 平台落地和验收顺序

1. 审核组级资源、基础角色、精确S3前缀、告警主题与已有接管入口；运行时和资源登记器只做一次平台发布。
2. 创建共享控制器和空登记，保持调度硬暂停；接入collector的managed_groups及snapshot只读权限。已有生产模块不自动实例化/迁移。
3. DTE验证新增名字首发、同名日更、配置15轮只影响新请求、故障恢复、取消重建、显式回滚；确认调用方/AB无变化。
4. 审核PROD计划，授权后启用共享触发并逐个接管。验证旧/新入口互斥、原模型权重没变，随后观察下一次日更及指标。

Source: 用户确认的Spec042需求；实施落点为本仓model_management、doku-rec-deploy管理skill、doku-iac/model_management_group和doku-monitoring/model_metrics_collector。以上生产步骤尚未执行。
