# Spec 009：树模型每日训练与发布设计

状态：2026-10-13 用户授权实施；本仓已提供 CPU、本地/S3 发布入口，云端与 mono 接入未验证。
需求见 [requirements.md](requirements.md)。

Source: 2026-10-13 用户确认的日更、命名、切分、指标门禁、导出、发布、清理和版本选择方案。
下文具体内部字段、文件布局和并发协议若超出会话确认，标记 [INFERRED]，不冒充现有实现。

## 1. 实施前状态与来源（历史核查）

- 当前 model_name 是 rerank/rerank_tree 插件选择器，尚无独立业务模型名称。
  Source: recommend/train/comm/config.schema.yml、comm/model_params_validator.py。
- TreeRunner.fit 集成训练、validation 早停、checkpoint 序列化/保存和 validation 报告；
  CheckpointAgent 已隔离文件 I/O；training_pipeline.test_run 能加载 checkpoint 独立测试。
  当前选模 unavailable 会阻止测试，需要避免与目标“记录可用产物、独立发布判断”混淆。
  Source: recommend/train/models/rerank_tree/runner.py、comm/checkpoint_agent.py、comm/training_pipeline.py。
- ModelExporter 仅接口；没有树模型推理包、S3 发布清单或清理程序。
  Source: recommend/train/comm/model_exporter.py。
- Finder 有独立 eval_rerank.py 和 export_rerank_model.py；训练 hook 调 CheckpointAgent 保存，
  没有必要另设保存批处理。Finder 的 comm/training_pipeline.py 是输入流水辅助，不能误认为
  本仓顶层编排器。仅借鉴边界，不引入 TF/session/MPI/私有存储。
  Source: finder-train/recommend/train/{train_rerank.py,eval_rerank.py,export_rerank_model.py}、
  comm/{session_manager.py,checkpoint_agent.py,training_pipeline.py}（相邻只读参考仓库）。
- 本地 IaC 声明上游每天 06:00 UTC（北京时间 14:00）启动，处理前一 UTC 日；
  join 后执行 --project-training，投影成功发布后衔接训练，不能以固定延时猜测完成。
  这只是本地声明，不是本次核实的云端部署事实。
  Source: doku-iac/envs/prod/fyp_rank_samples.auto.tf:150、modules/ecs_scheduled_task/main.tf:65；
  doku-offline/fyp-rank-samples/src/fyp_rank_samples/{cli.py,daily_dataset.py}。

## 2. 组件与执行边界

```text
上游训练投影成功发布
  → iac 触发单次 run.py
  → 冻结输入/解析日期/去重
  → train_rerank.py（validation 早停 → CheckpointAgent 保存）
  → eval_rerank.py（加载冻结 checkpoint → test 报告）
  → 发布指标门禁
  → export_model.py（原生模型包 → 加载和预测一致性检查）
  → S3 完整上传 → 发布清单提交

独立每日调度 → 清理入口
显式运维入口 → 回滚到保留的成功版本
```

run.py 复用现有 composition root，入口调用用例接口，不拼 shell 字符串，不包含特征算法。
train_rerank.py、eval_rerank.py、export_model.py 只做用例装配；模型适配器负责内存中的
训练/预测/序列化，CheckpointAgent 负责 checkpoint I/O；公共评估组件负责指标。
训练中早停不可移成训练结束后的独立任务；训练后 test 可以独立重跑。
导出器只读取成功产物及公开合同，不 import 训练 runner 内部实现。
[INFERRED] 发布、回滚、清理独立用例与存储接口，统一由 run.py 暴露命令；具体命令参数在实现设计细化。

## 3. 配置与冻结输入

model_type 为配置指定的模型插件，model_name 为配置指定的业务名称，两者不混用。
迁移所有 schema、工厂、搜索、报告和 fixture 中旧字段语义，不保留同字段双重解释。
独立实验配置仍不随源码发布；测试使用合成配置。

dataset 仍仅声明 manifest_uri，不恢复 aws_profile、adapter 或 daily_prefix 等旧配置。
[INFERRED] 日更用例的窗口策略由 YAML 声明，先在上游已完成的版本化交付内解析可用日期，
生成明确的日期列表和精确冻结 Manifest，再传给训练用例。禁止扫描业务存储补数据，
禁止训练直接消费可变 latest/daily 指针。日期列表不能用 min/max 连续范围替代而重新要求缺日存在。

设 test 日为 T，validation 为 T 之前最新可用日 V，训练日期为 [V−30日, V−1日]
内所有可用日。至少一个训练日，V/T 各一天；三段按 UTC 事件时间不重叠。
没有足够日期则跳过本次流水线，不降级、不覆盖旧版本。train-only 拟合特征状态。
每个阶段复用冻结输入、resolved config 和 checkpoint；目录日期是 T，不是启动时间。

### 3.1 已查明的交付事实

Source: `doku-offline/fyp-rank-samples/src/fyp_rank_samples/daily_dataset.py:20–89`、
`cli.py:165–193`；本仓 `recommend/train/comm/datasvr/daily_input.py:15`、
`published_dataset.py:45–99,144–160` 与 [Spec 007](../007-manifest-only-dataset/design.md)。
2026-10-13 只读核查本地源码，未读取生产 S3，也未验证云端部署。

- `publish_day` 只构建一个日期，写入 `daily/environment=<env>/date=<D>/run_id=<id>/`，
  再通过条件写更新该日 `_SUCCESS`。标志是 `doku-train-day/v1`，携带日期、不可变
  manifest_uri、Manifest 文件 SHA256 与 projection_sha256；不是多日数据集。
- `join --project-training` 在投影成功后输出 published，但忽略 `publish_day` 的返回 URI；
  当前 join 状态输出不包含精确投影 Manifest，不能假设已存在可直接使用的完成事件合同。
- 历史日期目录及其 `_SUCCESS` 构成现有日级目录协议；未找到单独的版本化多日索引。
  `build-dataset --dataset-version` 能重新投影一个日期范围，但不是复用既有每日产物的冻结器。
- 本仓已有独立 `resolve_days`：校验日指针、投影与合同，合并分片引用、记录 source_manifests、
  生成 content_sha256 并重查日指针。它返回内存对象，尚无单次日更的发现/上传入口。
  它接受显式日期列表；缺日选择、版本信息从 Manifest 派生、原始证据归档需要补齐。
- 当前 S3 reader 可以引用不同日不可变目录的分片；`--local-root` 镜像模式却要求分片在
  Manifest 同前缀。多日引用的本地验证不能直接沿用这个假设。

### 3.2 建议采用的多日冻结交付协议

[INFERRED] 将“多日冻结”放在 **doku-train 的训练前置准备阶段**，复用本仓辅助函数；
上游继续只生产单日投影，不重算 Parquet、不复制上游实现。iac 只做触发和传递上下文。
这细化已有窗口需求，不改变训练入口 `dataset: {manifest_uri: ...}` 的唯一配置合同。

1. iac 仅在投影完成成功后调用准备入口，显式传环境、日投影根路径、触发数据日期、
   冻结输出位置以及实验配置。这些属于准备/调度参数，不塞回 dataset.daily_prefix。
   为精确关联触发，建议 offline 的 published 结果补充 manifest_uri、文件摘要、日期；
   尚未完成该跨仓变更时，准备阶段可按触发日期读取 `_SUCCESS` 并记录本次实际读到的版本，
   但不能声称它一定是原触发运行产出的版本。
2. 在该环境日根目录分页发现日期键，读取 `_SUCCESS`；成功标志之外的孤儿 run 不算可用。
   仅考虑不晚于触发数据日期的日记录，避免延迟任务越过自己的截止日。找到最新两个非空
   有效日 T/V 后，只选 [V−30日,V−1日] 中的非空有效日训练。缺失日期/合法零行日记录为
   未采用；403、读取超时、坏摘要、非法合同不是“缺一天”，本轮失败而非静默跳过。
3. 日 Manifest 的 dataset、投影、feature/label/label_config 合同必须一致；以选中实际
   Manifest 派生预期身份，不把旧版本字段重新交给用户填。源合同不兼容则失败，不能通过
   混合训练绕过；这与“不同模型版本允许同名合同变化发布”是两种不同边界。
4. 一次性固定所选日 URI/摘要和合同，重读已选指针发现切换则重试准备阶段。保存日期发现
   结果、遗漏日期、指针原文、各日 Manifest 原文与摘要；不承诺跨天全局事务快照。
   冻结后上游补数只影响后续运行，不改变本轮输入。
5. 生成 `doku-train-dataset-manifest/v1` 多日 Manifest：保留原分片 URI/摘要，汇总行数、
   时间范围和 source_manifests，生成逻辑摘要；相邻写 contracts/features.json、labels.json。
   写到本仓显式输出前缀的不可变冻结目录，先合同和证据、最后 Manifest；全部校验后才把
   精确 S3 manifest_uri 交给 train。输入指纹一致可复用已校验冻结对象，不能覆盖旧对象。
6. train 保存 run/input 冻结副本及明确日期列表；test/resume/export 复用，绝不重新查 daily。
   源 Parquet 不复制，输入读取权限仅覆盖批准的源前缀；上游保留期须覆盖本轮读取及需要的
   重跑期限。保存 Manifest 不等于永久保存原始样本，不承诺源数据被删后的重训可用性。
7. 本地合同测试用 URI→内容的测试存储映射覆盖跨前缀引用；不得为了测试扩大任意本地路径
   访问权限。生产冻结目录和源日目录的权限由 iac 装配。

职责结论：offline 单日交付；train 前置冻结及训练；iac 成功后衔接。精确完成上下文属于
跨仓待实施合同，不再作为未决设计选择；本次不修改上述仓库的代码。

## 4. 训练、评估与发布状态

[INFERRED] 阶段状态分别记录：training、evaluation、export、publication，各自有成功、失败
或适用的跳过/拒绝状态。训练成功后，评估失败、指标拒绝、导出失败或上传失败均不得
修改训练成功状态；整体流水线只有清单提交成功才声明“已发布”。

validation 选轮数指标与 test 发布指标在 YAML 分开声明。没有相对下降判断、隐藏阈值，
不因合同不兼容额外拒绝发布。训练任务缺有效标签是执行失败；指标不可计算记录 null、
原因和样本统计，并继续计算其他任务，最后由发布门禁作出整包结论。

regression_auc = (同序对数 + 0.5 × 预测平局对数) / 真实标签不同的对数。
在整个测试集按各任务有效性 mask 计算，不按用户分组；真实标签相同的对排除。
[INFERRED] 依据已确认的等权样本对公式，发布指标不加样本对权重。真实标签/预测值的尺度按任务合同转换后计算。
[INFERRED] 用精确排序计数算法实现，避免 O(n²) 枚举或静默采样；用小样本穷举结果验证。
分类 ROC AUC 和 regression_auc 各自明确命名；≥0.60 通过，null 或 <0.60 拒绝整包。

## 5. 导出合同与版本布局

目标布局（[INFERRED] 文件名，目录语义已确认）：

```text
<artifact-root>/<model_name>/<test-date>/<model_version>/
  manifest.json
  models/<task>.ubj
  feature-contract.json
  fitted-feature-state/state.json
  metrics.json
  lineage.json
<publish-root>/<model_name>/publication.json
<training-root>/<run_id>/...  # 内部训练产物，独立于 serving 包
```

manifest 声明合同格式版本、model_type/name/version、test 日期、创建时间、来源 run/config/
源码/依赖/数据摘要、任务文件与所有 payload 校验和。manifest 自身摘要由发布清单保存，
避免自包含哈希。各任务显式声明输出名称、类型、单位及 sigmoid/逆变换等还原语义。
feature-contract 固定输入顺序、类型、缺失/OOV/类别编码与状态引用；fitted state 保证
调用方不重建词表。合同变化允许自动发布，但必须记录变化，便于调用方排查。

### 2026-10-20 派生规则交付修复

Source: 用户确认补齐导出合同方案；指定版本
`320d4acdadfb4da1b9226f721f3c7cae` 的 S3 只读核验与 Python/Go 对照，
见 [验证](verification.md#2026-10-20-派生规则交付修复)。

当前本地实现：`doku-tree-features/v1` 的 `encoding` 完整携带 `features`、
`feature_default_values`、`feature_transformations`、`hist_options`、`history_scalars`、
`input_features`、`tree_inputs`。`history_scalars` 原样取自同次冻结训练配置；
没有派生规则时写空对象，不推断规则、不增加模型输入。列顺序、类型、权重和 fitted state
语义不变，manifest 中合同摘要随内容重新生成。历史包遗漏该字段，不能据其 format 相同
推断消费者已支持完整合同。本次保留 format 名称，部署必须协调消费者能力。

目标消费状态：正式合同加 fitted state 足以重建输入，不依赖训练归档。服务端
`CompileArtifact` 和 `ArtifactCompiler.Compile` 都需支持合同自带的规则；完整合同优先，
旧包才允许按配置摘要及身份核验后补载归档，不得覆盖合同中的规则或自行补零。
现有五种历史算子无需新增；必须继续验证依赖、输入列顺序与类型。

常规切换顺序为先升级消费者，再用新制品版本发布；不覆盖任何旧版本文件。
用户后续明确授权本次直接切换 DTE/PROD 当前模型，并临时开启 DTE 发布与 Redis 开关；
本次制品操作及验证见 verification.md，不能将制品切换等同于消费者已适配或训练镜像已部署。
切换及旧包回滚依赖解除后再删除 `training_config_files` 和文件。可从成功训练产物
重新导出，不要求重训。回归测试使用独立合成 fixture，分别覆盖非空和空规则，
从实际导出合同与 state 重建加工并比对训练向量和独立预期值；此 Python 测试不替代
Go 集成验证。现有 `synthetic_encoded_cpu` parity 仍只证明模型序列化一致性。

版本文件 create-only。发布前重新加载导出的原生模型，用合成/受控验证输入比较训练
checkpoint 与导出模型最终输出，不仅比较 raw margin；不将真实敏感样本装进发布包。
[INFERRED] parity 容差、模型格式兼容版本和具体合同 schema 在实施前固定并加入合同测试。
不能把 Python parity 冒称为已验证 Go/C++ 消费者；后者由 mono 集成验证。

## 6. 清单、去重、并发与消费

[INFERRED] 每个名称使用单个权威 publication.json，同时承载 current 和按数据日期的
已发布版本引用，以一次受条件保护的提交避免多个权威指针部分成功。每个引用包括
version、manifest URI/digest、发布时间；消费者不按目录排序猜最新版本。
若以后拆分清单，必须重新定义一致性协议，不能逐文件更新后假设原子。

去重身份使用冻结输入摘要、有效配置、源码和执行环境/依赖身份，排除 run 时间、输出路径等
非语义字段；显式重跑产生新版本并记录原因。同名流水线串行由外部编排与本仓发布保护共同保证。
[INFERRED] 发布提交使用预期清单 revision 的条件写；冲突不能无条件覆盖，旧数据日的自动
发布不能替换更新数据日 current。显式回滚是例外，并记录操作来源。
上传结果不明确时校验完整版本与权威清单后恢复，不生成虚假的重复成功。

业务调用方只提供 model_name + 日期，公共组件解析一次 version，以同版本状态编码，
自动传递版本给 predictor；服务按请求版本预测，缺失版本明确失败，不替换为 latest。
这些是交付合同，不在本仓实现 RPC。日期清单仅指向成功发布版本；历史已清理版本返回不可用。
当前版本回滚可指向仍保留的成功包，不更改版本文件，不重新训练。

## 7. 保留与清理

以可信创建时间计算年龄，超过 30 天才进入候选；数据日期不作为保留年龄。
当前发布版本及其必要依赖始终保护；被替换版本从替换时起至少保护 24 小时，
两条规则与 30 天保留共同生效。训练产物、导出产物、失败残留纳入统一清单化管理。
活动训练/导出/上传不可删除；保护当前发布包依赖的 feature state 与 lineage 等对象。

### 7.1 已查明的能力边界

Source: `doku-mono/docs/predictor-rpc.md` 的交付范围、调用方职责与模型更新约定；
`doku-mono/predictor-svc/idl/predictor-svc.thrift`；本仓 `comm/model_exporter.py`。
当前只有 Predictor 契约，没有 S3 加载器、活动模型登记或释放信号。旧 RPC 约定甚至不传
每日版本，与本 Spec 的固定版本目标不同；必须由 mono 后续适配，不能声称已具备保护能力。

24 小时只是最短过渡期，不是“没有使用者”的证据。S3 文件最后访问时间、服务失联、
两次读清单或逐实例独立 heartbeat 列表都不能单独消除“读后新增引用”的删除竞态。

### 7.2 建议采用的活动引用协议：持久 pin + 单清单 CAS

[INFERRED] 首版用既有每名称 `publication.json` 作为唯一协调记录，新增每版本生命周期、
活动引用 pins 和写入 operation_id/revision。发布、回滚、引用登记/释放、清理状态变更
全部条件更新同一个对象；不增加 Redis/数据库锁，也不以可过期 S3 锁文件充当安全保证。
S3 `If-Match` 比较原 ETag，首次创建 `If-None-Match: *`；冲突重新读取并重算操作。
ETag 只作并发令牌，不当文件 SHA256。每次写入使用新 revision，避免相同内容恢复造成 ABA。

Source: [AWS 条件写官方说明](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html)。
该机制提供单对象条件提交，不提供跨对象事务；安全性来自所有参与者遵守下述状态机。

- pin 包含 owner_kind（训练/导出/上传/消费者）、实例启动唯一 owner_id、版本或 run 引用、
  创建时间和 operation_id。实例按加载的版本登记，不每个 RPC 写 S3。
- 调用方编码组件及 predictor 都须 **先成功登记 pin，再下载/使用该版本**。注册只允许
  可用且未 draining 的版本；CAS 冲突重查。如果版本已退出可用状态则明确失败或重新解析
  尚未编码的新请求；已完成编码的请求绝不换版本。
- 本地每个请求继续固定版本。消费者停止接入该版本的新请求、等待本地在途请求和下载结束、
  释放模型/特征状态后，才释放 pin。调用方和 predictor 各自的 pin 都必须释放。
- pin 不自动过期。失联或心跳超时只告警，不能证明进程已停止；崩溃后由运维核实该实例
  已终止且不会恢复使用旧引用，再显式释放孤儿 pin。新进程使用新 owner_id。
  这是首版以可能多保留存储换取不误删，后续若引入租约必须同时设计消费者自我停止机制。
- 活动训练/导出/上传同样先注册 run/version 保护，完成后释放。源数据准备与本地文件
  也需执行所有权检查，不能把正在运行的目录当成失败残留。
- 引用协议不由本仓实现 mono RPC；本仓提供记录合同及管理入口，mono 负责实例登记/释放。
  **消费者尚未完成接入时，清理默认仅 dry-run，不自动删除可能被服务使用的模型产物。**
  不能将“尚无登记功能”解释为“当前无活动引用”。

### 7.3 清理状态机及发布/回滚竞态

[INFERRED] 版本生命周期为 available → draining → deleting → deleted；deleted 保留紧凑
审计墓碑，同一 version 永不复用。每次转换在同一权威清单上 CAS。

1. 超过 30 天、不是 current、满足原替换 24 小时保护后，CAS 标为 draining，同时撤销
   指向它的按日期可解析引用；设置 drain_started_at。已持有 pin 的实例继续受保护，
   但禁止新 pin/新解析，通知使用方迁移。draining 后额外留足 24 小时以覆盖历史解析缓存；
   这可能使实际保留期长于 30 天，符合“超过 30 天进入清理候选”而非精确 TTL 删除。
2. 消费者在 drain 期间停止接入该版本并排空释放。清理只有在 drain 过渡期结束、pins 为空、
   仍非 current、无活动 producer 引用时，才 CAS 转为 deleting。缓存老清单的注册者
   因 ETag 不匹配而失败，不能越过删除检查新增引用。
3. 回滚 available 版本正常提交；回滚 draining 版本可原子恢复 available 并置 current，
   同时取消 drain。与删除 CAS 竞争时仅一方成功：若已 deleting/deleted，回滚拒绝，
   不允许恢复半删目录。清理进程不得用旧计划绕过状态重查。
4. 只有观察到 deleting 才按已登记 manifest 对象集逐个删除，记录结果并支持中断重试；
   失败保持 deleting，所有发布/引用入口仍拒绝使用它。完成后标 deleted。版本文件不可变、
   键不复用，因此旧删除任务不会误删后来同名新版本。
5. 历史清单引用先失效后删除，不能回退到目录里的次新文件。当前版本的完整依赖始终保护；
   若共享训练输出/冻结输入，删除须检查所有版本与 run 引用，不能只看某一个版本过期。
   未登记所有权、无法判断引用的对象默认跳过并报告，不以年龄推断安全。

每次清理支持 dry-run，输出候选、保护原因、引用和计划摘要。只作用于显式本项目前缀；
不删除上游投影数据。iac 不得给受保护模型前缀配置无条件 30 天 S3 Lifecycle 过期，
否则会绕过 pin/current 保护；桶版本化时旧对象版本回收也须遵循同一保护边界。

该建议闭合了活动版本保护设计：有引用不删、失联不自动释放、CAS 使注册与删除互斥。
消费者拒绝迁移时可以长于 30 天保留并告警，不假装完成清理。尚需跨仓实施和合同测试，
但无需再由用户选择锁或判断服务是否“活跃”；本次只补充设计。

## 8. 与旧规则的关系与后续验证

本 Spec 扩展本仓到树模型 serving artifact 与 S3 发布，取代“只保留 exporter 接口/不更新
active”在该范围内的旧限制；训练产物仍 create-only，训练内不加质量门禁。
新增的是独立发布门禁，不恢复 Spec 001/003 的旧门禁或强制 refit。
Spec 007 的冻结输入与单 Manifest 入口继续有效；Spec 008 的 DNN 平台独立。

后续测试覆盖需求文档第 7 节，重点包括：并发提交及清理竞态、同名不兼容合同仍正常发布、
重跑不改 checkpoint、版本单次解析合同、空/平局回归指标、缺日列表、仅两天跳过、
导出前后最终输出一致性、失败不改变 current。仅完成文档检查不代表这些实现验收通过。

部署前另行填写环境、S3 前缀、身份、CPU/内存/超时等参数并获得云端执行授权；
这不阻塞当前文档整理，也不在本次创建 tasks.md 或修改实现代码。

## 9. 本轮设计补全验收矩阵（后续实现，不是已通过测试）

| 场景 | 预期结果 |
| --- | --- |
| join 成功但无 URI 字段 | 准备入口按日期解析并记录实际版本，不冒称原任务版本 |
| 多日缺日/补数/重复触发 | 冻结精确源集合；缺日允许；已选指针变化重试；输入一致去重 |
| 403、坏摘要、跨日合同不一致 | 准备失败，不当成缺日跳过 |
| 冻结后源指针更新 | test/resume 仍使用原冻结引用 |
| 注册 pin 与 deleting 并发 | 仅一个 CAS 成功；不得注册到 deleting |
| 回滚与清理并发、清理中断 | current 不删；deleting 不可回滚；重试只处理原不可变版本 |
| 实例失联、consumer 协议尚未接入 | pin 保留并告警；未接入时模型删除仅 dry-run |
| 历史日期仍被加载、30 天到期 | draining 后等待排空和保护期，不靠 TTL 强删 |

Source: 2026-10-13 本轮只读源码核查与设计推导。仅修改 Spec 文档，没有运行上游任务、
上传 S3、创建协调资源或修改 mono；AWS 能力来源见第 7.2 节。

## 10. 本仓实现与消费边界（2026-10-13）

Source: 用户确认 S3 方案并授权实施；实现文件与命令见 README、tasks 和 verification。

- 配置 `model_type` 选择实现，`model_name` 指定业务身份，`daily.training_window_days`
  声明窗口；`publication.<task>.metric/minimum` 声明发布指标。任务声明 `output_unit`，
  原生预测以 raw margin 加 `predict_fn` 解释；不会臆造未声明的逆变换。
- `eval_rerank.py` 独立评估，临时编码目录可重跑；test-evidence 绑定 checkpoint、配置和
  指标摘要。`export_model.py` 独立进程，校验 feature state/checkpoint，输出原生 UBJ，
  以 8 行合成编码探针验证加载后的输出一致性（rtol=1e-6、atol=1e-7），不是 Go parity 证据。
- `comm/daily_publication.py` 前置冻结、独立训练/评估/导出及上传；任务窗口来自 YAML，
  环境与存储位置来自显式调度参数。当前触发衔接采用按 cutoff 日期解析日指针的已定义回退，
  精确完成 URI 仍由 offline/iac 后续接线。缺数据返回 skipped，门禁返回 rejected。
- `comm/artifact_store.py` 提供 file/S3 CAS，`comm/publication.py` 管理注册、pin、current、
  dates、回滚及清理。首版 publication 与 artifacts 共用一个显式根前缀；文件为
  `models/<序号>.ubj`，manifest.tasks 显式映射业务任务名，不将用户任务名拼入文件路径。
- 模型清单外另登记 `training/` 归档。失败/拒绝 run 使用 diagnostic.json 登记，不可发布。
  仅清理明确登记对象；本地清理跟随远端 deleted 墓碑。共享冻结输入原件保守保留，不扫描
  删除上游数据或任意未知目录；清理未知孤儿需要人工确认所有权后处理。
- 同名 active_run 是持久所有权，不以超时自动解锁；崩溃恢复须确认旧进程停止。失败上传
  可在原 owner 结束后接管相同内容的 uploading 版本；已发布版本只可回滚，不重新覆盖。
- 请求路径不访问 S3：消费者后台读清单/下载并注册实例 pin，业务请求在本地已就绪映射
  中固定一个版本。S3 current 是已发布版本，不是已在全部服务实例可用的证明。
  多实例就绪协调和本地缓存属于 mono；本仓没有创建这些服务，也未添加 Redis 模型存储。
- native XGBoost 导出与训练/评估分别进程隔离，避免 macOS OpenMP 与 PyTorch 混用；
  Python 及平台身份加入 lineage/去重依据。不增加依赖，不引入 GPU。

此前“本次只写文档/不生成 tasks”的陈述记录设计阶段范围，已由本次实施授权取代。
所有云端写入仍需显式授权，本次验证只用合成 fixture、file 存储和 S3 SDK stub。

## 11. 发布与清理审查修正（2026-10-13）

Source: 用户要求重点审查 S3 发布、版本一致性与清理；本次故障注入测试及修复。

- CAS PUT 响应丢失（包括 SDK 重试后返回 412）时，读取 revision 确认是否已提交；不可变文件写入异常后按完整内容确认。
  无法确认时仍失败，不猜测成功；finish 以清单内已发布事实恢复 run 状态，避免把已提交
  版本记录为 failed。清单已被其他操作更新而无法确认本次 revision 时仍需显式恢复。
- manifest 最后上传，清单仍在所有登记文件读回校验后提交。拒绝保留路径、缺少任务文件及
  不完整包；归档目录须匹配导出配置摘要及 run_id，不能附加另一个 run 的文件。
- 独立 publish 新增 --run-dir，用于恢复中断的日更训练归档；已有归档可以经校验复用，
  缺文件且未提供 run 目录则明确拒绝。不缩减原先登记的对象集合，也不将部分上传视为成功。
  已发布的重试会重新验证远端对象，已删除版本不返回“成功跳过”。
- 自动发布以 (数据日期, 创建时间) 为高水位，拒绝同日更旧包；显式回滚不降低高水位，
  因此迟到自动任务不能覆盖回滚后的状态。此规则不增加特征合同兼容性门禁。
- 多个清理进程完成同一不可变版本时允许幂等 deleted，保持 deleting 的禁止回滚/pin 规则。
- 本地训练目录按 run_id 汇总所有版本引用；任一未删除版本仍引用时不删。删除前 CAS 登记
  retired_local_runs，阻止后续版本重新登记该 run；状态改变导致竞争时跳过而非强删。
- 本地 test/export/上传归档持有 run 共享文件锁，清理取得非阻塞排他锁后才退休并删除。
  锁文件在 run 目录外，避免删除后产生不同锁 inode；活跃读者存在时保留，不提前退休。
  未登记目录仍保守保留；外部读者必须遵守同一锁或 pin 协议。

### PROD 编排与容器接入（2026-10-13）

Source: 用户确认接在投影成功之后，PROD 启用，DTE 不新增定时训练；独立北京时间 20:00 清理。

- iac 保留 PROD 06:00 UTC 的上游日任务，`join --project-training` 发布完成且容器退出为 0 后，
  在同一 Standard workflow 的上游重试范围外启动训练 ECS。`--cutoff` 从原始 Scheduler 时间减
  24 小时生成；不使用训练开始/重试时的墙钟。训练无自动重试，不回到上游；异常锁不自动清除。
- 独立 12:00 UTC 清理 workflow 调用 `recommend.train.scheduled cleanup`，等待退出码，
  不自动重试；消费者协议未就绪仍 dry-run。DTE 不创建对应调度或训练资源。
- `recommend.train.scheduled` 是容器边界适配器。环境显式提供输入、冻结输入、产物前缀、
  模型名、配置 S3 URI/sha256 和 KMS key。配置摘要与 model_name/model_type 校验后才调用日更用例。
  清理不需要读取训练配置。结构化完成日志区分 daily/cleanup，异常返回非零并输出 completed=false。
- 生产配置以本地 `daily-expanded-20261013/configs/xgboost.yaml` 为基础，保留特征和树参数；
  将实现键迁移为 model_type，业务名为 rerank_tree，窗口 30 天，四任务发布阈值均为 0.60，
  watch_seconds 补 regression_auc 和 seconds 单位，二分类为 probability。本地生成配置位于
  `runs/prod-daily/config.yaml`，不随源码发布。实际使用的摘要在 iac tfvars 声明并随部署审查。
- 写 S3 时显式传递 `DOKU_TRAIN_S3_KMS_KEY_ARN` 对应的 SSE-KMS 头，同时保留条件写。
  针对前缀限制 ListBucket 权限下的 GET 403，只有授权的精确前缀 LIST 证明对象不存在才映射
  FileNotFoundError；对象存在或 LIST 被拒绝仍失败，不能把权限问题当成空发布清单。
- 独立训练/清理角色：训练能读输入和指定配置、写冻结输入及本模型制品，无删除权限；清理只可
  读写/删除本模型前缀，不可删除共享冻结输入。模型前缀不匹配数据湖按天过期规则。
- 训练仓提供 CPU 镜像构建与 PROD task definition（PSM doku-reco-train，4 vCPU、16 GiB、100 GiB 临时盘）；
  模型线程仍由 YAML 显式指定。训练 deadline 初始为 8 小时，是执行上限，不是吞吐 SLA。
  本地 Run/缓存位于 /work，任务结束临时盘销毁；远端模型和训练归档由发布流程保存。
- 镜像携带构建 Git revision、实际源码和 uv.lock，source_identity 在无 Git 镜像中仍计算源码及
  依赖摘要。镜像不包含本地实验数据或配置；依赖仍使用 CPU Torch/XGBoost。
- 上线顺序：准备精确配置并按摘要路径上传（携带 KMS 头）、先以 tree_publication_enabled=false 创建必要 IAM/日志资源、生成 binding、
  由 dkd 构建注册训练 task definition，再启用含训练节点的编排及清理调度。iac apply 与云端验收
  需另行执行；不能在 task definition/config 尚未就绪时切换正在开启的上游 workflow。

AWS GetObject 对不存在对象的 403/404 行为依据：
https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html 。

### 接入审查补充

- PROD `tree_publication_enabled` 默认 false，防止首次普通 apply 把已有日投影接到尚未准备的 TD/config。
  显式启用后才串接训练并开启清理；DTE 不新增这些资源。
- 清理固定使用同步工作流；开关只控制调度状态，不能切换成直接 ECS 或重建工作流。
- dkd PSM 必须是三段式，统一使用 `doku-reco-train`，TD family `doku-reco-train_prod`。
  容器默认无业务命令时失败，不能用 `--help` 的退出 0 冒充运行成功。
- Scheduler 在同步模式只获 StartExecution；ECS RunTask/PassRole 留给工作流角色。
  训练角色 ListBucket 的模型范围与其 Get/Put 范围一致，限制为配置 model_name。


### Redis 分发扩展（2026-10-14，Spec 010）

Source: 用户要求保留 S3，并新增 Redis 模型/特征产物分发。
[Spec 010](../010-redis-model-distribution/design.md) 已实现本仓可选 Redis 分发，扩展此前的
“不引入 Redis 模型存储”选择；本轮未部署该功能。Spec 009 的不可变版本、发布门禁和持久
引用继续保留。Redis current 可能落后 S3，v2 registry 强制走双侧引用保护及协调清理，
旧清理器不得处理 Redis 分发模型。使用说明和跨仓接入步骤见 Spec 010。

## 2026-10-16 跨日合同演进

Source: 用户确认并授权 Spec 012。跨日标签版本和特征投影不再要求完全相同；
按 [Spec 012](../012-daily-contract-evolution/design.md) 合并标签版本及新增特征，
旧日期未声明的已选特征使用 YAML 显式默认值。同名特征定义冲突、合同声明后
缺列、标签其他语义变化及校验和错误仍失败。逐日原合同和版本随输入冻结。
