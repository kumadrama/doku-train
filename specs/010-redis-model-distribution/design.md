# Spec 010：模型与特征产物 Redis 分发设计

状态：用户已授权实施；本仓发布、恢复及清理代码已实现，验证见 [verification.md](verification.md)。
本轮未部署云端；mono 消费者和 iac Redis 装配仍待接入。
需求见 [requirements.md](requirements.md)。

Source: 2026-10-14 用户需求；Spec 009 和现有 `comm/publication.py`、`comm/daily_publication.py`、
`comm/artifact_store.py`。用户随后要求开始实施代码；下述存储和发布协议对应本仓实现，消费者行为仍为跨仓合同。

## 1. 决策与系统边界

保留 S3 为完整制品、生命周期及恢复事实来源；Redis 是线上分发副本，服务内存是执行缓存。
不把 Redis 和 S3 当成一个事务，不把 Redis 指针提交当成所有实例切换成功。

```text
训练 → 评估/门禁 → 导出 → S3 完整包 + publication.json 提交
                                  ↓ 精确版本、摘要
                          Redis 同步与完整性校验
                                  ↓ 原子切换线上目录
                       特征组装服务       predictor
                       后台加载/持久 pin  后台加载/持久 pin
                                  ↓
请求：解析一次版本 → 同版本编码 → RPC 携带精确版本 → 同版本预测
```

当前 S3 包无需重训或更改 model_name 即可补同步；同一版 S3/Redis 使用相同 model_version。
本次不解决实时用户特征从哪里读取的问题；这里的“特征产物”是合同、词表、编码和归一化状态。

| 所有者 | 目标职责 |
| --- | --- |
| train | 从已发布 S3 包构建 Redis 分发、独立重试/恢复、线上指针回滚、统一清理协调。exporter 本身不访问 Redis。 |
| mono | 两个服务的后台加载、本地版本句柄、实例 pins、请求版本传播、路由/冷加载、排空与错误处理。 |
| iac | 环境隔离的 Redis 装配、容量/持久化/副本策略、TLS/ACL/网络、任务身份及调度接线。 |
| monitoring | S3→Redis 延迟、加载失败、版本不一致、孤儿 pin、容量和清理阻塞告警。 |

### 1.1 本仓入口结构与统一编排（用户已确认）

Source: 用户明确要求 publish_model.py 与训练/导出入口同级，由 run.py 统一编排，先更新 Spec，
随后授权实施。本节结构已落地，操作入口及参数见 README。

```text
recommend/train/
├── train_rerank.py              # 保留现有训练入口名称
├── eval_rerank.py               # 独立评估入口
├── export_model.py              # 导出本地推理包及 parity 校验
├── publish_model.py             # S3 发布、Redis 同步/恢复的发布用例入口
├── run.py                       # 参数解析、composition root、统一阶段编排
├── scheduled.py                 # 显式环境/调度参数适配，调用统一用例
└── comm/
    ├── daily_publication.py     # 收敛为日输入准备等辅助能力，不再持有完整日更编排
    ├── publication.py           # 权威清单、引用、生命周期及清理协调
    ├── artifact_store.py        # S3/file 存储访问
    ├── redis_distribution.py    # Redis 同步、校验、目录提交与恢复逻辑
    └── redis_artifact_store.py  # Redis 存储访问
```

统一日更流程由 run.py 在进程内通过用例接口装配，不拼 shell 命令：

```text
scheduled.py（或 CLI）→ run.py 的统一编排用例
  → 冻结输入/日期选择/去重
  → train_rerank → eval_rerank → 发布指标门禁 → export_model
  → publish_model
      → S3 包上传、校验、publication 提交
      → Redis 同步、校验、catalog 提交
```

run.py 只负责调用顺序和阶段结果汇总，不实现特征算法、文件上传循环或 CAS 状态机。
publish_model.py 只装配发布用例，底层制品校验、上传、并发保护放在 comm；不读取训练
内部模型对象，也不承担训练、测试、导出。导出器继续不访问 S3/Redis。

发布入口需要覆盖两种调用模式：完整发布接收已导出的本地包及训练归档引用，依次完成 S3
和 Redis；独立 Redis 同步/恢复接收已成功发布的精确 S3 版本与摘要，不重新训练、评估或导出。
独立调用 publish_model.py 与 run.py 的发布子命令复用同一发布用例，不各写一套流程。
S3/Redis 状态分别返回；Redis 失败/未知时整体分发不得报告成功，但必须保留 S3 已发布事实。

现有实现迁移边界：

- 将 comm/daily_publication.py 中 daily_run 的训练→评估→导出→发布总编排迁到 run.py
  的共享用例；scheduled.py 不再直接调用旧 daily_run。冻结输入/日期选择等辅助逻辑留在 comm。
- 现有 publish_package 的上传/校验/登记能力下沉或保留为 comm 内窄接口，由
  publish_model.py 装配；不把底层上传细节整体搬到顶层入口。Registry 与 ObjectStore 继续解耦。
- 现有 run.py publish 命令改为调用 publish_model 的用例；如需短期兼容旧入口，仅允许薄委托，
  不保留第二套阶段顺序、门禁或错误恢复逻辑。
- 日更命中去重时，统一编排定位原成功版本，继续检查/恢复未完成 Redis 分发；被门禁拒绝的
  run 没有可发布版本，不能误送同步。完整发布重试也不得覆盖同版本不同内容。
- 回滚/清理仍由 run.py 暴露管理入口并复用 comm 协调协议；publish_model 不另实现一套清理状态机。

上述入口结构已实现；daily_run 仅在 run.py 定义，调度适配器复用该用例。

## 2. 产物范围与数据布局

同步以下原始字节，保持原 SHA256，不在 Redis 重新生成特征状态或改写 UBJ：

| 对象 | 特征组装 | predictor |
| --- | --- | --- |
| 原始 manifest.json | 身份及摘要 | 身份、任务到 UBJ 映射、输出语义 |
| feature-contract.json | 必需 | 校验输入合同/顺序/类型所需 |
| fitted-feature-state/state.json | 必需 | 按合同校验所需；不重新拟合 |
| models/*.ubj | 不加载 | 全部声明任务必需 |
| metrics.json、lineage.json、training/** | 不同步，留 S3 审计 | 不同步，留 S3 审计 |

原 Manifest 可包含不在 Redis 的文件；新增分发 descriptor 明确 `required_files` 子集，
包含源 S3 Manifest URI/摘要、各文件路径/长度/SHA256、合同摘要、model_type/name/version/date。
消费者校验原 Manifest 与子集一致，不把“非 serving 文件未同步”误判为坏包，也不跳过必需依赖。
若未来 Manifest 声明额外运行时依赖，须升级分发合同，不能默认排除未知依赖。
分发 descriptor 在独立 S3 distribution 前缀 create-only 保存并登记所有权，不能修改旧包 Manifest。

实际 Redis 键（`E` 为环境，`N` 为经过白名单验证的 model_name，`V` 为版本）：

```text
doku:model:{E:N}:catalog
 doku:model:{E:N}:v:V:descriptor
 doku:model:{E:N}:v:V:file:manifest.json
 doku:model:{E:N}:v:V:file:feature-contract.json
 doku:model:{E:N}:v:V:file:fitted-feature-state/state.json
 doku:model:{E:N}:v:V:file:models/0.ubj
 ...
 doku:model:{E:N}:v:V:ready
```

代码块缩进只用于可读性，不属于键名。一个模型的所有键使用同一 hash tag `{E:N}`，
环境/名称仅允许 `[A-Za-z0-9][A-Za-z0-9_-]{0,127}`，禁止花括号注入。这样小型控制事务可以同槽操作，但同模型占一个槽，
不能假设单个大模型跨分片扩容。数据按文件存储，不把所有权重塞进 catalog 或长时间 Lua。

catalog 一次性包含 `format/epoch/generation/operation_id/current/dates`；每个引用带
version、S3 Manifest 摘要、descriptor 摘要。不要用多个独立 current/date SET 造成半次切换。
ready 仅表示 Redis 包校验完成，不代表消费者就绪。键不设置 TTL；容量不足使同步失败、旧指针保留。
建议使用专用容量且 noeviction；不得未经评估复用业务 Redis。具体 endpoint、逻辑 binding、
预算与引擎版本在 iac 实施前确认，不写入本 Spec 的生产默认值。

Source: Redis 官方 [Cluster 同槽限制与 hash tags](https://redis.io/docs/latest/operate/oss_and_stack/management/scaling/)。

## 3. 两个服务的版本一致性

### 3.1 请求合同

业务方仍可仅提供 model_name + 日期。特征组装服务从后台维护的已验证目录快照解析一次，
固定如下上下文；未指定日期时使用该快照 current，指定日期不存在则明确失败：

```text
model_name, model_version,
items（每候选 item_key + 具名标量 features；按该版本合同编码）
```

2026-10-15 用户确认：同一环境内 `(model_name, model_version)` 唯一绑定不可变产物。请求不传摘要，不传 current/latest 或日期作为 predictor 选版依据。两个消费者在加载时各自校验权威 registry、ready、descriptor、原始 manifest、feature-contract 和 fitted state/权重文件的摘要；已有发布冲突拒绝规则保持不变。摘要是加载完整性检查，不是调用方的选版参数。

predictor 按精确版本查找不可变本地句柄，并校验名称、摘要、输入结构；不再读 current 选版本。
响应回传实际 model_name/model_version，组装方核对；RPC 不携带 manifest_sha256 或 feature_contract_sha256。重试同一已编码请求继续使用原版本。
只有重新开始整个请求、重新编码才可以重新解析；禁止用旧特征配新模型。

两服务不必同时更新全部实例。新版本先到特征服务、后到 predictor 的窗口，由 mono 的
精确版本路由或有截止时间的合并冷加载处理；加载不及时返回 `MODEL_VERSION_NOT_READY`，
缺文件/坏摘要返回 `MODEL_ARTIFACT_INVALID`，合同不支持返回 `MODEL_CONTRACT_UNSUPPORTED`。
明确错误不替换版本。若单机不能同时容纳新旧版本，停止接入相应请求并排空后切换，不强制淘汰在途句柄。
同名不兼容合同仍可发布；本 Spec 不承诺零错误切换，也不引入“全部实例 ACK 才发布”的新门禁。

### 3.2 后台加载与引用

- 两服务各自以实例启动唯一 owner_id，先在 S3 权威 registry 成功 pin 精确版本，再读 Redis。
  S3 pin 是控制路径操作，不是每请求操作；Redis 只读 ACL 不承担生命周期写权限。
  mono 可使用窄控制接口或受限 S3 CAS 客户端，实现位置由其设计确定，不能把权限交给普通调用方。
- 校验 ready、descriptor、原 Manifest 和文件 SHA256，构造完整本地对象后原子加入本地版本表。
  predictor 还须验证原生加载及输出语义；Python 导出 parity 不能冒称 Go 加载验收。
- 每请求获取本地版本句柄引用计数；卸载必须停止新引用，等在途编码/推理和下载归零，再释放 S3 pin。
  组装服务 pin 至少持有到其已发 RPC/重试结束；predictor 独立持有自己的 pin。
- 后台轮询目录与生命周期，Pub/Sub 若使用仅作唤醒，不能作为可靠发布证据。
  draining 后停止新请求并排空；进程失联不自动过期 pin，按 Spec 009 运维核实终止后释放。
- Redis 不可用时，已经加载且持有 pin 的精确版本可继续；新实例/未加载版本失败。
  首版不在请求路径自动下载 S3，也不因 Redis miss 自动选择旧版或新版。后台修复由同步器承担。

## 4. 权威控制记录与跨存储并发

在现有 S3 registry 的版本化扩展中新增 distribution 状态；具体升级须 fail closed：
旧清理器不识别扩展就拒绝实际删除，不能忽略 Redis 引用。旧 producer/回滚命令也不能绕过协调。

`publication.json` 升级为 `doku-publication/v2`，新增 `distribution` 对象：

- `prefix/epoch/generation`：绑定环境前缀、恢复代次、已分配递增序号；
- `operation`：持久 `id/owner/kind/versions/target/expected_base64/created_at`，保存目标目录和原目录完整字节（base64，不假设损坏目录仍是 UTF-8 JSON）；
- `catalog`：最后完成确认的 Redis 目录，保护其 current；pending target 由操作保护；
- `versions`：每版 `reference/objects/bytes/descriptor_key/descriptor/replaced_at/deleted`；
- `last_operation`：最后完成/取消操作的 ID、类型和时间；`protocol_ready` 初始 false。

生命周期进度复用顶层 versions 的 available/draining/deleting/deleted、pins、drain_started_at。
删除逐键幂等重放，不为每个 DEL 保存游标；在完整 Redis 删除核实前不会删除任何对应 S3 文件。
所有者状态存 S3，不自动过期。原 v1 二进制遇到 v2 格式必须报错，禁止降级写入。
generation 仅由同一个 S3 CAS 控制记录分配；现有 registry 的随机 revision/ETag 仍用于 CAS，
不能直接拿它当递增序号。普通消费者 pin 更新不改变发布 generation。

首版每名称的发布、Redis 同步、回滚和实际清理使用持久 control operation 串行化；
原有 active_run 继续保护训练，不代替跨存储操作所有权。该保护不自动过期。
故障恢复必须确认旧执行者及其未决请求已终止/隔离，再接管同一操作；无证据时保持阻塞并告警。
不同 model_name 可独立执行；消费者登记/释放 pin 仍通过同一 registry CAS 并发进行。

Redis 只对小型 catalog/ready 元数据进行原子比较提交，比较原目录完整字节（含 epoch/generation/operation_id）及
目标 ready 摘要；先验证所有前置条件再写目录。已经是同一 operation/内容则幂等成功，
更高 generation 或不同 epoch 则拒绝。数据上传在事务外，不能依靠事务回滚撤销大文件写入。
不能仅用 Redis 过期锁解决 stale writer；S3 和 Redis 没有跨系统原子事务。

Redis 异步复制可能丢失已确认写入，故 Redis 的 generation 不是独立持久权威。
发生 failover、数据丢失或目录回退，先停止控制写入/删除并核对 S3；消费者只接受完整校验的精确包，
可能短时继续使用旧完整版本，不能声称全局瞬时最新。恢复时隔离旧 writer、建立新 epoch 并重建，
消费者丢弃旧 epoch 的解析快照，已有带 pin 的请求仍按原精确版本完成。

Source: Redis 官方 [复制](https://redis.io/docs/latest/manual/replication/)与
[Cluster 写入安全边界](https://redis.io/docs/latest/operate/oss_and_stack/reference/cluster-spec/)。

## 5. 发布顺序、重试与回滚

1. 沿用 Spec 009：训练、test、门禁、导出、完整 S3 上传校验、S3 publication CAS。
   记录 `s3_publication=published`；Redis 故障不修改训练成功或 S3 发布事实。
2. 在控制记录中持久登记分发操作，记录目标精确版本/源 Manifest 摘要和新 generation。
   同时保护目标和旧 Redis 目录引用；即使 S3 current 已前进，旧 Redis current 仍受保护。
3. 校验 S3 状态为已发布 available，按登记文件集读 S3；写 descriptor 与不可变 Redis 文件。
   同键重试只接受字节/摘要一致；缺文件可补，不同内容视为冲突，不能覆盖同版本不同内容。
   写后读回校验全部 serving 文件，最后写 ready。
4. 重查控制所有权、版本状态及目标 generation，原子提交完整 catalog（current 和 dates 一起）。
   迟到旧操作不能覆写新目录。尚未提交的 Redis 数据不可被解析/冷加载入口视为已发布。
5. 读回 catalog 核实 operation_id/摘要，再 CAS 标记 `redis_distribution=committed`，记录线上
   引用和替换时间，释放同步操作。只有到此可声明 Redis 分发成功；消费者就绪由 mono 另报。

超时/失败恢复：

| 失败点 | 处理 |
| --- | --- |
| S3 尚未发布 | 不开始 Redis 同步，沿用原流水线错误处理。 |
| S3 成功，Redis 写入/校验失败 | 旧 catalog 保持，目标及旧引用保护保留；独立重试同一版本，不重训。 |
| catalog 写入响应丢失 | 读回核对 operation_id；无法确认则 unknown，保留新旧保护，不直接重发旧指针。 |
| Redis 已切换，S3 分发确认失败 | 不撤销 Redis 新包；恢复同一持久操作完成确认，期间禁止清理相关版本。 |
| Redis 丢键或 failover 回退 | 校验失败的版本不新增本地加载；按 S3 操作记录修复/新 epoch 重建，不盲信 ready。 |

未完成分发操作禁止后续同名指针变更。`redis-resume --operation-id ... --confirmed-stopped`
在确认旧执行者及未决请求停止后接管原操作。重复输入会进入同步检查，遇到 pending 明确报
`distribution_failed` 并给出 operation_id；不会重训、静默跳过或未经确认自动接管。
pending 时 Redis 丢失/回退/部分损坏，使用 `redis-resume --rebuild-redis --confirmed-stopped`：
在同一持久操作中以 CAS 分配新 generation/epoch，保存当前 Redis 原字节作为比较条件，
按原目标目录重建全部保留包后完成原操作。目标来自 S3 操作记录，不采信 Redis 备份引用。
rollback 保留回滚目标；drain/delete 保留已撤销引用；stage 只恢复原已确认目录，不自动上线。
删除恢复必须先提交重建后的安全目录，再恢复精确键删除；中途再次失败仍保留同一操作。
这一恢复操作不能 cancel，必须 resume；没有 pending 时使用独立 redis-rebuild。
`redis-cancel` 仅允许取消尚未提交 catalog 的 sync/stage 操作，且必须确认旧执行者停止。
取消保留旧目录及已登记文件，后续重新同步或按保留规则清理；已提交/未知结果及
rollback/drain/delete/rebuild 必须 resume，不能用取消绕过中间生命周期状态。
`redis-sync --stage-only` 只上传并校验，不更改 catalog；仍登记并完成持久操作、消费一个
单调序号。用于切换消费者前的 shadow 验证；不能把 staged 声称为已上线。

回滚是更高 generation 的显式操作，选取仍完整保留版本；先恢复 available 并保护、必要时补齐
Redis 包，再切目录。deleting/deleted 不可回滚。回滚 current 是否同时改该日期映射须在同一
目标目录中明确，首版沿用 Spec 009：同时更新该版本日期映射。旧自动发布高水位仍保留，
不得因回滚允许迟到旧训练覆盖它；不更改任何版本文件。

## 6. 统一清理与故障恢复

### 6.1 保护与保留

年龄基于可信创建时间，超过 30 天才候选；S3 current 和 Redis current 无条件保护。
替换时间取 S3/Redis 各自相关引用最后被替换时间的较晚值，至少再保护 24 小时；
Redis 同步长期失败时，旧线上版本可以保留超过 30 天。

同步中目标、未决控制操作、活动 producer、消费者 pins 均保护。历史 dates 引用不是永久保留：
候选可先退出解析并 drain，但两服务已有 pin 不得被强行释放。未知引用或 Redis 不可读时
暂停实际删除。consumer_protocol_ready=true 仍不足以单独授权本 Spec 清理：还需验证 Redis
分发/引用协调已接入；新增独立 `distribution_protocol_ready` 门禁，初始 false。

### 6.2 删除顺序

1. dry-run 仅读取/计算，输出两存储 current、producer/sync/pin、年龄和保护原因，不写状态。
2. 实际清理取得持久控制所有权，重查两侧目录/操作；S3 CAS 标记候选 draining，禁止新 pin。
   此时旧 Redis 目录可能暂留，但新加载必须先 pin，因 draining 被拒绝，不能绕过保护。
3. 更高 generation 原子撤销 Redis 的历史日期引用和可加载入口，同时撤销 S3 dates；
   两步失败时保持 draining 和操作保护重试。两侧都确认撤销后才记 distribution_drain_started_at，并完成/释放本次控制操作。
   drain 等待期不占用跨存储操作所有权，允许其他版本正常发布或显式回滚；状态本身仍禁止新 pin。
4. 从两侧撤销均完成的时间起至少等待 24 小时；两服务停止新请求、排空在途并释放 pin。
   时间到不代表无人使用：pins 非空、producer/sync 活动或 current 改变时仍不删。
5. 后续清理优先处理已到期的 delete 批次，再处理新 drain，避免每日新候选导致删除饥饿。
   重新取得控制所有权，重读两侧目录及当前生命周期；若已回滚恢复 available 则放弃旧计划。
   在 S3 CAS 中重查并转换 deleting，禁止回滚、新 pin、新同步；先撤销 Redis ready，
   按登记键清单逐项删除 payload/descriptor。记录进度，失败留 deleting，重试原键集。
6. 每项 Redis/S3 删除前重查目录完整字节与引用；Redis 删除后、转入 S3 删除前再次核对。
   若目录回退或出现待删版本引用，立即停止，保留 deleting 和 pending 操作，走显式恢复。
   确认 Redis 不再引用该版且键删除完成后，才按 Spec 009 删除 S3 已登记制品及分发 descriptor，
   完成后保存 deleted 墓碑。Redis 和 S3 部分删除都不再对外解析，不能恢复半包为 available。

Redis 目录撤销、S3 CAS 和删除不是一个原子事务，安全依靠持久操作、先禁止 pin、消费者遵守
pin-before-load 及 deleting 状态。旧清理计划不能跳过重查；旧 Redis 写入者必须已终止/隔离。
Redis 历史备份恢复须进入新 epoch 并核对 S3 墓碑，不能重新提供已删除版本。

未登记 Redis 孤儿只报告；已登记但从未发布的半包，在确认原操作终止、无引用且超过保留期后
按相同状态机回收。删除以精确键集合执行，不用 KEYS/FLUSHDB，不按 TTL 或扫描结果直接删除。
S3 上游投影、共享冻结输入保持 Spec 009 边界，不被 Redis 清理扩大范围。

### 6.3 本地缓存与内存容量

本地卸载和 S3/Redis 删除是不同动作：本地只需排空本实例请求后释放其 pin，远端由统一清理决定。
Redis 首版不另设更短淘汰期，也不使用 LRU 提前回收受保留规则保护的包；容量不足拒绝同步并告警。
部署前测量 serving 子集、旧新双版本/30 天积累、复制开销、网络与加载耗时，不能用 S3 全归档
大小直接当 Redis 内存用量。后续若需要只缓存少数版本，须另行设计精确版本回填及可用性合同。

## 7. 接入顺序与验收

1. 本仓实现与任务清单见 [tasks.md](tasks.md)；先部署支持 registry v2 的发布/清理二进制，停止旧任务后再启用 Redis 分发。
2. 按第 1.1 节收敛统一编排并增加 publish_model.py，验证 CLI/调度复用同一用例；iac 装配
   Redis 与权限。通过发布入口独立 shadow 同步既有成功版本，不改线上解析。
3. mono 接入两个服务的精确版本、持久 pin、摘要校验和排空；测试跨实例及 Go 原生加载。
4. 在 PROD 手动验证 S3→Redis 完整分发及一次完整请求，再将日更 S3 发布后接 Redis 同步。
   独立同步重试不重跑训练，重复输入被训练去重后仍可修复未完成分发；回滚走同一协调入口。
5. 验证 dry-run 与竞态后，再显式开启消费者/分发清理门禁；不得因部署完成自动设置 true。
   发布过程不写 DTE，本次代码任务没有创建 Redis 云资源、修改 IAM、发布线上模型或启用门禁。

验收矩阵（本仓已执行范围见 verification；mono/iac 场景仍须跨仓验收）：

| 需求 | 场景与可执行断言 |
| --- | --- |
| R15 | CLI/调度入口走同一阶段顺序；独立发布/恢复不触发训练或导出；去重不跳过分发恢复。 |
| R2/R5 | 中途失败/坏摘要/少一个 UBJ：ready/catalog 不发布半包，旧版本完整可读。 |
| R3/R4 | 编码后切 current、跨实例 predictor 未预热：响应精确原版或明确错误，绝不换版。 |
| R6/R7 | 同步重试、同日多版本、迟到操作、回滚竞态：generation 单调，旧任务不覆盖新目录。 |
| R6 | catalog 成功响应丢失、S3 确认失败：unknown 可恢复，不重训、不丢保护。 |
| R8/R9 | S3 新版但 Redis 旧版超过 30 天：旧 Redis current 仍不能删除。 |
| R9/R10 | pin 与 draining/deleting 并发；服务失联；两个服务分批排空：禁止新 pin，有 pin 不删。 |
| R10 | 目录撤销中断、Redis 删除一半、S3 删除一半：保持不可解析，按原键集幂等恢复。 |
| R11 | Redis 全丢、丢单文件、failover 回退、旧 writer：停止不确定控制写入，重建 epoch，不复活墓碑。 |
| R12 | 同名合同变化：允许分发，消费者明确报不支持，不静默沿用错误特征。 |
| R13/R14 | ACL 跨环境拒绝、包大小超预算、内存不足：拒绝写入，不影响旧目录。 |

本仓使用真实本地 Redis 验证原子目录提交及 UBJ 分发；这不是 Go 消费者、Redis Cluster/TLS/ACL 或 PROD 验收。

## 8. 运行参数与容量边界

所有 Redis 绑定显式来自 `DOKU_TRAIN_REDIS_*`；URL 包含运行时注入的凭据，不进入训练 YAML、
lineage、配置摘要或模型版本。没有绑定时保留 v1 S3-only 行为；已经迁移 v2 的模型不允许
因丢失绑定而静默降级为 S3-only 发布/清理。部分绑定视为配置错误。
连接 URL 必须 rediss（仅本地回环测试允许 redis），禁用 URL 查询参数覆盖 TLS/二进制配置。
显式声明 cluster、连接/命令超时、单文件、单包及同模型保留字节预算。SDK 为 redis-py。

预算计入 descriptor/ready 和已登记但上传未完成的包；总预算不足时保留旧目录并报错。
这是保留的 payload 字节预算，不是 Redis RSS：Redis 对象、复制、持久化和客户端缓冲开销由
容量验收额外测量。同模型同 hash slot，不能通过增加分片绕过该模型单槽容量约束。
由于保留旧版本的安全规则优先，容量不足的 pending 操作可确认停止后 cancel，再扩容或清理
符合条件的非 current 版本；不能为了新发布直接驱逐受保护版本。

## PROD 连接装配

运行时支持 DOKU_TRAIN_REDIS_SECRET_ID 与 URL 二选一，Secret 使用平台 username/password/host/port/db/tls 合同，强制 TLS 和 db=0；只授权训练/清理 task role 读取指定 readwrite Secret。密码不进入 Terraform 或工作流环境。db/doku-redis-model 为平台 setup 身份声明。
