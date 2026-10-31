# Spec 030 设计：配置身份、选中指标、共享投递与看板同步

状态：用户已确认，2026-10-31 授权实施；本地实现完成，云端未部署；验证见 verification.md。

Source: [requirements.md](requirements.md) 为已确认产品决策；现有 Spec 013/027/028/029 代码与合同为实施依据。以下新增字段、目录、异步投递与同步机制为 [INFERRED] 工程设计，需随本文整体审阅。

## 1. 核心架构

```text
模型配置 model_name + 当前训练配方
    ↓
现有 training_pipeline / early_stop
    ├─ training/selection.json
    └─ training/epoch-metrics/NNNN.json
    ↓ epoch_metrics 中只读汇总函数
selected-model-metrics.json（训练产物，保留真实 evaluation_split）
    ↓ 成功 run 封存 / 训练完成事件
平台适配：绑定主任务与实际指标类型，生成不可变 observation
    ↓
S3 监控 outbox ──→ 共享采集 worker ──→ 统一 CloudWatch Logs
    ↑                    ├─ 模型目录 CAS 更新
发布成功/失败/回滚事件    └─ 目录变更触发看板同步
    ↑                                      ↓
现有 Endpoint 发布状态机                原 Grafana dashboard UID

回填 CLI ──→ 同一读取、标准化与 outbox 路径
共享补偿任务 ──→ 重试未投递事件 / 补采成功 run
```

不重新实现选模，不跨仓 import 训练实现，不把 Grafana 凭据放入训练或发布容器。

## 2. 当前可复用的权威数据

| 现有文件/能力 | 使用方式 |
| --- | --- |
| comm/training_pipeline.py 的配置驱动选模 | 保持不变，唯一负责决定 epoch |
| comm/early_stop.py | 保持不变，仍由原训练循环调用 |
| training/selection.json | 读取最终 epoch/checkpoint/score |
| training/epoch-metrics/NNNN.json | 读取同 epoch 的 validation_metrics、value、原因及统计 |
| training/metrics.json | 仅作为已知旧格式的版本化适配来源，不另算最优 epoch |
| experiment-config.json / lineage / COMMITTED.json | 配置、真实来源和文件完整性校验 |
| binding.outputs | 默认主任务与输出单位的既有权威列表 |
| 发布状态 online/history | 真实发布版本、cutoff、EndpointConfig、回滚信息 |

日更入口已有读取 selection 的位置；只补一个汇总调用，不修改训练循环。训练文件中的 checkpoint 绝对路径不用于跨机器直接打开文件，按 selection.epoch 在已验证 run 内定位，并检查 epoch/checkpoint 身份。

## 3. 模型身份与配置

### 3.1 单一配置来源

在实际模型 YAML/JSON 顶层增加：

```yaml
model_name: tf_rerank_ln_lr001
```

model_name 为非空受限字符的稳定业务标识，不含日期、epoch、run 摘要；环境由部署绑定提供。日更监控接入要求该字段存在。离线已有配置可继续缺省，不隐式联网或自动登记；新 schema 对可选字段做统一格式校验。

所有实际产物继承配置中的同一个名称；部署绑定引用该名称并做一致性检查，不允许环境绑定静默覆盖配置。AWS Training Job 的 ModelName 标签、SageMaker Model/Endpoint 标签可用于排查，但标签不是指标归属的唯一权威。

### 3.2 配方与资源身份

模型目录键为 environment + model_name；同时保存 recipe_sha256。摘要输入是规范化后的模型、特征、任务、loss、优化器、选模、种子和日更窗口等训练配方；不包含动态日期/Manifest/run_id、路径、时间戳及部署资源身份。实施时显式列出纳入字段并测试，不能随意删除未知字段以制造一致。

同名绑定不同配方时拒绝合并目录和混画指标；在日更配置预检阶段指出应使用新名称。监控运行时发现冲突则隔离该事件并告警，不篡改旧目录或回滚已成功训练。

metadata 字段加入不得触发网络行为或模型算法分支。配置摘要/lineage 的变化属于新配置身份，不追改旧训练记录。尤其不能给既有 run 的 config 插入 model_name 后再声称原校验和不变。

### 3.3 部署映射

当前使用显式 model_name → region + endpoint_name，现有 Endpoint 不重命名。目录可持有版本化部署绑定引用；监控不把 model_name 与 Endpoint 名认作同一字段，也不强制其一一对应。未来共享 Endpoint 的 variant/component 路由属于独立部署 Spec，本次不实现。

业务实验组通过自己的配置选择 model_name；平台日更更新其在线 model_version，实验分桶不因日期改变。胜出后业务改变 baseline 路由，不改写模型身份。业务路由和归因代码不在本次范围。

## 4. 训练汇总与指标投影

在 `comm/epoch_metrics.py` 增加 `read_selected_metrics` / `write_selected_metrics`，不新增 selected_model_metrics.py。

1. 读取最终 selection.epoch；无可用 selection 时只记录 unavailable 状态。
2. 读取对应 epoch JSON，校验记录 epoch 与 selection 一致。
3. 保留该 epoch 的配置声明指标、func、value、原因及统计，生成 `selected-model-metrics.json`。
4. daily_train_model 在封存前调用，标准产物包含在 COMMITTED 校验清单内。
5. 汇总/监控转换失败不把成功训练改成失败：保留原 selection/epoch 产物、记录监控异常，由平台后续根据已封存原始材料补采；不得构造伪成功汇总。

训练汇总可包含该 epoch 全部声明任务指标，不在公共层硬编码四个主任务。平台按成功 run 对应的 binding.outputs 投影主任务，再按输出单位/配置 func 选择默认展示指标：当前 probability → 唯一 auc，seconds → 唯一 mse。无法唯一选择时要求可选显式展示映射；不能用任务名称字符串或 loss 类型猜测。默认模型不需要另配该映射。

展示名 Test AUC / Test MSE 与原始 evaluation_split 分开保存。MSE 不限制为 0..1；AUC 采用 0..1；不同度量不可混用轴或被统一标成 AUC。

## 5. 版本化合同

合同 JSON schema 与合成 fixture 在 doku-train 定义，monitoring 按版本消费并维护相同合同 fixture；不得复制 Python 内部类型或 import train。

### 5.1 训练观察：doku-selected-model-metrics/v1

- schema_version、event_type=training_selected、event_id。
- environment、model_name、model_type、recipe_sha256。
- run_id、run_uri、training_job_name/ARN、source artifact digest。
- data_cutoff、dataset_timezone=UTC、evaluated_dates、dataset/manifest digest。
- training_completed_at_ms：取真实训练结束元数据，不取重新导出或补报时间。
- selected_epoch、checkpoint identity、selection score；selection score 不是展示指标。
- evaluation_split（当前 TF validation）、display_label=Test。
- 主任务数组：task、metric_name、metric_kind、value 或 null、reason、有效样本数（若原产物提供）。

一个观察原子携带所有主任务值。辅助任务可留在训练汇总文件，不进入默认观察图。run_id 等高基数字段用于日志/归因，不建立每 run 的 CloudWatch custom metric 维度。

### 5.2 发布事件：doku-model-release-event/v1

独立记录 model_name、run identity、model_version、endpoint/region、operation、release revision、发生时间与状态：published / failed / rolled_back / restored。引用对应训练观察，不复制或改变其数值。

发布事件必须在状态机完成相应持久状态写入后产生。每次状态修订具有单调 revision；当前可采用既有持久 history 的追加序号并绑定发布状态对象身份，不依靠采集到达时间决定线上状态。失败尝试不替换 online；显式回滚/恢复更新 online 指针。保留旧记录顺序及来源，不改业务训练操作 identity，也不为投递 ack 增加业务状态迁移。

### 5.3 事件身份与回放

观察 event_id 由版本 + environment + model_name + run identity + selected epoch + evaluation split 计算；同一事件 ID 对应不同 payload 摘要是冲突，进入异常队列而非覆盖。

发布 event_id 由发布 operation + transition/revision 构成。sent_at、received_at 是运输元数据，不参与训练观察排序。缺失真实时间/模型归属时拒绝正常入库并报告原因。

## 6. 非阻塞投递与持久补偿

### 6.1 数据面

- 监控事件写入独立 S3 outbox 的 event_id + payload_digest 路径，create-only；不写回已封存训练 run。
- 已确认 outbox 对象通过共享事件/队列唤起采集 worker。队列允许重复，消费幂等。
- worker 校验合同与来源，写入一个环境级共享 CloudWatch log group，随后记录投递结果。日志接受不等于 Grafana 已可查，不声称 exactly-once。
- writer 已写日志但未写 ack 时可能重复；查询按 event_id/稳定观察排序去重，不能使用平均值消化重复。
- 所有模型复用共享队列、日志组和目录，首次平台装配一次，不按模型创建一套采集资源。

### 6.2 生产者失败也要可恢复

仅有队列重试不够：如果 ECS 在创建 outbox 之前退出，队列中根本没有消息。因此增加一个共享 reconciliation 入口，定期扫描显式授权的训练运行记录与发布 journal，补齐缺少的观察/状态事件，再重试未确认 outbox。

扫描使用分页、有界批量和持久游标；配置一个有重叠的回查范围处理晚完成任务，事件身份去重；首次历史范围由回填入口显式指定。不把已“看见”但未完成的 run 当成已完成游标。训练完成触发是快速路径，S3 成功产物/journal 是恢复依据。仅消费 COMMITTED 成功 run，不从目录存在推断训练成功。

监控投递失败只记录 monitoring_failed 与告警，不改变训练/发布状态。配置结构错误、run 校验失败不属于“投递失败”，不应吞掉或伪造数据。没有引入训练质量阈值。

## 7. 模型目录与看板自动同步

### 7.1 自动目录

采集到合法观察后，维护环境级目录条目：model_name、recipe、主任务/指标及单位、部署绑定引用、first_seen/last_seen、schema version。使用条件写避免并发覆盖。新模型可以在首次训练成功、尚未发布时出现，并显示未发布状态；首次成功上线不需要再注册一次。

目录更新产生同步信号，由 monitoring 统一发布器自动更新看板。周期 reconciliation 补偿丢失信号。图表渲染由目录中的结构数据驱动，不读任意模型 YAML 路径，不执行事件内容。

### 7.2 延用原看板

保持 UID `reco-train-test-auc` 和原文件夹/链接，标题升级为 Model Evaluation Prod。保留原树 Row 及其查询；只管理新增的 TF 自动模型区域。现有树配置保留为一次性的历史展示输入，不追加树模型适配器或历史回填。

每个 TF model_name 一个 Row，当前四主任务 2×2；新增模型或任务合同变更由目录驱动同步。面板 ID 从稳定模型/任务/度量键确定，排序稳定；同步使用 Grafana 版本控制，检测并发编辑冲突后重试，避免覆盖不受管理区域。Grafana API 凭据仅由 monitoring publisher 持有。

保留完成训练任务链接、选中 epoch、真实阶段说明、发布状态与当前在线模型版本信息；主图标题遵循用户约定 Test AUC/MSE。详情信息可以表格/tooltip 展示，不把每日版本变成新的模型 Row。

### 7.3 查询语义

- 新 TF 趋势按 data_cutoff 分组，以 UTC 数据自然日为事实；date 字符串保留原值，图中时间统一转换为该日期的 Asia/Shanghai 零点显示，避免时区显示把数据日期推到前后一天。
- 同模型/截止日，先按 `(training_completed_at_ms, run_id)` 选一条完整观察，再提取各任务字段；必须先选整条，再处理 null。
- 同一排序键重复日志必须具有相同 payload，否则属于冲突，不静默选择。
- CloudWatch 查询按完成时间/run 排序键编码整条固定结构 observation，再使用现有 sortsLast 模式选择；日志时间范围可能与数据日期不同，需查询带明确摄入范围的完整相关日志，再按 data_cutoff 过滤目标窗口。不能简单将 Grafana 时间范围同时当成摄入时间与数据日期。
- 补报时日志摄入时间为现在，但事件保留原 cutoff/完成时间。默认 30 天图查询的摄入范围至少包含当前时刻；查更早历史窗口时，仍应覆盖该窗口至当前的可能补报日志，受共享日志保留期和显式查询预算限制。超过预算/保留期显示覆盖限制，不宣称完整历史。
- 发布状态与训练观察分别查询。当前 online 由带 revision 的状态快照/目录读取；不依赖所选图表时间范围中恰好存在一次成功发布事件。发布器定期重发当前快照，保持 CloudWatch 可查，原 revision 不变。

实际 Logs Insights 查询必须用真实查询验证跨窗口回填、jsonParse/null、多任务原子选择；本地 JSON 测试不能替代数据源验收。

## 8. 回填入口

`platforms/monitoring/backfill_metrics.py` 接受显式环境、绑定、run URI 列表或限定日期范围及授权前缀，支持 dry-run，输出候选/跳过/错误清单。执行时调用与正常流程相同的读取、转换、写 outbox 函数。

历史 run 没有 model_name / 汇总文件时：

1. 校验原 COMMITTED；历史未封存 run 必须采用已审核的来源证据，否则拒绝。
2. 提供显式 model_name 与原配置摘要匹配的历史绑定，记录补充身份的来源，不改原配置。
3. 从已有 selection + epoch-metrics 提取值，读取 SageMaker Completed 时间等原始元数据。
4. 从发布 history 关联实际 model_version 与上线/回滚状态；训练成功本身不代表发布成功。
5. 写新监控事件到独立 outbox。再次执行得到相同 event_id/内容；不触发 train/export/deploy。

首次只回填现有成功日更 run，包括 2026-10-29 cutoff 的 28+1 任务。初始 bootstrap 离线实验不默认进入日更回填范围。若证据缺失，仅报告缺项。

## 9. 文件和仓库改动计划

### doku-train

| 文件 | 计划 |
| --- | --- |
| recommend_tf/train/models/*/config.schema.yml | 接受并校验 model_name；日更入口要求存在；不拓展未实现的模型发布能力 |
| recommend_tf/train/comm/epoch_metrics.py | 追加只读选中 epoch 汇总函数，复用现有格式 |
| recommend_tf/train/daily_train_model.py | 成功选模后写监控汇总；封存原始依据；不做网络监控投递 |
| recommend_tf/train/comm/training_input.py | 将配置身份随已存在 run/lineage 输出传递，避免模型实现各自解释 |
| recommend_tf/platforms/monitoring/contracts/*.schema.json（新增） | 观察、发布事件与模型目录的公开合同 |
| recommend_tf/platforms/monitoring/report_metrics.py（新增） | 标准化、来源核验、稳定事件 ID、不可变 outbox 写入 |
| recommend_tf/platforms/monitoring/backfill_metrics.py（新增） | 显式历史范围回填 CLI，复用 report_metrics |
| recommend_tf/platforms/sagemaker/daily_pipeline.py | 在训练成功与发布终态发出非阻塞监控通知，保留补偿来源 |
| recommend_tf/platforms/sagemaker/publish_model.py | 传递模型身份、发布修订记录；不重写已验证发布算法 |
| recommend_tf/train/tests/test_selected_metrics.py（新增） | selection 与 epoch 指标同源、null、两种现有选模路径 |
| recommend_tf/train/tests/test_monitoring_reports.py（新增） | 幂等、失败隔离、旧产物回填、配置身份 |

不新增 recommend/train/comm/selected_model_metrics.py，不修改树训练。export_model 仅在需要携带 model_name 的制品元数据处按版本合同扩展；不得将该字段误当成旧 sealed run 已有内容。推理响应不变。

### doku-monitoring

| 文件 | 计划 |
| --- | --- |
| src/monitors/reco_train/metrics.py | 新合同的原子观察查询；保留树历史查询 |
| src/monitors/reco_train/dashboards.py | 从自动目录生成 TF Row，保留原 UID/树区域，支持 MSE 轴与状态 |
| src/pkg/training_dashboard.py | 保留历史生成能力，新增目录输入模式 |
| src/pkg/model_metrics_collector.py（新增） | 共享采集 worker，日志、ack、目录更新 |
| src/pkg/model_metrics_reconcile.py（新增） | 按公开成功产物合同补采和 outbox 重试；不 import doku-train |
| src/pkg/model_dashboard_sync.py（新增） | 合并目录与保留区域，自动发布 Grafana，冲突重试 |
| src/monitors/reco_train/alerts.py | 投递积压、采集/同步失败告警，不新增 AUC 门禁 |
| tests/fixtures / tests/test_model_metrics*.py（新增） | 合同、乱序、null、回填、动态目录与看板同步 |

跨仓对 selected-model-metrics.json 的解释只依赖公开 schema。兼容旧 run 的底层 epoch 读取/回填由 doku-train CLI 承担；共享 collector 只消费标准文件或标准事件，不能反向复制训练内部选模逻辑。补采发现缺标准文件时记录待 backfill，不假装支持任意历史训练布局。

### doku-iac

新增共享模型监控模块及环境装配：监控 outbox/ack/catalog 前缀、共享队列/DLQ/日志组、事件与一个补偿调度、采集与同步执行身份、Grafana 只读日志权限。模型动态登记不需要逐模型新增这些资源。训练/发布身份仅能写授权 outbox，worker 有读取成功 run 的限定权限；Grafana 发布权限不授予训练方。

日志保留期、补偿扫描范围、运行预算和告警接收沿用该环境已批准的监控基线并通过 binding 显式传入，不在训练代码中设置生产默认值；环境数值在实施装配时核验，不阻塞本文的架构选择。

## 10. 验证、迁移和实施边界

1. 合同 fixture：两种模型、四主任务、不同 epoch 与 loss、null、重复、乱序、回滚、恢复。
2. 选模复用测试：最大任务 AUC 所在 epoch 不一定为综合选中的 epoch；必须取 selection 指定 epoch。测试不得另建选模 oracle 替换现有规则。
3. 故障注入：标准汇总失败、outbox 失败、日志已写 ack 未写、目录冲突、Grafana 同步失败；训练/发布结论保持，补偿可恢复。
4. 旧 run 回填前后文件清单和摘要一致，SageMaker 训练任务数不增加。
5. 新模型自动出现：不修改监控 YAML 或逐模型 IAM；同名不同配方被检测，不混入同一曲线。
6. 实际 CloudWatch/Grafana 验证：日期窗口、跨窗口迟到回填、整条 run 选择、MSE 轴、null 断点、旧树 Row 和当前线上状态。
7. 先对既有成功日更 dry-run/回填，再部署共享采集和看板同步，最后验证下一次自然日更事件；不为验收额外重训生产模型。
8. 默认只改文档；requirements/design 审阅后再拆 tasks.md。现有自动日更、在线模型和业务流量不因写本文而发生变化。

Sources:
- [Spec 013](../013-test-auc-dashboard/design.md)：旧树看板和排序/回放语义。
- [Spec 028](../028-tf-endpoint-rollout/verification.md)：真实日更与发布状态证据。
- [Grafana CloudWatch variables](https://grafana.com/docs/grafana/latest/datasources/aws-cloudwatch/template-variables/)：不假设任意日志字段可直接作为变量目录。
- [AWS Production Variants](https://docs.aws.amazon.com/sagemaker/latest/dg/model-ab-testing.html)：未来业务分流/部署拓扑参考，本次不实现。

## 11. 实施落点与已验证的设计细化（2026-10-31）

Source: 本次代码实现及本地测试；本节为当前本地行为，云端仍运行旧版本。

- 配置身份位于顶层 envelope，因此 schema 落在 `recommend_tf/train/comm/config.schema.json`，
  运行时由 `model_identity.py` 统一校验；不把 model_name 错加到 rerank/cangen 的嵌套 model schema。
  摘要保留未知训练配置，只排除身份、动态日期/Manifest、执行资源字段；split 的非日期字段仍参与摘要。
- 汇总位于 `training/selected-model-metrics.json`，纳入既有 COMMITTED。云端 composition root
  通过 `on_completed` 纯本地回调写 `training/monitoring-event.json`，随后统一封存上传；没有训练侧监控网络请求。
  此标准事件包含真实训练完成时间、选中 epoch、任务结果和模型身份，允许共享 worker 在 ECS
  尚未写 outbox 就退出时独立恢复。回调失败保留原始 selection/epoch 证据，待显式回填。
- 不可变 outbox 使用 event_id 单键 create-only，并比较完整规范化 payload；冲突不能覆盖。
  worker 使用 payload claim、delivery ack 和 CAS catalog。无效合同进入独立 quarantine，避免
  单条坏事件阻止后续合法事件；旧 run 缺标准文件时写 needs-backfill 记录。
- release revision、时间戳、标准监控事件与原 release history 一次持久保存。业务状态不依赖
  监控 ack。发布失败保留已有 online；回滚按已确认 target 版本更新 online。
- 共享 worker 分页扫描环境级 run/journal 前缀及 outbox，保存分页游标；完整扫描后从头开始，
  因而早先未完成的 run 后续仍会重新发现。每轮页数与页大小显式设限。首期不是按时间递增
  的 watermark，因此不需要猜测晚完成窗口；大规模前缀的扫描延迟须在部署验收中测量。
- Grafana 原生 panel 时间覆盖在 dashboard 使用绝对时间时无效，不能靠 timeFrom 保证补报。
  实施改为固定日志扫描范围 `now-retention_days` 到 `now`，隐藏原通用时间控件；新增
  `cutoff_from` / `cutoff_to` 目录日期控件过滤 TF 的 data_cutoff。旧树面板及查询原样保留，
  其扫描范围随看板使用保留窗口。超出日志保留期不承诺完整历史。默认控件可选择目录中的日期；
  不应使用 URL 的 from/to 参数覆盖固定摄入窗口。实际数据源行为仍需部署验收。
  参考：[Grafana 时间范围覆盖限制](https://grafana.com/docs/grafana-cloud/learn-and-build/visualizations/panels-visualizations/query-transform-data/)。
- online/release/latest_training 由 catalog 快照生成说明面板，不依赖所选数据日期里存在发布日志。
- Lambda 是独立构建单元：`scripts/package_model_monitoring.py` 只打包所需运行模块，隔离
  authoring registry 的 import 副作用；不修改 notifier 发布路径。
- IaC 提供 `modules/model_monitoring` 和独立 `envs/prod/model_monitoring` 装配入口；全部环境
  参数显式输入，未提交真实部署 tfvars、未生成伪造 binding。现有桶通知/KMS、producer policy
  和 datasource policy 由相应资源 owner 装配。启用前仍需生产配置与真实链路验收。

Production rollout refinement: AMG service-account tokens expire within 30 days. The dedicated
monitoring service account uses short-lived 600-second tokens created by its scoped Lambda IAM
role and revoked after each dashboard sync. No static token is written into config or source.
