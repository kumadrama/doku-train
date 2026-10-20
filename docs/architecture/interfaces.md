# 接口合同

Source: Spec004及实现。旧格式、固定四目标、固定28天与refit生命周期已移除。

配置：公共/模型config.schema.yml运行时合成；搜参独立search.schema.yml。未知字段报错。
模型工厂接收展开配置和feature state，返回fit/evaluate/load/checkpoint_name接口。
模型不读取环境配置、不访问上游存储；checkpoint写入由训练用例提供的本地run目录。

数据：当前版本adapter读取doku-train-dataset-manifest/v1；校验逻辑SHA256、合同版本、
指定分片SHA256/行数/事件时间，按显式train/validation/test自然日选择。
源合同对象摘要写入lineage；源v1未提供合同摘要及完整成熟性关联，不能因此声称已证明成熟性。
真实v4实验区间已完成独立历史证据核查；数据集名称不作为训练白名单。
详情见真实数据核查（本地实验记录，不随源码发布）。

模型产物：checkpoint.pt或checkpoint.xgb.zip、metrics.json、fitted-feature-state/state.json、
lineage.json和status.json；另有resolved-config、resume及编码缓存。status=complete最后写出，
失败仅记录failure，不覆盖原run。测试评估显式产生test-metrics.json，不参与训练选模。

分类预测均为概率；树使用raw margin后应用一次sigmoid。神经soft label预测按scale还原秒数。
比较分类logloss/AUC、回归MSE；训练loss和可比较指标单列，不跨模型直接比较不同目标函数。

## 2026-10-13 Dataset 简化（当前规则）

Source: 用户确认单 Manifest 地址方案。dataset 只声明 manifest_uri；旧版本/摘要、adapter、
aws_profile、allowed_prefixes 和训练内 daily_prefix 说明被本规则取代。
版本与摘要从 Manifest 读取并校验，AWS 身份由启动环境提供，无生产默认值。
每次 Run 冻结 Manifest/合同原文及摘要，resume/test 使用原冻结记录；不重新解析日更指针。

## 树模型每日发布：本仓实现与跨仓目标

Source: 2026-10-13 用户确认；[Spec 009 需求](../../specs/009-daily-tree-model-publication/requirements.md)、
[设计](../../specs/009-daily-tree-model-publication/design.md)。

当前状态：本仓已实现 Spec 009 的 file/S3 发布和清理入口；未部署云端，消费者协议尚待 mono 接入。
目标状态：上游训练投影成功后触发单次训练、独立测试、发布门禁、原生 XGBoost 导出与
S3 清单提交；本仓提供回滚及 30 天清理入口，iac 管基础设施，mono 管加载与 RPC。
model_type 和 model_name 均由配置声明；同名配置/合同变化允许发布，文件按日期与不可变
版本隔离。业务调用方可只指定名称/日期，但公共组件必须一次解析并传递同一版本给编码
和 predictor，不允许各自选 latest。保留当前版本和替换后至少 24 小时的旧版本。
新增门禁只影响发布，不推翻成功训练状态；旧文档的首阶段 exporter/本地输出边界由
Spec 009 在树模型日更范围内扩展。

Source: 2026-10-13 用户确认 S3 方案并授权实施。消费者后台下载与加载，在线请求只用内存中
已就绪的固定版本，不逐请求读 S3；发布成功不等于服务就绪，不引入 Redis 模型存储。


## Redis 分发：本仓实现与消费者目标（Spec 010）

Source: 2026-10-14 用户要求保留 S3 并同步线上所需模型和特征产物到 Redis；
[需求](../../specs/010-redis-model-distribution/requirements.md)、
[设计](../../specs/010-redis-model-distribution/design.md)。

当前状态：此前“未部署云端”的描述属于历史阶段；本会话已完成 PROD S3 发布、清理 dry-run
及调度启用验收。本仓已实现 Redis 分发入口，尚未部署；mono 双服务消费仍待实现，清理受消费者和分发双门禁保护。

实现及跨仓合同：S3 保存完整制品与生命周期权威；Redis 分发原 Manifest、特征合同、
拟合状态与任务 UBJ；服务后台校验加载，热路径用本地内存。特征服务解析一次版本，RPC
携带精确 model_name/model_version 与摘要，predictor 不自行选择 latest。
S3 发布与 Redis 分发分别记状态；完整 Redis 包校验后原子切换目录，同步失败保留旧引用。
清理同时保护两侧 current、同步操作与双方持久 pins；先停止解析并排空，再删 Redis，最后删 S3。
发布入口为 publish_model.py，日更统一由 run.py 编排，scheduled.py 只适配参数。
registry v2、Redis catalog v1、恢复/取消及精确键清理协议详见 Spec 010；消费者行为尚待 mono 实施。

## 跨日输入合同演进（Spec 012）

Source: 2026-10-16 用户授权；[Spec 012](../../specs/012-daily-contract-evolution/design.md)。

当前本地实现：每日输入独立验证源合同与投影摘要，标签版本集合合并；特征合同
允许字段新增，已有同名定义和其余元数据保持一致。旧日期合同未声明的已选字段
必须有 YAML 显式默认值，物理文件缺少已声明的必需列仍失败。冻结 Manifest 的
`daily_contracts_version=doku-daily-contracts/v1` 标记逐日合同证据；`source_manifests`
内保存原始合同与 Manifest 文本及摘要。`lineage.daily_sources` 记录日期、源地址、
原特征/标签版本、投影/合同摘要和 `defaulted_features`；实际默认值由冻结配置保留。
合并 Feature Schema 的版本由源合同摘要确定；不冒充任一源版本。

部署状态：本地验证已完成；用户随后授权生产部署，实际执行证据单独记录。

## 每日 test AUC 监控事件

Source: [Spec 013](../../specs/013-test-auc-dashboard/design.md)，2026-10-17 用户授权本地实施。

当前状态：daily_run 在独立 test 成功后冻结 test-auc-events.json；scheduled 通过 stdout
交付 doku-test-auc/v1 JSON-lines，由已有 awslogs 收集。事件同时随成功/未发布产物归档。
日志投递失败仅输出不含敏感详情的 reporting_failed 事件，不改写训练状态或发布门禁。
目标状态：monitoring 的 CloudWatch 看板按 model_name 四图分组展示；本次未部署，
云端查询/权限/实际呈现尚待后续授权验收。协议和同日重跑/null 规则以 Spec 013 为准。

## 树模型派生规则交付

Source: [Spec 009 派生规则修复](../../specs/009-daily-tree-model-publication/design.md#2026-10-20-派生规则交付修复)，2026-10-20 用户授权。

当前状态：本地导出的 doku-tree-features/v1 在 encoding 中包含完整 history_scalars，
空规则写空对象；历史发布包曾遗漏该段。既有 Go 消费者显式拒绝此字段，尚未部署适配。
目标状态：消费者仅凭正式合同和 fitted state 生成模型输入，完整合同优先于历史归档
补载。常规发布需同步升级两个编译入口；列顺序、类型、权重和 OOV 语义不变。
用户后续授权本次直接切换 DTE/PROD 的新制品引用，操作证据见 Spec 009 verification；
这不代表消费者适配或训练镜像已部署，日更持续生效仍需部署修复后的导出器。
