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
