# 多源输入设计

Source: Spec 034 已确认需求，用户授权先设计再本地实施。

多源声明位于运行配置 data_sources；单源旧路径保持不变。各 source 独立冻结合同，使用公共 PublishedDataset 校验与时间切分。显式配置错误在开始读取前拒绝。

读取器先以空 feature selection 载入源合同，再仅请求合同实际提供的配置列；缺少的模型特征须有显式默认值，跨来源标签可零占位，由 loss_weight 来源门控。datasvr_source 为读取层保留字段。

每个来源全部 split 写入独立 prepared/sources/<source_id> staging。任何输入读取/合同异常使该来源全部 prepared 数据丢弃；配置表达式及资源限制错误不吞掉。成功来源的 complete.json 文件引用合并进各 split complete.json，并附 source_id、source_rows；无 train 样本来源不发布。完整来源状态及冻结证据写入 input/sources.json，恢复仅使用冻结记录。

全来源 train 为空时写 run-status.json skipped，返回空指标列表且不调用训练。空 validation/test 仍写完整的空 prepared manifest，由训练执行层处理。

配置可为 manifest_uri 或 manifest_uris 非空不重复列表（二选一）。列表逐日独立验证和冻结到 input/<source_id>/<index>，不依赖 FYP label_config_version/projection_sha256 元数据。各来源所选同名特征的 dtype 必须一致；所有缺列（含另一来源标签占位）都须在 feature_default_values 显式声明，避免将过滤字段拼写错误静默补 0。

多日期覆盖先在来源内聚合：train 声明日期必须由来源内所有 Manifest 合计覆盖，否则静默跳过该来源。单份 Manifest 仅读取其实际日期与声明 split 的交集；validation/test 未覆盖日期允许为空。不同 Manifest 的事件区间重叠视为配置错误，避免双重计数；同一 Manifest 的并行分片允许共有时间区间。

验证：`test_mixed_days.py` 的逐日独立 Manifest 实际 TF 训练及重叠 Manifest 拒绝共 2 项通过；`test_mixed_input.py` 配置唯一性与双来源不可用记录共 2 项通过。
