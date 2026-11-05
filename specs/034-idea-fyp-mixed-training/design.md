# Spec 034：配置驱动多来源 TF 训练设计

Source: 已确认 requirements.md；2026-11-04 用户明确授权先写设计然后实现代码。
状态：本地实施中，无云操作。

## 接口与兼容

保留现有单 dataset_config 入口。新增顶层 data_sources 列表，每项包含 name、source_id、format、manifest_uri；共同时间切分、执行预算和默认值仍由 dataset_config 声明。来源格式为 doku-train-dataset-manifest/v1，来源 ID 必须唯一且稳定。多天输入先按来源独立解析，不将不同业务合同混为同一数据集家族。

各来源通过现有 PublishedDataset 校验 Manifest、合同与分片。读取投影以源合同与请求字段交集为准，缺少的特征按显式默认值补齐；不以数据源名称跳过合同校验。读取层注入 datasvr_source。字段配置错误应与来源不可用区别处理。

## 准备、冻结与事务边界

按来源准备各 split 到独立目录，成功后将完成清单引用写入混合 prepared 清单。失败来源不贡献已准备的半成品。source 记录持有不可变输入身份、状态与错误类别，不打印敏感值。全部 train 为空时写 skipped 状态，不进入图执行、不声明训练产物成功。

清单与来源冻结记录共同构成缓存 identity。恢复使用冻结输入，不重新解析指针。源文件路径采用 prepared 根内相对路径；校验与内存/磁盘预算继续生效。

## Shuffle

首次完成批量读取、过滤与预处理，训练 epoch 只读取本地准备好的列块。每来源独立游标，按剩余行数比例选择下一条样本引用；公共 buffer 以随机槽位替换输出实现 O(1) 单次操作。排空时随机出队。seed+epoch 控制顺序，来源清单顺序稳定；无跨 split 混合，不重复、不丢样本。

## Loss 与评估

任务新增显式 loss_reduction=sum，未配置时保留原单源语义。多源配置要求使用 sum；来源资格与 placement 通过 loss_weight 表达式控制，标签占位不自动有效。正则保持现有语义。

模型配置新增 eval_tasks，覆盖已有任务的指标配置，不创建额外模型头。metric.filter 独立于 loss_weight，图中求布尔 mask，聚合器按每项指标分别记录有效行数与 AUC 样本，不共享分母或 AUC 缓存。

validation/test 共用 eval_tasks，Idea 来源过滤与 placement 条件显式声明。空 validation 的指标记录 null，禁用基于其指标的早停/选优但保存最终 checkpoint；空 test 记录空评估。未选优不得标记最佳 checkpoint。

## 字段资格与限制

placement 来自上游每日投影 string 字段：WATCH_NOW 对应 watch_now，REGULAR 对应 vote（包括 Notify），NULL 不参与这两目标。此为用户接受的近似口径，不声明其等价客户端按钮。FYP 用户标识仅用于采样，不自动成为模型输入。实际训练目标、日期与特征由配置声明，不写入框架常量。

## 验证

合成数据覆盖双源、缺字段、来源失败回滚、空 split、固定 seed、每 epoch 行数守恒、指标独立过滤和 sum loss；运行 TF 回归与 make check。专项实施说明见 evaluation-design.md、input-design.md、shuffle-loss-design.md（随实施补齐）。

### 实施接口补充

单来源可用 manifest_uri 或 manifest_uris（二选一）声明精确不可变日文件；后者顺序冻结每份合同，无需上游 FYP 专有 label_config_version/projection_sha256 字段，也不发现 latest。跨日窗口在来源集合上处理。
缺少任意所选列（包括另一来源标签的占位列）必须显式声明默认值；不会把字段拼写错误自动变成 0。跨源同名特征 dtype 冲突作为配置/合同不兼容显式失败。

## 2026-11-05 单入口兼容补充

Source: 用户要求原 manifest_uri 能力兼容到多源。每来源 manifest_uri 支持精确 Manifest 或显式 `{date}` 模板。原 TF 单源 reader 是精确读取，模板解析为多源新增适配，不声称旧 reader 原生发现每日产物。解析日期来自已有 split 配置；支持 FYP/Idea _SUCCESS 的 manifest_uri+manifest_sha256 公共合同，不要求 Idea 补 FYP 专有字段。解析及加载期间验证指针未变化、摘要一致、时间范围正确；冻结 URI 列表与指针原文，重放不解析可变入口。
