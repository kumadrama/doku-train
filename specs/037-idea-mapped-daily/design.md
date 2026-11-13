# Idea mapped 日更

2026-11-13，用户确认以 08 第 7 轮首发到独立 Endpoint，以 11 月 12 日为首个日更验证日。三个任务按等权验证 AUC 选模，训练权重为 1 / 0.5 / 10，完整训练 20 轮。

从已发布指针冻结连续 UTC 自然日，最后一天验证，前面最多 30 天从头训练。起点为 2026-11-02。遇到缺日等待，重试复用发布状态和确定性任务标识。上游 mapped 发布器将两个剧历史 map 的 keys 按原顺序写为列表并保留空值，训练侧核对日期、行数和合同摘要。每日实时输入和校验均由 Idea 数据合同决定。

从确定的训练产物导出三个预测，推理广播采用 item/idea_drama_id。与参考 checkpoint 对照后，用部署探测检验新模型；保留原配置供回滚。S3 成功标记的事件和计划补偿检查都指向同一独立执行器。保存配置摘要和状态，不记录真实请求。

Source: 用户 2026-11-13 的上线决定及实验 08 训练产物。

## 日训、选轮与导出任务

配置顶层的 `export_tasks` 列出 SavedModel 和 Endpoint 需要公开的预测，例如
`export_tasks: [idea_read_more, idea_vote, idea_watch_now]`。每个名称必须出现在
`model.tasks` 中，列表至少包含一项并且名称唯一。训练仍构建全部任务，选轮仍读取
原有选模配置。缺少此字段的历史配置继续使用发布 binding 的 outputs 选取输出。
单位沿用 binding；新增 sigmoid 输出可直接使用 probability，其他输出须提供单位。
上传前对照配方核对签名的输出名与单位。Endpoint 探测从同一配方解析预期任务，恢复旧发布也按此核对。监控只报告公开任务及其单位，训练指标覆盖全部训练任务。

Idea 适配器按发布进度确定下一验证日并冻结 mapped Manifest。增长窗口计算、训练入口、
训练任务恢复、按选定 checkpoint 导出、校验、发布、
切换及回滚复用 FYP 的公共实现。选模权重和导出列表可以独立配置。

Source: 用户 2026-11-13 指定在 config 增加需导出的 tasks。

## 生产数据源与公共入口

本次迁移继续使用 mapped/v1 日期指针。上游合并 PR #64，将两种账户历史按原顺序写为
drama_interest_ids、not_interested_drama_ids 两个 list<int64>。训练直接读取标有这两列的不可变
分片。公共 executor 接受事件交付的冻结 ready，也可根据绑定的 mapped 成功指针逐日追赶。
data_source.freeze 选择 ready_manifest 或 idea_mapped；后者保存每个日期的 Manifest URI 和摘要。
online_config.first_day 将固定窗口截短至首个训练日；冻结配方显式记录实际天数。
相同的 prepare_job 和发布状态机负责两路日训。Endpoint、状态 URI、上传路径和作业前缀由绑定指定。

显式回滚使用独立发布操作。自动触发遇到进行中的显式回滚时等待，显式回滚成功后从原发布高水位的下一日继续。由探测失败引发的回滚仍需显式重试。
