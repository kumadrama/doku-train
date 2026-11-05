# Shuffle 与 loss 设计

Source: Spec 034 Q7/Q10；用户已授权先设计后本地实施。

准备阶段允许显式 allow_empty，默认单来源行为保持不变。混合 manifest 记录每个分片 source_id；训练读取按各来源剩余行数随机抽取来源，在各来源内部顺序批读。seed 为 run seed + epoch。无 seed 的拟合与评估顺序读取。

共享 shuffle buffer 满后随机选择槽位输出，并以新样本引用替换；尾部通过末尾交换移除随机槽位，每条操作 O(1)，不复制样本张量。单来源原有无 seed 行序保持不变。prepared 分片只允许位于 prepared 根目录内部（包括同级 sources 目录），拒绝路径逃逸。

任务新增 loss_reduction=sum 时直接 reduce_sum(weight * losses)，旧任务未配置时保留既有归一化。混训组装层显式设置 sum。指标计算独立于该训练 objective。
