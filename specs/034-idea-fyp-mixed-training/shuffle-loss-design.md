# Shuffle 与 loss 设计

Source: Spec 034 Q7/Q10；用户已授权先设计后本地实施。

准备阶段允许显式 allow_empty，默认单来源行为保持不变。混合 manifest 记录每个分片 source_id；训练读取按各来源剩余行数随机抽取来源，在各来源内部顺序批读。seed 为 run seed + epoch。无 seed 的拟合与评估顺序读取。

共享 shuffle buffer 满后随机选择槽位输出，并以新样本引用替换；尾部通过末尾交换移除随机槽位，每条操作 O(1)，不复制样本张量。单来源原有无 seed 行序保持不变。prepared 分片只允许位于 prepared 根目录内部（包括同级 sources 目录），拒绝路径逃逸。

组 batch 后走公共 TF 任务实现。每头的 `weight` 包含来源和任务资格。原有 `loss_by_weight: false` 对所有行的 `weight * loss` 取均值，`true` 以非零权重行数为分母，全零时返回 0。整轮任务损失沿用同一分母跨 batch 加权汇总，短 batch 使用实际行数。训练 objective 沿用各任务 loss 之和加正则。验证指标按各指标声明的 filter 统计。

Source: 2026-11-11 用户要求复用 FYP 损失计算，替代 PR #38 中由组装层强制设置的批内加权求和。
