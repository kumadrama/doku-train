# Spec 018：位置正则化

Source: 用户确认全局默认与位置覆盖方案并授权实施。需求与设计已确认。

支持 optimizer.weight_decay_overrides 的 embeddings、shared、tasks.<task> 覆盖。未指定位置继承全局 weight_decay，允许0，拒绝负数、未知位置和未知任务。不改变学习率、loss或模型结构。
