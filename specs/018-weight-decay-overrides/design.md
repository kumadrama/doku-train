# Spec 018：位置正则化

Source: 用户确认全局默认与位置覆盖方案并授权实施。需求与设计已确认。

模型侧按 embeddings、shared、排序后的任务塔建立稳定且不重不漏的参数组。bias继承所属位置。空覆盖保留原单组路径。所有优化器共享配置；不改变各自衰减语义。恢复使用相同配置和组顺序；现有resume identity校验拒绝变更配置。旧镜像不支持新字段，需后续新建镜像。

配置示例（默认所有参数 1e-3，仅点击塔 1e-4）：

```yaml
optimizer:
  name: adam
  learning_rate: 0.001
  weight_decay: 0.001
  weight_decay_overrides:
    embeddings: 0.001
    shared: 0.001
    tasks:
      immersive_click: 0.0001
```

位置覆盖与全局值相同也允许；未覆盖任务继承全局值。参数组顺序固定为
embeddings、shared、按任务名排序的塔；无参数模块跳过。
