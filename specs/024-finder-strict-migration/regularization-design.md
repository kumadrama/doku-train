# Finder Dense 与命中 Embedding 正则迁移提案

状态：2026-10-29，用户已确认，本地实施；未修改云实验配置、镜像或运行任务。

Source: 用户要求按 Finder Dense 正则＋命中 Embedding L2 设计，先核对重复 ID 计数，再确认方案。

## 原始实现与计数证据

Source: `/Users/yujiatang/myProject/finders/finder-train/recommend/train/` 下：

| 文件 | 已核实行为 |
| --- | --- |
| `comm/nested_vocab/basic_group.py:113` | TokenUniqueGroup 用 tf.unique 去重，保留逆映射 indices |
| `comm/nested_vocab/anonymous_fusion_strategy.py:360` | gather 恢复重复出现位置，随后分发 named_lookup_results；discriminative 路径第279行同样恢复 |
| `example_model/embedding_utils.py:76` | 将 lookup 结果加入 ps_embeddings，再做 pooling/reshape；spadd 路径跳过 |
| `example_model/example_model.py:176` | Dense regularization collection 加入 loss；sparse 按表名筛选 lookup 结果，乘 sparse_weight_decay 后加入 loss |

结论：上述普通 lookup 路径按**模型输入位置的出现次数**累计，不按 batch 内唯一 ID 计数。底层查询去重不意味着正则去重。同一 ID 跨样本、跨输入位置出现，重复项都保留。准备阶段历史去重已删除的项不再计数；这不是按原始日志出现次数计数。

设有效 lookup 输出为 e_p，Embedding 正则为 λ_s/2 × Σ_p ||e_p||²；不除以 batch size、历史长度或唯一 ID 数。ID A 出现三次，贡献 3λ_s/2 × ||E_A||²，对该行的正则梯度为 3λ_s E_A。兴趣权重、pooling、attention 和 task loss_weight 不缩放该正则。

边界：这套完整 loss 汇总来自 Finder example_model，不代表基础 rerank 已完整启用。基础 rerank 配置 Dense kernel_regularizer，但没有把 regularization collection 加入总 loss。不能将本次称为逐行复制基础 rerank。

原 example_model 对同表的 tensor 列表直接 tf.nn.l2_loss(v)，会要求各 tensor 可堆叠为相同形状。普通 fusion 的 (embed_name, group_name) 唯一键断言也不支持一个组内多字段共享同一表的重复键。迁移不复制这两个结构限制。

## 已确认实现

| 项目 | 目标行为 |
| --- | --- |
| Adam | 保持 TF Adam；L2 加入 loss，由 Adam 的一、二阶矩处理，不调用 AdamW 衰减 |
| Dense | 沿用每层 regs: [l1, l2]，本轮 L2 使用 [0.0, 系数]；仅配置层 kernel，不含 bias、LN gamma/beta、未配置的最终 logits 层 |
| Dense 系数 | 沿用 Keras L1L2：λ_d ΣW²，梯度 2λ_d W；与 sparse 的半平方和系数约定不同 |
| Embedding | 按有效 lookup 出现位置计算半平方和，再求和；跨字段/输入组共享表时各使用位置都计入，不按 ID 去重 |
| 历史 | 在 pooling/attention/兴趣权重乘法之前取 embedding；length 之外 padding 不查询、不正则；真实 OOV 命中本地 OOV 行，按出现次数计数 |
| 表筛选 | 沿用 sparse_weight_decay_vars 的表名子串匹配；[] 表示全部；启动记录实际匹配表名，有非空选择但零匹配时报配置错误 |
| 开关 | dense_use_weight_decay、sparse_use_weight_decay 默认 false；sparse_weight_decay 非负；启用但没有目标层/表时明确失败，避免静默无效 |
| 总目标 | optimization_loss = 原有 data_loss + dense_reg_loss + sparse_reg_loss；仅 TRAIN 更新参数 |
| 指标 | 分别记录三项及总目标；验证/test 的 task 指标与既有选模公式保持原义，不能把正则混入原 task loss；如记录 EVAL 正则，用独立诊断标签 |
| 默认兼容 | 关闭正则时输出、梯度、随机数消费和现有实验配置语义不变；不改变已提交的七组实验 |

本次仅为 Adam 开放新增正则；AdamW 与新增 L2 同时开启时报错，避免无意双重正则。若以后需要组合，单独显式扩展。

必要适配：本地冻结词表替代私有 PS；对每个 lookup tensor 分别计算 l2_loss 后求和（支持不同历史长度/组形状）；以当前真实 length 排除 padding；按字段/组保留独立记录，避免 fusion 键碰撞；空有效历史贡献标量零。相同形状时逐 tensor 求和与原 list 堆叠数学等价，浮点求和次序可能不同。

不增加 full-table L2，不实现唯一 ID 平均或按频率归一化。未命中行的本次 **L2 梯度** 为零，但不能据此保证 Adam 状态更新后该行完全不动：历史动量及所用 Adam sparse 更新行为须单独测试。

## 验证与实施顺序

用户已确认，需求和任务同步更新。预定覆盖：重复 ID 的 loss/梯度倍率；跨样本、跨字段共享表；不等长 lookup；空历史与 padding；OOV；Dense 系数及参数范围；关闭开关前后数值一致；Adam 一步与带历史动量的多步更新；checkpoint 恢复；独立正则指标与选模不变；CPU 固定切片耗时与峰值内存。通过本地及原生 Linux 验收后，云端新增实验另行确认。

审计探针：`runs/finder-regularization-audit-20261029/probe.py` 使用 TF 2.11 的 unique→lookup→gather→l2_loss，校验重复行梯度、同形状列表和异形状列表行为。它不运行 Finder 私有 PS，也不构成完整 Finder 端到端验收。结果在同目录 result.json。

## 用户确认与实现接口（2026-10-29）

Source: 用户明确授权开始实施和验证本配置驱动方案。上文“待确认”状态由本节取代。

在现有 model 命名空间增加：

```yaml
optimizer: adam
optimizer_args: {}
dense_use_weight_decay: true
sparse_use_weight_decay: true
sparse_weight_decay: 0.00003
sparse_weight_decay_vars: []
output_layer:
  dims: [128, 64]
  acts: [relu, relu]
  regs: [0.0, 0.00003]
```

这是字段示意，不是已批准的新实验参数。regs 对该 block 所有隐藏层 kernel 生效，task block 可分别配置；本轮仅允许 L1 为零。Dense 开启要求至少一个非空隐藏 block 的 L2 为正；有正系数但关闭开关报错。Embedding 使用当前有效输入表筛选，空匹配报错。历史空序列贡献零。

训练指标新增 train_objective_batch_mean，按实际更新 batch 等权平均，含 data_loss、dense_reg_loss、sparse_reg_loss、optimization_loss。它与既有按任务有效分母聚合的 train_loss 口径不同，尾批也算一次更新；总目标分解在这一相同口径下可相加。TensorBoard 路径为 train/objective_batch_mean/<name>。

实现中仅 TRAIN 构建正则目标；EVAL/test 使用原任务 loss，不额外记录正则诊断。避免复用层时重复汇总 collection。
