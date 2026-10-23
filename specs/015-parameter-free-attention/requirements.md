# Spec 015：NN 无参历史 Attention 需求（待确认）

Source: 2026-10-21 用户要求参考 Finder rerank，增加相似度、softmax、reweighted sum 的无参 attention。
参考实现：finder-models/recommend/train/models/rerank/modules/attention.py 的 hist_attention_layer / attention_head；finder-train 为训练入口，模型实现位于相邻 finder-models。仅参考语义，使用本仓 PyTorch CPU 独立实现，不复制私有源码或引入 TensorFlow/GPU 依赖。

## 已明确范围

- DNN/DeepFM 共用 rerank 实现，新增由 YAML 显式启用的历史 attention；原 mean/weighted_mean 默认行为不变。
- 计算流程为 query/key 相似度、沿历史有效位置 softmax、对 value 加权求和。
- Attention 本身不新增可训练参数；既有 embedding 继续通过 query/key/value 路径接受梯度。
- 保持训练、评估、checkpoint 恢复计算一致；本地与 SageMaker 共用模型实现。
- 不启动云训练，不发布模型，不变更树模型语义。

## 建议首版语义（待确认）

[INFERRED] 候选标量类别 embedding 为 query，单个历史类别序列 embedding 同时作为 key/value；维度必须一致，推荐同类 ID 共享 embedding 表。
[NEEDS CLARIFICATION: 相似度采用原始点积还是余弦？建议原始点积。] 首版建议 score_i = dot(q, h_i)，alpha = masked_softmax(score)，output = sum(alpha_i * h_i)，不隐式加入 sqrt(dim) 缩放或可训练温度。
[INFERRED] Padding 不参与 softmax；空历史输出零向量且不得产生 NaN；OOV 继续使用既有 OOV embedding，不等同 padding。
[INFERRED] attention 输出可供既有 DNN/FM 输入消费；历史截断与重复处理沿用显式配置，不与既有外部 weight_feature 混合，需保留兴趣权重时继续使用 weighted_mean。

## 验收要求

- 用手算向量验证 similarity、softmax、加权和；验证 padding 不改变结果、空历史零输出。
- 验证 query/history 梯度可传播，attention 无新增参数。
- 验证错误引用/维度/配置组合明确报错、既有 pooling 行为保持不变。
- 验证 DNN/DeepFM CPU 训练与 checkpoint 恢复；运行 make check 并记录可执行证据。

需求确认后再确认 design.md，随后生成 tasks.md 并实施。
