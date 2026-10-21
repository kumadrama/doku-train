# Spec 015：NN 无参历史 Attention 需求

状态：需求及点积/softmax/加权和方案已由用户于 2026-10-21 确认，授权配置驱动实施及 schema 同步。
Source: 2026-10-21 用户要求参考 Finder rerank 增加无参 attention，后确认点积、固定 attn_scale、masked softmax、reweighted sum 方案。

- R1：DNN/DeepFM 共用 rerank，通过 YAML 在历史组内声明 hist_attn_query/skip_hist_attn，在顶层声明固定 attn_scale；未配置的组保持现有 mean/weighted_mean 行为。
- R2：score_i = dot(q, h_i) / attn_scale；有效历史位置 softmax，再对原始历史 embedding 加权求和。无投影、额外可训练参数或隐式 sqrt(dim) 缩放；既有 embedding 正常接收梯度。
- R3：padding 不参与归一化；空历史输出零向量、无 NaN；OOV 使用既有 embedding，保留为有效位置。
- R4：首版 query 为单个标量类别 embedding，历史为单个类别序列，维度相等；输出沿用历史组名，可用于既有 DNN/FM。
- R5：schema 限定配置类型与正数尺度；模型校验引用、形状语义、维度及外部 weight_feature 冲突。attn_scale 缺省为 1.0。
- R6：本地与 SageMaker 共用模型逻辑；checkpoint 保留配置、feature state 和模型参数，恢复及独立评估采用相同 attention。
- R7：保持 CPU-first，不新增依赖，不复制 Finder 私有源码；不实现 Transformer、多头、cosine、外部权重混合或树模型 attention。
- R8：本次仅本地实现及合成数据验证，不启动云训练或发布。

验收：手算数值、padding/空历史/OOV、反向梯度、无新增参数、配置错误、既有 pooling 回归、DNN/DeepFM train/test 与 epoch resume 等价性，以及 make check。

Source: 用户进一步指出 Finder 启用方式；采用其常规路径字段层级，去除未发布的自定义 hist_attn_configs 接口。旧配置无显式 query 时维持原 pooling，不隐式使用 item 组。
