# Spec 015：字段组合与无参历史聚合

状态：2026-10-21 用户确认按 Finder 的设计职责拆分 combiner 与历史聚合，并授权 schema/实现同步修改。
Source: 用户先确认点积/固定尺度/softmax/reweighted sum，后明确要求“把combiner和历史聚合拆开，不要只是对齐字段名”。本规则取代此前未发布的 attention 启用接口，以及 Spec004 将 mean/weighted_mean 放入 combiner 的描述。

- R1：combiner 仅组合组内字段：concat/sum/avg，沿字段维度，不减少历史位置维度。允许多个字段组成历史和 query。
- R2：历史组必须显式选择 history_pooling：mean、weighted_mean、attention 或 none；标量组不能声明历史聚合。none 保留序列，不隐式回退 mean。
- R3：attention 使用已组合的标量 query 和已组合的历史 embedding，执行 dot(q,h)/attn_scale、masked softmax、加权求和；不新增参数，embedding 正常训练。query 组及历史组组合后维度必须一致。
- R4：padding 不参与归一化；空历史零输出、梯度有限；OOV 作为有效位置参与。输出复用组名，聚合后可进入 DNN/FM。
- R5：多个历史字段必须按位置对齐，长度和 mask 一致。不能把不同历史列表自动拼成对应事件。sum/avg 要求字段宽度相同；concat 可组合不同宽度。
- R6：weighted_mean 才能声明 weight_feature；attention 才能声明 hist_attn_query；顶层固定 attn_scale 缺省 1 且必须有限正数。schema 拒绝旧 combiner 值、skip_hist_attn 和之前未发布的 hist_attn_configs 接口，不提供静默兼容。
- R7：本地/SageMaker 共用 PyTorch CPU 模型；无新依赖、私有代码复制、GPU、多头或参数化投影。树模型不增加神经聚合语义。
- R8：迁移合成 fixture 及有效文档，验证数值、预处理、训练、测试和恢复。本次不启动云训练、不发布。
