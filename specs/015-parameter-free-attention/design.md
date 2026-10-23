# Spec 015：字段组合与历史聚合设计

Source: 2026-10-21 用户明确要求按 Finder 的职责拆分两层。参考 finder-models/recommend/train/models/rerank/rerank_model.py 的 embedded_features（字段组合）与 attention_layer（历史聚合），以及 modules/utils.py 的 avg_pool。独立 PyTorch CPU 实现，不复制 Finder 私有源码。

## 当前状态

本地实现两阶段计算；尚未构建或部署 attention 镜像。本设计取代本 Spec 之前仅对齐 hist_attn_query/skip_hist_attn 名称的未发布实现。原先 combiner=mean/weighted_mean 的历史配置必须显式迁移，不让同一个键同时表达字段组合和历史归约。

## YAML

```yaml
input_features:
  item_id:
    features: [i_drama_id]
    combiner: concat
  hist_drama_skip:
    features: [u_skipped__drama_id]
    combiner: concat
    history_pooling: attention
    hist_attn_query: item_id
  hist_drama_interest:
    features: [u_drama_interests__id]
    combiner: concat
    history_pooling: weighted_mean
    weight_feature: u_drama_interests__weight
attn_scale: 1.0
```

这是片段，完整配置仍需声明 embed_name/embed_dim、hist_options、网络、任务及数据日期。

- combiner 缺省 concat；concat 沿最后一维拼接字段，sum/avg 沿字段轴求和/平均。对 [B,L,D] 历史张量保持 B、L 不变；单字段三种组合等价。
- history_pooling 对历史组必填：mean 按有效位置平均；weighted_mean 按外部权重归一化加权；attention 按目标相似度加权；none 保持 [B,L,D]，不能直接被 DNN/FM 或 attention query 消费。
- 标量组只做字段组合，不能配置 history_pooling。允许由多个标量 embedding/数值字段组合 query；本期 query 不引用历史组，避免隐藏的聚合依赖链。
- hist_attn_query 仅在 history_pooling=attention 时必填，引用组名，与组声明顺序无关。attn_scale 为所有 attention 共用的顶层固定正数，缺省 1.0，不做隐式 sqrt(dim) 缩放。
- weight_feature 仅用于 weighted_mean；本期不混合 attention 和外部权重。

## 多字段示意

两个位置对齐的历史字段各 [B,L,8]：concat 后 [B,L,16]；sum/avg 后 [B,L,8]。然后 attention 才消除 L，输出 [B,16] 或 [B,8]。query 也先按其 combiner 组合，最终宽度须相等。实际业务需由上游合同保证两个字段代表同一批事件；本仓检查长度/mask，不能凭长度相等证明业务对应关系。

## 算法与边界

先计算所有组的原始字段组合 raw_groups，再独立执行历史聚合。attention 读取 raw_groups[query] 与当前历史组的 raw tensor：

1. score_i = dot(q,h_i)/attn_scale。
2. padding 置 -inf；全空行在 softmax 前使用有限零 logits，softmax 后将 padding 权重清零。
3. output = sum(weight_i * h_i)。全空行零输出，OOV=1 仍有效，padding=0 无梯度。

mean/weighted_mean 同样只沿历史位置归约。无新可训练参数，不改变 embedding 初始化顺序。sum/avg 对字段维度相同的 tensor 计算；concat 尊重 features 声明顺序。拒绝组内混合标量和序列。

## 数据编码与截断

FeaturePlan 不实现 attention。它保留各字段序列编码，校验多字段原始历史长度一致；tail 截断同步作用于组内所有字段和可选权重。编码后的有效位置 mask 再由模型核验一致。

top_weight 的按 ID 去重/排序只支持单历史字段，拒绝多字段 top_weight，以免按一个 ID 重排破坏事件对齐。同一源字段的 padding 容量统一分配，不受组声明顺序影响；若不同组为同一字段声明冲突截断策略则拒绝。本次不支持为同一源字段生成多份不同截断视图。

## Schema 和组件边界

- comm/config.schema.yml：combiner 枚举改为 concat/sum/avg，字段组合是公共描述。
- models/rerank/config.schema.yml：声明 history_pooling/hist_attn_query 的组属性及条件、顶层 attn_scale；公共 schema 加载器仅通用合并属性和 allOf，不 import 模型实现。
- models/rerank/model_params.py：通过 Feature State dtypes/tables 校验标量/序列、组合宽度、query 引用、聚合模式及 DNN/FM 消费形状。
- models/rerank/rerank_model.py：先组合后聚合；attention.py 保持纯张量算法。
- comm/feature_plan.py：对齐检查、同步 tail 截断；不选择神经网络聚合算法。

通用 validate 校验结构和条件；需要合同 dtype 的检查在模型构造时执行（fit/encode 后、优化前）。数据/配置错误不会写成功训练状态。

## 迁移与产物

| 原配置 | 新配置 |
| --- | --- |
| combiner: mean | combiner: concat + history_pooling: mean |
| combiner: weighted_mean | combiner: concat + history_pooling: weighted_mean，保留 weight_feature |
| hist_attn_query + skip_hist_attn: false | combiner: concat + history_pooling: attention + hist_attn_query |
| 需要保留历史序列 | history_pooling: none（不直接送 DNN/FM） |

scalar 的 combiner: concat 保持含义。旧接口被 schema 拒绝，修改旧配置后生成新的 run，不改已有 checkpoint/config，也不声称旧配置在新代码下可直接恢复。resolved-config、checkpoint 与 lineage 继续冻结完整配置与源码；同一新配置的恢复仍验证 identity。

## 验收

手算验证三种字段 combiner × 四种 history_pooling；检查组合前后的维度与数值、顺序无关 query 读取、空历史/padding/OOV/梯度、无新增参数、对齐检查、尾部同步截断、schema 拒绝旧值和非法组合。合成 DNN/DeepFM 分别验证单字段和多字段 attention train/test，以及 dropout 下 epoch resume 等价。执行 make check，记录实际结果。
