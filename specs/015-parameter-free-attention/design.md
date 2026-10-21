# Spec 015：配置驱动的无参历史 Attention

Source: 2026-10-21 用户确认设计并授权实施；本仓 rerank 的共享 embedding/pooling/FM 结构。
参考：finder-models/recommend/train/models/rerank/rerank_model.py 的 attention_layer 常规分支，以及 modules/attention.py 的历史 query/key/value 分离思路。Finder train 只作为训练入口参考。以下是本仓独立 PyTorch CPU 实现，不复制 TensorFlow/私有平台代码。

## 当前行为与配置

rerank 专属 schema 新增可选 hist_attn_configs，按已有历史 input group 名配置；不修改公共 input_features 的合同，不让 comm 反向 import 模型。

```yaml
input_features:
  item_id:
    features: [i_drama_id]
    combiner: concat
  hist_drama_skip:
    features: [u_skipped__drama_id]
    combiner: mean
embed_name:
  i_drama_id: drama
  u_skipped__drama_id: drama
embed_dim:
  drama: 8
hist_attn_configs:
  hist_drama_skip:
    query: item_id
    attn_scale: 1.0
# 其余既有 YAML 配置保持完整
```

hist_attn_configs 显式覆盖相应组的 mean/concat pooling。query 引用 input_features 组，不是原始字段名；每个 query 必须只含一个标量 embedding，不能引用 attention 组（不支持链式依赖）。历史组必须只含一个序列 embedding。允许多个历史组共享 query，且与 YAML 组声明顺序无关。组维度必须相等；同类 ID 推荐通过 embed_name 共享词表和 embedding，但不强制共表。

每组 attn_scale 缺省 1.0，必须有限且大于 0，固定不训练；不接受未知键。不配置 hist_attn_configs 或配置空对象保持既有输出和参数布局。

## 计算及边界

模型从编码 batch 直接查 query/history embedding，保留历史序列，不先求 mean：

1. q: [B,D]，h: [B,L,D]，valid = history_ids != 0。
2. score: [B,L] = sum(q[:,None,:] * h, -1) / attn_scale。
3. 无效位置置 -inf；全空行临时置零 logits；softmax 后显式将无效位置权重归零。
4. output: [B,D] = sum(weights[:,:,None] * h, 1)。全空行输出零向量，反向梯度有限。

OOV=1 是有效位置；padding=0 被排除。缺失标量 query 仍遵循当前编码的 OOV 规则，不新增缺失特征。历史截断和重复策略仍由现有 hist_options 决定。外部 weight_feature/weighted_mean 与 attention 不混用，配置时报错；保留原 weighted_mean 可继续使用兴趣权重。

无新 nn.Parameter、Linear、LayerNorm 或可训练温度。输出沿用历史组名；DNN 拼接和 FM 点积均读取此结果。

## 文件与校验边界

- models/rerank/config.schema.yml：参数结构和范围，仅 rerank 可用，树模型拒绝此配置。
- models/rerank/model_params.py：利用 Feature State 的 tables/dtypes 和 embed_dim 校验引用、标量/序列与维度，编译为 history/query 字段及尺度。
- models/rerank/attention.py：无状态张量运算。
- models/rerank/rerank_model.py：在普通 pooling 分支前接入；不修改训练流水线、数据生产或 SageMaker 专属入口。

通用 validate 命令验证 schema；依赖 Feature State 的语义在模型构造时验证（训练 fit/encode 后、优化前），不会伪造成功产物。未来若需提前到数据读取前校验，应通过独立模型验证接口设计，不在 comm import rerank。

## 可复现性

现有 resolved-config、checkpoint config、Feature State 与运行 identity 包含配置和源码摘要，无需新制品格式。无 attention 的模型保持参数名称和初始化顺序；attention checkpoint 独立评估按保存配置重建。恢复仍遵循冻结输入及 identity 校验，不跨配置/源码冒充兼容。

状态：本地实现；不表示已部署包含 attention 的 SageMaker 镜像。上云前需另行授权构建/部署，旧镜像不会因本地 YAML 改动自动获得新能力。
