# Spec 024：当前实验迁移范围

当前状态：范围及必要适配已确认并实施；以下保留初始审计和后续确认记录。最终语义以 design.md 与本文末尾确认记录为准。

## 范围依据

Source: 用户指定实验的本地配置 runs/e1-extended40-lr004-20261027/config.yaml。
SHA-256: ae7ef0b526d6bd0d9e17b5888a344f9c49e096bc494383e8fe5c3c6e1ee69271。
本轮未核验云端冻结配置；这里描述本地配置证据。真实 YAML 不复制入源码。
Finder models revision: 0abdcc692a84319833d871e30c0bfd7ee6a22080。
Finder train revision: cbd00cdc32f9e843204eda52d52480d71fde0105。

## 当前实验需要的功能

| 项目 | 实际配置 |
| --- | --- |
| 输入 | 11 组：item_id、hist_drama_skip、hist_drama_interest、item_tag、user_profile、context、item_numeric、item_statistics、user_statistics、hist_drama_finish、hist_drama_immersive_enter |
| 表达式 | 时长单位转换、统计量 log2/round/clip 分桶、3 个阈值分桶、活跃天数 clip/cast；包括批量字段展开 |
| Embedding | drama、genre 等共享表；默认维度8、指定字段4；uniform[-0.01,0.01] |
| 历史 | skip/finish/immersive_enter 三路 dot attention，query=item_id、scale=1；interest weighted_mean；item_tag/user_profile mean |
| 历史输入规则 | 三路序列保留重复、tail100；interest 合并重复权重、top_weight100、raw_id_ascending 破同分 |
| 共享网络 | DNN 128→64，ReLU，无 norm，dropout 保留率0.9 |
| FM | item_id 与四历史各一对，item_tag 与 user_profile 一对，共5对 |
| 任务头 | 7个独立32维ReLU tower；6个单输出sigmoid/logloss；watch_seconds 为61输出softmax_soft/expectation |
| 标签 | completion、1-non_fast_swipe、immersive_click、watch_seconds>10/30/50、clip(watch_seconds,0,300)；后四项有nullable mask |
| 权重与soft label | 主分类权重1/1/3，三个play各0.2，时长0.3；normal scale=5、sigma=2 |
| 运行 | batch2048、40 epochs、seed42、Adam lr0.004、weight_decay0.0001；单机CPU |
| 数据评估 | item_filter；train 10/1–23、validation 10/24、test 10/25；四主任务loss加权选模，分类与时长指标 |

上述值用于确定范围，不成为框架硬编码。旧 TF 的实现、参数命名、初始化、checkpoint、数值结果均不作为兼容标准。

## 按 Finder 迁移的主体

目标目录 recommend_tf/train/。保留函数边界、调用位置、参数意义、计算顺序；不只添加同名包装。

| 函数/模块 | 本轮路径 | Finder源码证据 |
| --- | --- | --- |
| model_fn | 保留 raw_features；转换→embedding→attention→core→预测/标签/loss | rerank_model.py:80、100、106、116、155、230 |
| transform_features | 图内批张量表达式、分组作用域、字段展开与配置执行顺序 | rerank_model.py:53 |
| embedded_features | 共享key分组、length、稀疏处理和组合；后端单独适配 | rerank_model.py:369 |
| attention_layer | 当前三路默认dot/softmax路径，scale和mask | rerank_model.py:953、1018 |
| rerank_core_model | 输入拼接、共享DNN、5对FM、独立任务头 | rerank_model.py:1225、1246、1401 |
| dnn_layer / target_layer | 当前dense/ReLU/dropout，初始化、scope、输出squeeze | rerank_model.py:505、571 |
| 公共依赖 | 所需eval_expr算子、fm_layer、get_predictions、两种loss、normal soft label及传递依赖 | modules/utils.py；comm/common_utils.py:1157；comm/nn_utils.py |
| schema / params | 当前路径的字段意义、默认值与校验；显式映射Doku配置 | rerank/config.schema.yml、model_params.py |

“迁移”允许 import 与 TF compat API 的必要机械适配，不代表原文件可以未经处理直接运行。

## 必要适配与待确认差异

| 编号 | 项目 | 事实及必须解决的边界 |
| --- | --- | --- |
| A1 | Manifest/Parquet | 替代Finder私有native reader；保留Doku合同校验，输出所需batch分组、dtype/shape、length/SparseTensor，不提前逐行执行特征表达式 |
| A2 | 本地embedding后端 | vocab_manager/auto_fusion有私有依赖；替换lookup，保留embedded_features主体。key映射、OOV/padding、共享表及状态写出须明确；未默认批准新的拟合/缓存方案 |
| A3 | weighted_mean | 未找到当前weight_feature对应的Finder直接路径；需显式加权pooling适配，明确空序列和零权重；不能称原样迁移 |
| A4 | 连续值直通 | item_numeric混合quality/heat连续值与桶embedding；基础rerank非skip字段查表；已补查到rerank_prom.scalar_features及input_type=scalar，可迁移该已有路径，具体来源组合待确认 |
| A5 | nullable标签 | mask/is_null/metric_label属于Doku合同；需validity tensor、无效标签处理及loss分母。Finder loss_weight不自动等价于有效行mask |
| A6 | 历史输入语义 | tail、重复合并、top_weight、tie-break映射到输入adapter/图的边界要明确；mean与稀疏表示要验证 |
| A7 | 配置/表达式 | feature_list映射apply_to_features/apply_to；FM嵌套形式、head from、clip/is_null等逐项核对；缺少算子最小补充，不能另造标量解释器 |
| A8 | Adam衰减 | Finder dense/sparse weight_decay与AdamW不同路径，不能把Adam+weight_decay直接改名AdamW；公式、变量范围、稀疏更新待设计核实 |
| A9 | 本地运行与评估 | 替代私有平台训练/evaluator/checkpoint服务；保留Doku时间切分、指标/选模与产物合同，标签和预测计算保留在图内 |
| A10 | TF2.11 CPU | compat.v1、移除平台imports；不借此改写网络、初始化和dropout语义 |

Source: rerank_model.py:369、571、1018；comm/nn_utils.py:532；comm/graph_helper.py:772；本地实验配置。
这些是待明确的设计差异，不是已经批准的新实现；不能声称当前Doku YAML可直接交给Finder运行。

## 本轮不迁移

[INFERRED] 当前配置未要求以下分支；映射时显式关闭相关开关，不让Finder默认值额外启用。

- LHUC/FiLM、额外ID/user/item/general-cross/multimodal tower、ESSM。
- 参数化/低秩/多目标/softplus attention、cross-history/cross-target、longhist2/HSTU、短曝光Transformer。
- LLM cache、GPU、Horovod/MPI、多机、WePS/PSStor、私有数据/评估服务及平台进程管理。
- 当前未使用的loss/activation/norm等不增加支持承诺；所选路径必要helper必须一起迁移。
- 线上exporter与部署、旧TF兼容、旧P1/P2缓存优化原型（不排除下方新确认的基础输入物化）、并行预取调优、Spec022/023新增能力。

基础PREDICT图与共享依赖不因没有单独配置而删除，但不扩展为线上服务项目。未支持功能明确报错，不静默降级。

## 验收顺序

1. 确认范围及适配需求，再给出设计并确认，随后生成tasks。
2. 逐函数记录源摘要、直接迁移/适配及理由。
3. 固定合成输入与权重，对比转换、embedding、attention、core、logits、loss、梯度及一步更新；替代后端独立验证，不假称复现私有服务。
4. 小规模CPU训练、评估、恢复、空历史、无效标签和失败产物边界验证。
5. 固定数据切片与机器，测准备/输入/训练/验证耗时、RSS，再讨论优化；不沿用旧P1/P2提速数字。

本轮仅整理文档，未实施迁移或启动云训练。

## 补充确认与 scalar 审计

Source: 用户本轮同意输入侧与model_fn分工、历史整理规则；要求兴趣weighted_mean在model_fn内配置驱动，并核查Finder scalar。

- 输入侧保持原始batch交付，model_fn的张量计算随每个batch执行；Python model_fn本身负责构图，不要求每batch重新调用构图。
- length在历史过滤/合并/截断后、padding前计算，作为独立batch字段保留；padding不增加length。
- Manifest/合同冻结、日期分配与schema校验可训练前一次完成。默认值、样本过滤、历史整理、length等确定性样本操作可以预处理一次，但跨epoch复用需物化数据；用户随后已批准此基础输入物化，见下方最终确认记录。
- 不增加新物化层时，上述样本操作随输入流分块进行；每epoch shuffle、batch划分及随batch最大长度padding不能统一固化为一套batch而保持原shuffle行为。
- 兴趣weighted_mean明确在model_fn调用链内由配置驱动；不在输入侧对embedding求均值。
- 更正此前过宽结论：Finder模型仓库已有scalar_features。rerank_prom/rerank_model.py:131在transform_features后调用，:135与embed_inputs合并；:586实现float32转换、dense扩维/稀疏转dense、concat/sum；schema:276提供input_type=scalar。基础models/rerank没有该同名路径，因此建议从此Finder变体迁移所需函数及配置，而非从零新增连续值功能，来源组合仍待确认。
- 当前item_numeric混合连续值和embedding桶；scalar按组分流，需在设计中明确拆组并保持拼接顺序，不能直接把整个组标为scalar。

## 输入准备与 shuffle 最终确认

Source: 用户确认“拆分输入组可以”，随后对训练前准备、逐epoch shuffle、逐batch padding方案确认“好的 那就还是按照你之前建议的实现”。本节取代此前“本轮不物化基础输入”的暂定建议。

- 训练前按分片流式执行默认值填充、样本过滤、历史去重/权重合并/排序/截断，保存基础输入；不要求全量入内存。
- 历史整理完成后、padding前计算并保存真实length，ID与权重对齐；保留标签有效性信息。
- 原始字段与整理后的变长历史可复用；特征表达式、scalar、embedding、pooling/attention、标签表达式及loss仍在model_fn图内按batch计算，不缓存最终编码。
- 按时间划分train/validation/test；仅训练集在每个epoch按seed与epoch重新shuffle，再划分batch。shuffle仅改变样本顺序，不改变样本内部历史或重建历史。验证/测试保持确定顺序。
- 每个batch按该batch历史真实长度padding，length不受padding影响；空历史的张量宽度与mask规则在设计明确。
- 历史必须满足曝光时刻可用信息合同，排除本次曝光自身结果及未来行为。此为要求，不代表本轮已经完成真实数据的point-in-time验证。
- 确认拆分混合输入组：item_numeric中的连续字段与分桶embedding分开配置，保持DNN拼接顺序和维度；scalar沿用已审计的Finder路径。
- 物化文件格式、容量/清理、完成标记与失效依据等在设计中明确；收益重新测量，不引用旧缓存实验数字。

本次仅记录已确认需求，尚未实现基础输入物化或迁移模型代码。

## 最终更正

用户已确认固定train-only词表、图内lookup、padding0/OOV1；基础输入共用特征与原始标签链路。标签默认值与参与训练由配置及loss_weight决定，撤回自动nullable mask。weight decay以Finder为准，撤回Torch全参数耦合L2。A2/A4的待确认表述已被后续决策取代；A5/A8以[设计](design.md)为最新目标。
