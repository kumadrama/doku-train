# Spec 041：直接复用 Finder LHUC

Source: 用户确认需求并指定“直接复用finder的lhuc结构”（2026-11-20）；
Finder 来源 revision 及函数见 [requirements](requirements.md) 与
`recommend_tf/source-inventory.json`。本设计记录该决定，不增加结构变体。

## 实现边界

复用已有 `dnn_layer` 和 `target_layer` 的 LHUC 分支，保持计算、初始化、正则和 scope
不变；从 Finder `rerank_core_model` 迁移条件输入拼接、梯度阻断和任务塔开关的调用。
模型专属逻辑全部在 `recommend_tf/train/models/rerank/`；公共训练与 export adapter
继续使用同一个 `model_fn`，不另建推理图实现，不新增依赖。

## 配置与张量流

先在 `config.schema.yml` 声明三个键，再在 `model_params.py` 增加白名单、默认值与校验：

| 配置 | 默认 | 语义 |
| --- | --- | --- |
| `lhuc_inputs` | `[]` | 从已编码/池化的输入组按声明顺序拼接，非空即启用共享 DNN 门控 |
| `lhuc_share_embed` | `true` | 对条件向量 stop_gradient；不会阻断输入在主干或 FM 中的梯度 |
| `target_layer_use_lhuc` | `false` | 将同一个条件向量传给每个任务塔，门控参数按任务与层独立 |

输入组列表只接受不重复的非空字符串，必须引用 `input_features` 中的组；两个开关
只接受布尔值。空列表时不创建门控变量，即使任务塔开关为 true，也沿用 Finder 的无门控行为。
条件输入使用本仓既有 scalar/embed/history 编码；不自动增加 embedding 或场景特征。

共享 DNN 每层计算顺序为 `门控 -> Dense -> norm -> activation -> dropout`。
门控为 `x * 2 * sigmoid(Dense(condition))`，宽度是当前 x 的宽度。
FM 输出按原逻辑拼入共享输出。若任务塔开关开启，每个任务隐藏层前及 logits 层前
均进行门控；任务 `stop_grad` 同时阻断主干输入与条件输入，任务自己的参数仍可学习。
隐藏层门控 kernel 沿用对应层 regs；logits 前门控沿用 Finder 无局部正则的实现。

2026-11-20 用户指定仅国家进入 LHUC。当前本地配方使用 `ctx_ip_country`，
从原 `context` 组的最后一个字段拆成独立 `country` 组，复用 `ip_country` embedding。
`country` 紧随 `context` 加入 `dnn_inputs`，主干特征拼接顺序/维度保持原样；
国家只 lookup 一次，避免额外 lookup 改变 embedding 正则的计数。
`lhuc_share_embed: true` 保持 Finder 默认，国家 embedding 仍从主干收到梯度。
共享 DNN 和任务塔都启用 LHUC，条件只含国家；其他特征继续保留原主干路径。

配置片段（其余配置沿用原配方）：

```yaml
model:
  input_features:
    context:
      features: [ctx_platform, u_deep_link_drama_id, u_adjust_network]
      from: item
      input_type: embed
      combiner: concat
    country:
      features: [ctx_ip_country]
      from: item
      input_type: embed
      combiner: concat
  lhuc_inputs: [country]
  lhuc_share_embed: true
  target_layer_use_lhuc: true
```

本地实验配方：`runs/lhuc-country-20261120/config.yaml`；基于已有 FYP 全国家基线的
本地快照 `runs/fyp-us-country-20261119/live-config.json`，模型标识为
`tf_rerank_ln_lr002_3feat_lhuc_country`。`source.json` 记录来源和配置摘要；实验 YAML
依仓库规则仅保留本地。未变更原始快照、国家过滤、学习率、训练窗口或线上配置。

## 恢复与导出

配置与 checkpoint 随既有训练产物冻结，训练/EVAL/PREDICT 共用模型实现。
默认关闭不增加图节点或变量；开关或输入维度变更后不声称可恢复旧结构 checkpoint。
SavedModel 导出从成功训练产物恢复所有门控权重，验证条件专用特征仍可达签名输入。
生产配置、发布行为和上游合同不在本次变更内。

构建集成补充：现有 Idea worker 的源码包覆盖 TF 模型代码，统一检查要求本地
`idea-mapped-daily/deploy/prod/task-definition.yml` 中 `ARTIFACT_SHA`/`ARTIFACT_KEY`
与源码同步。按既有 `package_source.py --pin` 流程更新这两个内容摘要字段；训练配置、
binding、资源、任务族不变。本步骤仅准备本地可审查清单，不上传、注册或部署。

## 验证设计

使用合成输入和显式权重，数值 oracle 独立按 Finder 数学公式计算；验证多层、任务塔、
FM 拼接及多输出维度。梯度测试分别观察条件专用输入、主干共享输入和门控参数，覆盖
share_embed 与 task stop_grad。验证 regs 的作用范围和 EVAL 变量复用。
默认关闭与旧调用路径比较变量、预测、loss、梯度及一步更新。
使用真实 CPU 训练入口与本地合成 Manifest 验证成功产物、checkpoint 恢复和 SavedModel
parity；不访问生产数据或启动云任务。

## 已授权的单次 SageMaker 实验

Source: 用户后续明确要求基于 `doku-exp-fyp-us-other50-20261120-0a923a14` 启动一次训练。
此授权取代上文针对本轮云实验的限制，线上发布仍不在范围内。

- 从指定任务的实际 recipe/config.json 冻结基线；验证配置 SHA256 与 DescribeTrainingJob
  的 DOKU_CONFIG_SHA256 相同，提交前再核对远端 recipe 元数据字节。
- 保留 `video_finish_seen > 0` 的曝光有效性条件，移除国家过滤和其他国家 50% 抽样。
- 拆分 country 输入组并仅传入国家 LHUC；保留主干顺序、stop-gradient=true、任务塔开启。
- 保留原模型标识、30 epoch、batch 2048、LR 0.002、30/1 天窗口、选模和单机
  ml.m5.4xlarge/100 GB 磁盘，使用新的 run-scoped JobName/S3 目录。
- 复用原镜像与锁定依赖。File channel 中附加经过摘要固定的 bootstrap 与三文件覆盖包
  （model_params.py、rerank_model.py、config.schema.yml）。启动前校验镜像原始训练源码
  摘要、锁文件摘要、各文件替换前后摘要，复制到独立工作目录后应用。
  校验后推导实际训练源码摘要并记录日志/输出证据，原 cloud_train 再次校验该摘要。
  镜像中的其他源码不变；不替换平台入口或安装依赖。
- 本地请求与来源证据在 `runs/fyp-all-country-lhuc-20261120/`；提交幂等绑定唯一 JobName，
  不因重试新建第二个任务。确认源码验证及进入训练后完成“启动一次训练”的请求。

### 后续 context + country 对照实验

Source: 用户明确授权再开一次训练，将 context 和 country 都加入 LHUC。
以 `doku-exp-fyp-all-lhuc-20261120-440fa353` 为基线，只将 `model.lhuc_inputs`
从 `[country]` 改成 `[context, country]`。条件字段按序为 `ctx_platform`、
`u_deep_link_drama_id`、`u_adjust_network`、`ctx_ip_country`；国家仍仅出现一次。
共享输入组和主干不变，stop-gradient=true、任务塔开启；使用相同冻结分片、源码覆盖包、
依赖镜像、优化器、日期、选模及资源。请求与配置保存在
`runs/fyp-context-country-lhuc-20261120/`，独立 JobName 和输出前缀，不停止或改写前一次实验。

## 国家 LHUC 容量实验

Source: 2026-11-20 用户明确指定复用 06 的 MLP、05 的 embedding 及组合，授权三个任务。

以 `doku-exp-fyp-all-lhuc-20261120-440fa353` 为基线，从原提交配置复用对应配置块：

| 变体 | output_layer.dims | embed_dim.drama |
| --- | --- | --- |
| LHUC + mlp256 | [256, 128] | 默认 8 |
| LHUC + e16 | [128, 64] | 16 |
| LHUC + e16 + mlp256 | [256, 128] | 16 |

05 的实际差异仅是 drama 维度，其他 embedding 维度保持原值；06 的实际差异仅为
output_layer.dims。LHUC 条件仍为 `[country]`，stop-gradient 与任务塔 LHUC 均开启。
复用原源码覆盖包和固定镜像，30 epoch、batch 2048、LR 0.002、单台 ml.m5.4xlarge、
100 GB；train 2026-10-19～11-17、validation 11-18。每个任务使用独立不可变路径。
提交前验证配置 schema、还原指定差异后与基线相等、冻结输入和源码覆盖包字节一致，
再与远端原 recipe 核对；提交使用固定任务名并 describe，重试不重复创建任务。

## 三个独立对照实验（2026-11-22）

Source: 用户明确独立基于 03，并确认共享 MLP [512,256] 两层。

1. network-country：从 context 中分出 u_adjust_network 为 context_adjust_network，
   在主干 context 后、country 前插回，保持特征拼接顺序和各字段 lookup 次数；
   LHUC 使用 [context_adjust_network,country]，沿用原共享 embedding 和 stop-gradient。
   分组名排序仍位于 context 与 country 之间，维持序列化配置的 embedding lookup 顺序。
2. mlp512：仅 output_layer.dims 从 [128,64] 改为 [512,256]，层数仍为 2。
3. task64：仅全部七个任务隐藏层 dims 从 [32] 改为 [64]；num_outputs 保持原配置。

三项互不叠加，固定 30 epoch、batch 2048、LR 0.002、单台 ml.m5.4xlarge / 100 GB，
train 2026-10-19～11-17、validation 11-18。继续使用原镜像和原源码覆盖包。
配置 schema、差异还原、冻结字节、主干字段顺序和重复 lookup 计数在提交前验证。

## 两组容量融合实验（2026-11-22）

Source: 用户明确指定融合此前三项容量调整并提交两组训练。

基于 03 国家专用 LHUC，两组分别设置共享 MLP [512,256] / [256,128]，
均将全部七个任务的隐藏层设为 [64]，并复用 05 的 embed_dim：仅 drama 从8维升至16维。
其余配置与 03 一致，包括 LHUC [country]、stop-gradient、任务塔门控、输出维度、
30 epoch、batch2048、LR0.002、单台 ml.m5.4xlarge / 100GB，冻结 train 10/19～11/17、
validation 11/18（2026年UTC）。继续复用原固定镜像和源码覆盖包。
还原三项改动后须与基线配置相等；任务名及产物路径独立，提交可按固定名称重试。

## 国家、渠道与 deep link LHUC 实验（2026-11-23）

Source: 用户指定LHUC三个条件字段，沿用03对照配置。

将context拆为保留ctx_platform的context、context_deep_link、context_network；
主干按原特征顺序插回后两组，复用原embedding表和lookup次数/顺序。
LHUC输入有序为[country,context_network,context_deep_link]，准确对应用户三个字段。
MLP仍[128,64]，七个任务隐藏层仍[32]，drama维度仍8，LHUC stop-gradient及任务塔门控开启。
仍30epoch、batch2048、LR0.002、单台ml.m5.4xlarge/100GB；冻结UTC训练10/19～11/17、
验证11/18（2026年）。原固定镜像、源码覆盖包不变，各项验证通过后提交独立名称任务。

## 全 context LHUC 六组容量实验（2026-11-23）

Source: 用户明确基于04提交六组容量配置，沿用此前超参定义。

六组分别仅修改指定项：共享MLP [256,128]、[512,256]、drama embedding16、
全部七任务隐藏层[64]，以及两种MLP分别与e16/task64组合。
基线为04的[context,country]，LHUC字段完整保留ctx_platform、u_deep_link_drama_id、
u_adjust_network、ctx_ip_country；stop-gradient与任务塔门控均开启。
其他项沿用04，包括MLP未改时[128,64]、task未改时[32]、其余embedding维度、
30epoch、batch2048、LR0.002、单台ml.m5.4xlarge/100GB，冻结UTC训练2026-10-19～11-17、
验证11-18及固定镜像/源码覆盖包。六组路径和任务名独立，远端核对原recipe后提交。

## inner_watch_lr003 + 全 context LHUC（2026-11-23）

Source: 用户要求以inner_watch_lr003为配置基础，复用04的LHUC和日期。

采用已发布成功训练b1db1706ac28347cde3e01c0e1d1763db887a2760c92af3051f378539620f582的
experiment-config，配置及冻结合同经COMMITTED摘要校验。保留其8任务、LR0.003、
inner_watch_seconds的0.3有效标签加权、101维输出、选模规则及columns/prefetch2输入管线。
将原context末尾国家拆为country并在主干原位置插回；LHUC=[context,country]，
stop-gradient与任务塔LHUC均开启，四条件特征与04一致。
使用04冻结Manifest/合同与UTC训练2026-10-19～11-17、验证11-18。
该合同包含series_watch_duration_ms；不改变原item_filter和任务指标定义。
使用04固定镜像与LHUC源码覆盖包，30epoch、batch2048、单台ml.m5.4xlarge/100GB。
独立实验输出前缀，原日更与Endpoint不受影响。

## inner-watch LHUC 六组容量实验（2026-11-23）

Source: 用户明确基于inner_watch_lr003 + 04完整LHUC任务8651e039提交六组容量变体。

分别只改共享MLP [256,128] / [512,256]、drama embedding16、全部八个task隐藏层[64]，
或指定三项组合；包含inner_watch_seconds的隐藏层，但其101维最终输出不变。
保持LR0.003、原8任务标签/损失/选模、[context,country]门控、stop-gradient和任务塔LHUC，
30epoch、batch2048、单台ml.m5.4xlarge/100GB。冻结UTC训练2026-10-19～11-17、验证11-18，
镜像与源码覆盖包均沿用8651e039。任务名及输出前缀各自独立。
提交前验证schema、恢复容量改动后与8651e039相等、冻结字节一致，并远端核对基线状态和recipe。

### 2026-11-24：14号账号特征实验配置

Source: 用户指定为14号模型增加account_id。已核实原冻结31个自然日合同均声明u_account_id。

仅修改实验YAML：dataset_config.features和dtypes增加u_account_id/int64，缺失默认值显式为0；
新增account输入组（from=item、input_type=embed、combiner=concat）及独立account embedding表，
将account追加至dnn_inputs。实施选择16维，与本实验drama ID表维度一致；原有其他embedding不变。
词表沿用训练分区拟合机制，不使用验证账号拟合。LHUC仍为context+country。
共享MLP[512,256]、八任务隐藏层[64]、学习率0.003、数据过滤与日期等均保持原配置。
不更改通用schema或模型实现；本地配方见runs/inner-watch-account-id-20261124/config.yaml。

### 2026-11-24：账号特征实验的inner-watch权重0.2变体

Source: 用户明确指定48c03be4为基线并授权提交。

实验配置model.tasks.inner_watch_seconds.loss_weight从
`0.3 * cast(series_watch_duration_ms > 0, "float32")`改为
`0.2 * cast(series_watch_duration_ms > 0, "float32")`。
除新run的输入/输出位置外，将此字段还原后要求配置与48c03be4完全相同；
冻结manifest内容、特征/标签合同、bootstrap和源码overlay逐文件保持一致。
本地配方为runs/inner-watch-account-id-w02-20261124/config.yaml，不修改通用模型实现。

### 2026-11-24：权重0.2的无账号特征对照实验

Source: 用户明确要求基于596ba302移除u_account_id并提交。

从模型input_features和dnn_inputs移除account组，同时移除embed_name.u_account_id、
embed_dim.account以及dataset features/defaults、dtypes中的u_account_id声明。
原始冻结数据与合同保持不变，只是不再投影该字段供模型使用。
恢复这些账号配置及原manifest地址后，配置须与596ba302完全一致；另独立核验该配置
与14号原模型仅inner-watch loss系数0.3→0.2及run位置不同。
本地配方：runs/inner-watch-no-account-w02-20261124/config.yaml，不改模型实现。

## 2026-11-25：模型14独立上线设计

Source: 同日用户授权模型14上线与FYP日更，不开AB实验。

沿用既有 FYP 日更 family、训练/发布角色、冻结输入、运行开关、监控 outbox/catalog
及共享 AppConfig publication lock。原实验配置只为日更设置独立模型名及滚动窗口，
不改变特征、网络、loss、选模目标。独立 Endpoint 的IAM、日志、告警、ready Lambda
及上游/恢复触发器由 doku-iac 的 tf_endpoint stack 管理。

首次从校验封存的原始训练产物导出，校验恢复和导出预测一致性。导出包含
serving_with_model_version 签名。先在私有FYP网络探测版本、行数与有限分数，
再按skill生成追加别名的模型定义，以线上ai-engine同提交编译验证旧列表/新定义
和新列表/新定义组合。持有共享publication lock，按定义→列表顺序CAS发布，
每步等COMPLETE及数据面一致；验证每个ai-engine实例加载、新旧别名RPC可用。
最后启用独立日更触发器并提交首个冻结日更输入。

发布镜像以线上CPU镜像为基础，只加入原实验的3个LHUC模型文件；pipeline另登记新别名。
所有镜像以digest固定，保留source/lock digest及源码overlay校验和。
运行配置、回执和实验结果保持本地及不可变S3，不提交源码仓库。

合同核验补充：原inner_watch_lr003把ctx_ip_country放在context组；模型14将其拆为
独立country组。两者原始输入字段及tensor signature一致，但feature-config不同。
因此首次登记新合同，嵌入模型14导出的原始feature-config/signature/state/manifest
与真实摘要；复用经编译验证的tf-tensors/v1处理器和FYP检索链，不篡改合同来通过
同合同日更检查。后续日更以新别名自己的合同为基准。
