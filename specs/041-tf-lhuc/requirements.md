# Spec 041：TF rerank LHUC 结构

状态：需求已确认，用户指定直接复用 Finder LHUC 结构并继续实施。

Source: 2026-11-20 用户要求新建分支，为模型增加 LHUC，参考并迁移 finder-models 当前实现。
随后确认：“确定 直接复用finder的lhuc结构”。
随后指定：“把国家这个特征加入lhuc 其他特征不加”。本次实验条件仅为
`ctx_ip_country`，通用 LHUC 框架仍由 YAML 声明输入组。

## 实施前状态与参考来源

- 已创建 `codex/lhuc-structure`，基于 `912ded9`；原有 Spec 009 未提交修改保留。
- 本仓 `recommend_tf/train/models/rerank/rerank_model.py` 的 `dnn_layer`、
  `target_layer` 已保留 LHUC 计算，但 `rerank_core_model` 未传入条件输入，
  `model_params.py` 与模型 schema 尚未支持对应配置。
- 参考本机 `finder-models` 的 `recommend/train/models/rerank/`，
  HEAD 为 `0abdcc692a84319833d871e30c0bfd7ee6a22080`。
  `rerank_model.py` 的 `dnn_layer`、`target_layer` 和 core 调用，以及
  `config.schema.yml` 的三个配置项为本次依据；不泛化迁移其他 Finder 模型变体。

## 已确认需求

以下范围已确认；计算结构、默认值、梯度行为直接沿用所列 Finder 来源。

1. 扩展 `recommend_tf/train/models/rerank`，支持现有 DNN 与带 FM 的模型；
   不创建新模型类型，不改动 Torch 或树模型。
2. 在模型 schema 中声明并校验 Finder 同名配置：
   - `lhuc_inputs`：有序、不重复的输入组名列表，默认 `[]`（关闭）。
   - `lhuc_share_embed`：布尔值，默认 `true`；沿用 Finder 语义，
     对拼接后的 LHUC 条件输入执行 `stop_gradient`，仅阻断门控支路回传到输入，
     主干梯度不受影响。`false` 时允许门控支路更新共享输入。
   - `target_layer_use_lhuc`：布尔值，默认 `false`；控制任务塔是否使用同一组条件输入。
3. 条件输入来自 YAML 已声明的 `input_features`，在现有编码和历史 pooling 后按列表顺序拼接；
   不自动挑选特征、不引入原始 user_id 或额外 embedding 表。未知组名或不合法类型显式失败。
4. 按 Finder 公式 `x * (2 * sigmoid(Dense(condition)))`，在共享 DNN 的每个 Dense
   之前缩放其输入。每层独立学习门控参数，输出宽度匹配该层输入宽度。
5. 开启任务塔 LHUC 时，每个任务的隐藏层 Dense 前及最终 logits Dense 前均应用独立门控；
   继承任务 `stop_grad` 语义。FM 计算本身保持现有语义。
6. 保持 Finder 初始化、变量作用域和正则位置语义：隐藏层门控沿用该层 `regs`；
   任务塔最终输出前门控不另加 layer regularizer。现有全局正则策略继续生效。
7. 默认关闭必须保持现有模型变量、预测与训练行为；训练、评估、恢复与导出使用一致图结构。
   不承诺旧无 LHUC checkpoint 可以直接恢复为新增 LHUC 结构。
8. 本轮只做本地实现和验证；实验特征选择由实验 YAML 声明，不修改生产日更配置或启动云任务。

## 验收要求

- 固定合成输入与权重，对照 Finder 公式验证共享层和任务塔数值、门控维度及梯度。
- 验证 `lhuc_share_embed` 与任务 `stop_grad` 的梯度边界。
- 验证默认关闭与显式空输入的基线一致性，以及非法配置报错。
- CPU 最小训练、checkpoint 恢复和导出预测一致性通过；记录实际命令与结果。
- 更新来源追踪与相关 Spec，明确取代 Spec 024 中仅针对 LHUC 的“不迁移”限制。

## 下一步

用户已指定直接复用 Finder 结构；[设计](design.md)记录这一决策与必要接线，
不引入新的结构选项。实施与验证见 [任务](tasks.md)、[验证](verification.md)。

## 单次云实验授权（2026-11-20）

Source: 用户指定以 `doku-exp-fyp-us-other50-20261120-0a923a14` 为基线，
“item_filter用所有国家”“增加lhuc的结构 输入特征是国家”，并要求“启动一次训练”。

本次授权覆盖一次 SageMaker CPU 训练及其必要的不可变配置/源码覆盖包上传。
仅将过滤改为 `video_finish_seen > 0` 并启用国家 LHUC，其余训练参数、冻结数据、
依赖镜像和资源复用原任务。不发布模型、不更改日更或 Endpoint。

Source: 用户随后要求“再开一个训练任务，把context和country都加入lhuc”。
授权再创建一个独立实验，基于上述全国家任务，仅将 LHUC 条件组改为
`[context, country]`；其他设置保持不变。原国家专用实验继续运行。

## 国家 LHUC 容量实验授权（2026-11-20）

Source: 用户要求基于 `03_All_country-LHUC`，分别使用 `06_All_mlp256` 的 MLP 配置、
`05_All_e16` 的 embedding 配置以及两者组合，提交三个训练任务。

授权三个独立 SageMaker CPU 训练及必要的不可变 recipe 上传；复用国家专用 LHUC、
冻结日期与分片、训练设置及资源。实验配置与运行产物保留本地。

## 三个独立对照实验（2026-11-22）

Source: 用户指定基于 03，分别增加 u_adjust_network LHUC 条件、扩大共享 MLP、task layer
提高到 64；明确三项相互独立，随后确认 MLP 使用 [512,256] 两层并提交。

授权三个独立 SageMaker CPU 训练及必要 recipe 上传，其余保持 03 配置和冻结数据。

## 两组容量融合实验（2026-11-22）

Source: 用户要求基于 03 提交 mlp512 + task64 + embedding16 和
mlp256 + task64 + embedding16 两个训练任务。
授权两个独立 CPU 云训练及必要配方上传，继续使用此前超参定义与冻结数据。

## 国家、渠道与 deep link LHUC 实验（2026-11-23）

Source: 用户要求再训练一个模型，LHUC输入为ctx_ip_country、u_adjust_network、
u_deep_link_drama_id。沿用刚比较的03/04/12配置，授权一个独立CPU训练及必要recipe上传。

## 全 context LHUC 六组容量实验（2026-11-23）

Source: 用户指定基于04_All_context-country-LHUC分别提交256、512、e16、task64、
256+e16+task64、512+e16+task64六个任务。授权六个CPU训练及必要的不可变配方上传。

## inner_watch_lr003 + 全 context LHUC（2026-11-23）

Source: 用户要求找到inner_watch_lr003配置，加上04的LHUC结构，使用与04相同训练和评估日期，
提交一个训练任务。授权单次CPU实验及必要recipe上传。

## inner-watch LHUC 六组容量实验（2026-11-23）

Source: 用户引用8651e039，要求分别提交256、512、e16、task64、256+e16+task64、
512+e16+task64六组训练。授权六个独立CPU任务及必要不可变recipe上传。

### 2026-11-24：14号模型增加账号特征

Source: 用户要求基于14_InnerWatch_lr003_context-country-LHUC_mlp512-e16-task64增加account_id，
并核实训练数据流是否已提供。

本实验显式启用合同字段u_account_id（user_features.account_id，int64），作为独立类别embedding
接入共享DNN。此项是用户指定的身份特征实验，不改变默认配置，也不自动选择其他身份字段。
沿用14的冻结日期、MLP、任务塔和LHUC条件。当前交付本地配置，不启动云训练。

### 2026-11-24：账号特征实验的inner-watch权重0.2变体

Source: 用户明确要求基于48c03be4将inner-watch loss权重改为0.2并提交训练任务。
仅改变inner_watch_seconds的loss系数，保留有效标签mask、其他任务、网络容量、账号特征、
LHUC、选模设置与冻结数据；使用独立run路径提交一个实验任务。

### 2026-11-24：权重0.2的无账号特征对照实验

Source: 用户引用596ba302并要求基于该任务提交不加入u_account_id的训练。

移除账号特征输入及其独立embedding，保留inner-watch loss系数0.2、有效标签mask、
其余模型设置与冻结数据，提交单独任务作为账号特征消融对照。

## 2026-11-25：模型14 FYP独立候选日更

Source: 用户要求按 doku-rec-deploy skill 上线
`14_InnerWatch_lr003_context-country-LHUC_mlp512-e16-task64`，启动日更，加入 FYP 任务组，暂不上 AB。
本授权取代上文仅本地验证的限制，仅适用于该独立候选。

- 初始模型来自成功任务 `doku-exp-iw003-ctx-lhuc-mlp512-e16-task64-20261123-60bb5168`，
  保持封存输入与原选模结果（epoch 4）。
- 新别名 `inner_watch_lhuc_m512_e16_t64`，独立 Endpoint
  `doku-prod-fyp-inner-watch-lhuc-m512-e16-t64`，加入既有 `doku-prod-tf-rerank-daily` 家族。
- 保持 MLP [512,256]、drama embedding 16、task layer 64、context+country LHUC、
  学习率0.003、inner-watch loss系数0.3、无account特征及原八任务。
- 导出 completion、fast_swipe、immersive_click、watch_seconds、inner_watch_seconds。
- 日更30天训练、1天验证，30 epoch，原 validation loss 选模策略。
- 不改默认模型或实验分桶；调用方新别名绑定作为后续开实验前的接线交付。

## 2026-11-26：后续日更训练改为15轮

Source: 用户确认模型已上线且已有实验，要求不影响当前线上，把下次训练改为 `max_epochs: 15`。

仅调整 `inner_watch_lhuc_m512_e16_t64` 后续日更的最大训练轮数，取代上述上线时30轮配置。
保留当前Endpoint、模型版本、AB实验、已完成训练及其不可变配置；不手动启动训练。
30天训练＋1天验证窗口、网络、loss、特征、导出任务和validation loss选模规则保持原值。
