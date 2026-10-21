# Spec 004：CPU 三模型对比需求

状态：已确认需求对应的功能已实现；2026-10-12 整理源码交付。实验配置、权重和报告不随源码发布。

Source: 用户本轮交接及 2026-10-10 全仓审查、配置权威和门禁移除确认。以下具体数据集和超参数保留为原始实验需求背景，不构成运行时默认值。

## 实验范围

- R401：在本机 CPU 比较 XGBoost、shared-bottom DNN、DeepFM。保留已有
  `rerank_tree` 重命名及 `train_rerank.py` 入口合并，不重复重构；三个独立 YAML，
  不引入 common.yaml。新 schema/YAML 为唯一行为依据，不保留旧格式、旧目标、
  旧标签口径或固定 27/1/28 生命周期的兼容路径。
- R402：冻结数据集 `manual-20261009-v4-10d-prod-01`、schema `fyp-rank-features/v4`，
  源 Manifest 逻辑 SHA256 为
  `7a3ed9d1366ad2f7dc49832ae769c700fc0f4d54de9c4491f07ce0e45a4343f3`。
  数据集总计 930,112 行，日期 2026-09-29 至 10-08；执行前重新校验发布对象。
- R403：按 UTC 事件日期，10-01 至 10-06 训练，10-07 验证，10-08 测试。
  草案预期行数分别为 555,695、111,047、111,272，合计 778,014，均为 mask 前行数。
  日期必须有序、不重叠、完整覆盖；不得把 09-29、09-30 加入训练。
- R404：仅训练集拟合词表、归一化及任何学习得到的特征状态；验证集选择 epoch/round；
  测试集仅用于选择完成后的最终评估。实验不做 production refit。

## 数据合同与标签

- R405：通过版本化 adapter 消费 `doku-train-dataset-manifest/v1`，校验源逻辑摘要、
  分片 checksum、行数、实际事件时间、feature/label 合同版本。记录源 URI、合同摘要与
  adapter 版本；不改写上游，不 import 上游实现，不使用 latest 或当前业务存储补数据。
  字节长度、时间边界可从已校验对象获得；标签成熟性不能用猜测值补齐。
- R406：主任务为 completion、fast_swipe、watch_seconds、immersive_click。
  item_filter 使用 video_finish_seen > 0，显式对齐主 Grafana 的外流人口。
  completion 直接用上游标签；fast_swipe = 1 - non_fast_swipe；watch_seconds =
  watch_duration_ms / 1000；immersive_click 直接用上游 0/1，不使用只标正例的 valid。
  三个分类任务按该人口训练，不额外按其 *_valid 筛选；时长另要求非空。
- R407：实验采用上游最大播放进度 >=95% 的完播定义，不重算观看时长比。
  DNN/DeepFM 时长任务用 softmax_soft：0～300 秒、5 秒间距、61 个输出、normal
  soft label、sigma=2 秒，predict_fn=expectation。另加 play10/play30/play50
  三个辅助分类任务，分别严格大于 10/30/50 秒，仅在时长有效时计算。
  XGBoost 仅训练四个主任务，时长须用回归目标而非二分类。
- R408：[NEEDS CLARIFICATION: 成熟性证据] producer 的 as_of_ms 是数据结束日次日
  00:00 UTC，不是标签完成处理时间；原始 Manifest 无 label_maturity_hours。
  真实数据训练前需落实版本化标签窗口及完成归档证据；可先完成 adapter 和 fixture 验证，
  不以猜测成熟性启动真实实验。此项是输入完整性查证，不是模型指标质量门禁。

Source: `../doku-offline/fyp-rank-samples/src/fyp_rank_samples/dataset.py::build_manifest`；
`../doku-offline/fyp-rank-samples/src/fyp_rank_samples/runtime.py`；
`../doku-offline/fyp-rank-samples/contracts/feature-spec-v4.yaml`；
`../doku-reco-data-pipeline/impression-label/internal/labelbuilder/build.sql:446`；
`recommend/train/models/rerank/data_utils.py::completion_labels`。
上述为本地源码证据，尚未确认与 PROD 已发布数据的部署 revision 完全一致。

## 同口径特征与模型

- R409：实体为 drama。DNN 与 DeepFM 使用相同特征编码和训练集拟合的共享 drama
  词表/embedding；候选 i_drama_id、兴趣历史和快划历史共享表。首轮使用 8 维。
  快划 u_skipped__drama_id 使用 mean；兴趣 u_drama_interests__id 使用 weighted_mean，
  weight_feature 指向 u_drama_interests__weight。XGBoost 使用同一原始历史派生的标量，
  作为独立基线，报告明确编码差异，不把全部差异归因于模型。
- R410：排除原始用户 ID、u_fyp_confidence、线上 score/position、
  u_finished__drama_id、u_immersive_enter__drama_id。后两项训练集全部缺失。
  不从当前业务存储回填，不将缺少映射的 episode 历史转换为 drama。
- R411：采用 Finder 式交互网络：共享 embedding 输入分别进入 DNN 和显式 FM 配对，
  FM 标量与 DNN 输出拼接后进入各任务塔，不强制经典 DeepFM 三路 logit 相加。
  fm_layers 采用 fm_type: embed + features 配对声明；首轮配对为 item_id ×
  hist_drama_interest、item_id × hist_drama_skip、item_id × user_profile、
  item_tag × user_profile。user_profile 为 onboarding v2 题材，item_tag 为 item v2 题材，
  二者共享题材表，题材表与 drama 表独立。
- R416：schema 为运行时权威；采用 Finder 式声明配置，公共与模型 schema 分工，
  拒绝未知字段、冲突及无效引用；示例中的业务字段不能直接移植。
  目标集合、label/filter/loss/metrics、输入编码、网络结构、日期与执行阶段均由
  schema 约束、YAML 选择；本 Spec 的具体实验值不是框架硬编码常量。
  Python 类型不得继续作为一套独立且可能冲突的权威验证定义。
- R417（2026-10-21 Spec015 更新）：weighted_mean 与 mean 为历史聚合方式，通过 history_pooling 选择；combiner 仅组合组内字段。
  仅 weighted_mean 接受并要求 weight_feature。校验 ID/weight 配对、有限非负权重；
  合并重复 ID 权重，padding/零权重不参与，OOV 保留权重；缺失、空列表、全零状态
  显式处理。采用稳定的逐样本归一化，不加 log/温度变换或 pooled 向量 L2 归一化。
- R418：四主任务初始相对权重各 1、辅助各 0.2；先按初始权重训练一次，观察各主任务 loss 后再调整，不做训练前自动校准。
  各任务按有效样本平均，不能将 softmax 桶维度计入样本均值分母。
  最佳 epoch 由四主任务验证加权 loss 选择，辅助任务不参与选择分数；校准细节见设计。

- R419：user_statistics 首轮使用七类行为的 7 天 all/fyp 计数共 14 项，
  加 u_active_days_7d，共 15 项；14/28 天字段可被 schema 描述但本轮不启用。
  u_statistics_as_of_ms 仅作时间点审计。默认值由配置声明；本轮不默认添加统计存在/缺失分组指标；未来需要时使用任务 metrics.filter，不新增模型输入。
- R420：两神经模型统一 8 维 embedding、公共层 [128,64]、任务层 [32]、ReLU，
  首轮无 normalization/dropout；差异为已确认的 FM 交互分支。
- R421：pooling 按有效长度 mask，padding 不参与分母；缺失、空列表、全零权重
  输出零向量；按用户最新决定不生成额外状态输入。
- R422：XGBoost 时长采用秒数平方误差回归。神经网络仅软标签生成阶段将越界值
  裁至 300 秒；原始时长用于双方 MAE/RMSE，并报告越界率。

## 执行与验收

- R412：严格验证完整实验配置，拒绝未知字段与未实现语义；输入 URI、PROD 只读身份、
  本地输出位置显式传入。仅允许读取本次指定 PROD 发布数据用于本机实验，
  不启动云训练、不上传或发布模型、不修改生产资源。
- R413：运行产物 create-only、run-scoped，保存 checkpoint、metrics、feature state、
  lineage；额外记录未提交源码摘要、配置/依赖摘要、数据切分、目标语义和 adapter 版本。
  不把当前 HEAD 单独当作未提交工作区的完整源码身份。
- R414：分类报告 AUC、logloss、valid rows、positive rate，时长报告秒级 MAE/RMSE；并记录机器 CPU/内存、
  耗时、吞吐、峰值 RSS；不可比较或单类目标显式报告，不产生虚假排名。
- R415：覆盖 checksum/版本/成熟性拒绝、独立切分、训练集独占 fit、历史缺失/空值/OOV、
  目标选择、DeepFM 显式 pairwise 等价和共享 embedding 梯度、checkpoint 回读一致性、
  三模型 fixture smoke、配置驱动行为及旧格式明确拒绝测试；通过 make check 后才验证真实输入和启动实验。

## 后续流程

整体需求与设计已确认；先审阅 config-contract.md 的字段实例化提案，再生成 tasks 并实施。
精确 S3 URI、读取身份与成熟性证据在真实输入验证前必须落实。

Source: 2026-10-10 本对话 Q6～Q27 及 pooling 组件接口确认。后续明确决策取代早期草案冲突项。

Source: 用户后续明确要求以新配置为准，撤销旧行为/格式兼容要求。

## 全仓审查后的补充要求

以下条款取代历史 Spec/architecture/README 中冲突行为，不将旧实现测试视为需求来源。

- R423：直接移除指标质量门禁及其流水线依赖、配置和测试：包括任一目标不可评估即 BLOCK、
  AUC 下限、隐藏的 0.02 下降告警。首轮不建设可选门禁框架。不可计算指标逐项保存 null、
  原因、有效数与类别数；单类 AUC 不影响可计算的 logloss，不阻断其他任务或产物保存。
- R424：任务定义贯通数据、模型、损失、评估和 checkpoint；删除全局 TARGETS 与固定输出数。
  选模无法计算时记录 selection unavailable、保留可用 checkpoint，不伪称最佳，不隐式换指标。
- R425：取消没有后端依据的维度、层数、轮次等任意上限和资源固定值；schema 仅保留数学定义、
  已实现能力和后端真实限制。词表策略/容量、缺失/OOV、变换、优化器与资源策略显式声明；
  拟合得到的基数由 feature state 提供，不要求用户填写后又被覆盖。
- R426：评估采用有效样本数正确聚合，不能平均 batch 均值或因其他任务缺失重标本任务权重。
  指标与训练轨迹采用有界存储方案；资源配置应实际约束读取、临时文件、缓存和训练。
- R427：公共 Batch/任务/指标接口不依赖具体模型。模型由注册接口装配，CLI 不复制模型训练流程。
  原生树模型类别编码和历史标量方案显式报告；不把 QuantileDMatrix 宣称为已实现外存保证。
- R428：验证源合同语义及对象完整性；配置显式给出允许的 S3 范围和身份，正确保留对象 key。
  保留精确 Manifest 要求，移除根据 latest 字符串或 Manifest 父目录推导的任意限制。
- R429：checkpoint 与配置、任务、feature state、代码（含未提交内容）、依赖身份绑定；
  不以 unknown 伪造可追溯性。恢复须覆盖 RNG/优化器/进度并可验证，区分回读与真正续训。
  存储应正确处理 create-only 冲突、校验和、路径、流关闭和部分写出，只有完整产物才声明成功。
- R430：迁移旧测试及文档；不保留强制草案状态、固定任务/日期、Pydantic schema 快照权威、
  源码文本禁词等测试。保留实际依赖边界、数据隔离、数值与存储正确性验证。

Source: 2026-10-10 全仓只读审查；用户确认其他项基本同意并明确要求直接移除门禁。

## 字段级确认补充

- R431：沿用 embed_name 的“源特征→共享表”映射，embed_dim 按表名及 '*' 声明；
  label/mask/loss_weight 分离，表达式采用受限 AST，权重不被分母抵消。
- R432：历史仅输出 pooling value，不自动生成 missing/empty/zero_weight 模型输入。
  mean 保留重复 ID；校验和有效长度仍在组件内部处理。
- R433：hist_options 显式限制100项：快划取交付顺序末尾，兴趣合并权重后取最高权重，
  同权重按原始 ID 稳定排序。三模型共享截断结果，不声称取时间上最近。
- R434：树历史标量为 skip_count、candidate_in_skip、interest_count、candidate_interest_share
  及 genre_overlap_count，不附自动状态特征；原始 ID 比较先于 OOV 编码。具体空值语义见配置合同。

Source: 用户随后确认 Q1～Q5 及 Q6 截断建议；[字段合同与示例](config-contract.md)。

- R435：遵循用户 Finder YAML 列表风格；移除 feature_encoding/vocabularies 配置。
  feature_default_values 声明缺失填充；feature_transformations 统一变换，并支持
  rerank_prom 的 [*] + feature_list 模板。共享表与类别词表从已有输入声明推导，
  不重复配置字段集合；保存训练拟合状态，不自动添加缺失输入特征。

Source: 用户对配置示例的最新修正。

- R436：用户最新决定取代旧缺失指示设计：直接使用 feature_default_values，数值一般为0，
  需要区分时显式设为-1等值；不自动生成 *_missing、state_features 或额外缺失 embedding。


## 最新数值与 YAML 形式确认

计数仅 log1p 并生成 _log，其他连续值保留原值；本轮移除 standardize。
feature_default_values 只允许 features 中原始字段，先填默认值再转换；派生字段无默认值。
文本值加单引号，数值/布尔保留原类型；loss_weight 使用字符串表达式。
norms 无归一化用空字符串，省略 dropout；soft_label 使用 Finder 的 func/scale/sigma，
裁剪放在 label 表达式，metric_label 保留原始秒数，expectation 按 scale 换算回秒。
保持扁平特征结构，不增加 apply_to/from 或 item 域。本条替代先前标准化及派生默认值建议。

- R437：删除独立 evaluation 块及 test_usage/slices；测试日期使用 split，评估指标使用
  tasks.*.metrics。保留训练、验证、测试隔离，不默认新增分组指标。


## 首轮搜参与模型比较（用户已确认）

- R438：首轮4个DNN + 4个DeepFM + 1个XGBoost，共9组运行。神经模型仅搜索公共层
  [128,64]/[256,128] × 学习率0.0003/0.001；batch_size=2048、时长训练权重1.0。
  其他训练参数继承各自基准配置，不将原16+16+1计划作为默认执行范围。
- R439：验证集按task比较。completion/fast_swipe/immersive_click统一使用logloss与AUC，
  相对同一XGBoost基线仅在loss严格更低且AUC严格更高时明确判胜；一好一坏为取舍，
  相等或部分相等不算严格判胜。指标不可计算则该项不可比较，不阻断训练。
- R440：watch_seconds统一比较基于原始秒数标签的MSE，越低越好；无AUC。
  MAE/RMSE仍报告。神经softmax_soft训练loss单独记录，不与树回归loss直接比较。
- R441：先提交9组验证结果，逐task展示涨跌，不通过自动总分决定精搜模型。
  再选择至多一种神经模型精搜；不自动启动第二阶段。参数/模型冻结后再评估测试集并给出报告。
  最终每种模型使用一个明确选定的checkpoint，不拼接不同trial的task形成虚构赢家。

Source: 用户确认4+4+1及按task loss/AUC比较规则；搜参配置（本地实验记录，不随源码发布）。


## 实施授权与模型目录（最新确认）

DNN 与 DeepFM 都使用 model_name: 'rerank'，不创建 models/deepfm。
空 fm_layers 表示 DNN；显式配对表示 Finder 式 DeepFM。模型插件只注册 rerank/rerank_tree。
搜参使用基准配置文件名 dnn/deepfm 作为实验身份，不能按 model_name 合并结果。
用户已授权实施训练与4+4+1搜参代码，并在本机CPU各运行一次XGBoost、DNN检查链路。
不自动执行整轮9组搜参；真实发布数据的成熟性验证与本地fixture smoke区分报告。


## 当前实现状态

本轮已开始并接入运行时schema、表达式、rerank内FM、树任务、独立进程训练和搜参；
之前“仅设计、运行时schema不存在”的记录为当时状态，当前以verification的最新检查为准。
真实v4成熟性证据仍未完成，生产数据训练入口拒绝该未验证输入；本机验证使用合成fixture。


## 2026-10-10 真实数据成熟性核查更新

此前成熟性未决项已由历史证据闭合，详见真实v4核查（本地实验记录，不随源码发布）：
192小时归档均有更早的完整构建日志，778,014行标签逐项比对零差异，窗口和生成时间零异常。
此前“真实数据一律拒绝”的临时状态已失效。Manifest缺少成熟性字段仍如实记录，
外部证据绑定冻结Manifest并随本次真实run保存，不增加固定等待期或fixture白名单。
按用户要求只执行单次XGBoost与DNN，不启动整轮搜参。

## 2026-10-10 分桶 embedding 实施确认

用户批准按当前 YAML 补齐派生桶 embedding 支持，本机各跑一次 DNN/DeepFM，
只用验证集与旧同参数神经模型、旧搜参结果及 XGBoost 对比 loss/AUC。
不启动搜参、不读取测试集。numeric 桶各用独立 4 维 embedding；
时长/集序/总集数边界及计数 log2 桶见 config-contract.md 最新确认项。
此项替代神经网络此前直接输入 `_log` 浮点及原始时长/集序/集数的设计，
不改变 XGBoost 的已冻结基线配置。

## 2026-10-11 完播单目标消融

用户要求 DNN/DeepFM 仅完播 loss_weight=1，其余六个 task 全部为 0。
当前两份神经 YAML 及 examples 已同步，selection 仅使用 completion loss，
防止未训练目标影响 checkpoint 选择。search 固定项中的 watch_seconds 权重同步为 0，
本轮不执行 search。其他特征、分桶 embedding、网络、优化器及日期保持不变。
保留其他输出头与指标用于运行完整性记录，但这些头没有对应监督损失，
不能作为有效的多目标效果与旧模型比较。仅本机各训练一次，测试集不读取。
Source: 用户本轮明确要求仅使用完播 loss，重新训练 DNN 和 DeepFM。

## 2026-10-11 DeepFM 正则化 A/B/C 单因素实验

用户确认按 A、B、C 顺序各跑满 20 轮，仅完播监督与选模，patience=20，不读取测试集。
三组均从无 dropout 的完整 20 轮基线独立出发：
A：共享 output_layer 和 completion tower 的 layer_dropout=0.9（保留率，丢弃 10%）；
B：仅 AdamW weight_decay 从 0.0001 调至 0.001；
C：仅共享 dims=[64,32]、completion dims=[16]。其他配置固定，不累计前组改动。

新增 layer_dropout 沿用 Finder 保留率语义，范围 (0,1]，缺省 1；隐藏层激活后应用，
仅训练生效，eval 禁用。不作用于 embedding pooling、FM 点积或最终 logits。
未引入 regs/额外正则损失；B 是既有 AdamW 衰减，不宣称等价于 Finder kernel L2。
独立配置保存在 configs/model-comparison-20261001/regularization/。
Source: 用户本轮明确授权 A/B/C 依次本机训练。
