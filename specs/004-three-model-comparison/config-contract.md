# Spec 004：字段合同与完整配置示例

状态：配置驱动运行时已实现，以下保留字段设计与决策依据。实际可接受字段以公共及模型 schema 为准。

Source: 已确认的 Spec 004 设计、运行时 schema 与合成 fixture 验证；2026-10-12 功能交付整理。

## 阅读入口与范围

公共 schema 位于 `recommend/train/comm/config.schema.yml`，模型 schema 位于各插件目录；
搜参 schema 为 `recommend/train/comm/search.schema.yml`。每份运行 YAML 独立声明数据身份、
日期、特征、任务和输出位置，不依赖 common.yaml。

实际实验 YAML、结果报告与模型产物仅保留本地，不随源码发布。自动化验证使用
`recommend/train/tests/synthetic_config.py` 与 `experiment_fixture.py` 生成合成配置和数据，
不依赖生产账号、实验文件或已训练模型。下面提及具体日期、维度和分桶边界的内容是设计背景，
不是框架默认值；重复运行必须使用新的 run_id。

## 配置权威、字段与默认值

公共 schema 声明数据、资源、表达式、特征状态和任务公共字段；插件 schema 扩展网络或树参数。
加载时检查结构、组件能力、引用与维度，输出带版本的 resolved config。运行时只消费已解析计划。
所有对象拒绝未知字段；不根据特征或目标名字猜算法。共享表配置不得出现同一字段对应不同词表。

| 字段 | 合同与所有者 | 缺省处理 |
| --- | --- | --- |
| config_version / model_type / model_name | 公共；配置版本、插件类型、业务模型名 | 必填；示例提案版本 doku-train-config/v2 |
| dataset | 公共；精确源 Manifest、版本、逻辑摘要、adapter、读取身份、允许前缀 | 必填；无生产身份/URI 默认值 |
| split | 公共；timezone、train/validation/test 的 from/to 自然日（两端包含） | 必填；有序互斥；不读区间外日期 |
| execution | 公共；线程/worker/prefetch、内存、校验缓存路径与容量 | 显式；0 worker 仅为本轮选择 |
| seed / batch_size | 公共；随机流根种子与批量 | 显式 |
| output | 公共；root/run_id，create-only 是存储合同 | 必填；不提供覆盖开关 |
| features | 公共；原始模型输入字段白名单 | 必填；dtype 来源于已校验源合同 |
| item_filter | 公共；逐样本人口筛选 | 本轮显式；源字段表达式 |
| feature_default_values | 公共；features 中原始字段 → 转换前的缺失默认值 | 不声明则保留 NULL；不得改变标签资格 |
| feature_transformations | 公共；expr 单赋值与 feature_list，[*] 逐项替换；本轮计数仅 log1p | 不使用时 []；feature_list 默认 []，保留声明顺序 |
| hist_options | 公共；按 input_features 组名声明截断与重复 ID 策略 | 本轮两组显式；没有声明则不截断，不能隐式取最近 |
| input_features | 公共字段组合 features/combiner（concat/sum/avg）、可选 weight_feature；神经模型另声明 history_pooling | 必须能解析为已声明输入；不自动加入 DNN |
| embed_name / embed_dim | 神经插件；字段→表名、表名→维度（支持 '*'） | 示例显式；不手填拟合后基数 |
| embed_initializer / args | 神经插件；初始化组件及参数 | 示例显式 |
| dnn_inputs | 神经插件；有序组名列表 | 必填；顺序保存到 feature state |
| output_layer | 神经插件；dims/acts/norms（本轮省略 dropout） | 显式；每层 acts/norms 与 dims 长度相同 |
| fm_layers | DeepFM 插件；fm_type=embed 与显式向量配对 | DeepFM 显式；DNN 不需要该字段 |
| tasks | 公共任务身份/表达式/指标 + 插件头或 objective | 必填；没有默认任务 |
| max_epochs / early_stopping_patience / optimizer | 神经插件训练参数 | 示例显式；patience 依据 selection，不是质量门禁 |
| history_scalars / tree_inputs / model_params | 树插件；公共历史组件派生标量、树输入、后端参数 | 示例显式；不生成神经 embedding |
| selection | 公共选模计划；神经网络加权 loss / 树按任务指标 | 显式；无法计算时按 design 记录 unavailable |

`features` 不包含标签和纯审计字段。表达式依赖分析可读取标签、u_statistics_as_of_ms 等，
但只有 input_features/tree_inputs 引用的模型特征进入网络。标签派生 watch_seconds 只用于任务。
原始模型输入白名单不能把源合同标记的 label/audit 字段转换为模型特征，避免泄漏。

层配置只承诺此次使用的组件；以后扩充合法能力通过 schema 与组件注册共同完成。
表内技术字段的具体拼写是本次审阅提案，不声称来自 Finder。

## 表达式合同（已确认范围）

支持字段、常量、括号、+ - * /、比较、and/or/not；函数为 abs、minimum、maximum、clip、
log1p、log2、sqrt、round、cast、is_null、is_finite、fill_null、where。
函数固定签名并检查 dtype，cast 的目标为明确允许的标量 dtype；不允许任意动态函数、
Python 属性/对象访问、循环或跨样本聚合。本轮不提供 standardize；计数使用 log1p，其他连续值直接输入。
词表由共享表/类别输入声明推导，不出现在表达式中。

item_filter 先作用于原始字段；派生字段按声明顺序求值并检查依赖，tasks 可引用原始或派生标签。
表达式可用性按执行阶段检查，不能在 item_filter 引用尚未计算的字段。
NULL 传播，三值布尔逻辑；最终 filter/mask 的未知值视为 False。
where 选中分支之外的无效值不污染输出；被 mask 排除的标签在 loss 前剔除，不能乘零掩盖 NaN。
保留行上的除零、非法数值变换等应给出具名错误，不静默转零。

任务字段：label 为表达式，mask 为独立有效性表达式，loss_weight 为字符串表达式，求值为有限非负常数或逐样本权重；
首轮不向权重表达式暴露 logits/predictions。num_outputs、loss_type、predict_fn、soft_label、
metric_label 与 metrics 由任务声明。指标 mask 为 task mask 与指标额外 filter 的交集；
metric_label 省略时使用 label。没有有效样本只记录状态，不产生质量门禁。

L_task = sum(mask * loss_weight * loss_per_row) / valid_count；总损失为各任务之和。
softmax_soft 先对输出维求和，再对有效样本平均。验证 task_losses 指向未乘训练 loss_weight
的每任务平均 loss，selection 中的系数只乘一次。首轮主任务权重均1，辅助0.2。
分类三个主任务 mask=True 是此次配置选择；时长和辅助只要求 watch_seconds 非空。
NULL 分类标签如落入有效人口应报数据问题，不偷偷增加不同口径的 mask。

## 历史处理顺序与状态（已确认）

1. 保留源 NULL/空列表状态；校验 ID 类型、ID/weight 配对、有限非负权重。
2. 快划保留重复位置，按交付顺序取末尾100个，不称时间最近。
3. 兴趣按原始 ID 合并权重，按权重降序、同权重原始 int64 ID 升序保留100个。
4. 对选中历史做稳定归一化；padding/零权重不参与 pooling，OOV 参与。
5. 训练样本经过 item_filter 和同样截断后，用候选与历史联合拟合 drama 词表；验证/测试只查表。
6. DNN/DeepFM 查共享表后 pooling；XGBoost 在相同截断结果上按原始 ID 派生标量。

稳定计算细节：可先以原始行最大权重缩放再合并，避免重复项求和溢出；这不会改变 top_weight
排序或归一化结果。用 float64 计算并验证有限性；不通过任意 epsilon 抹掉极小的正权重。
源兴趣列表已被上游裁过，本次 top100 仅在已发布列表内部选择，不能声称恢复完整兴趣集合。

每组只输出 value [B,D]，不输出 is_missing/is_empty/is_zero_weight 特征。
默认空列表按 feature_default_values 声明，空历史/全零权重 pooled value 为零向量。
源 ID/weight 配对与有限值校验仍保留，padding 不参与分母，OOV 仍参与。
保留索引与张量列顺序记录在 feature state，padding 不是源业务 ID 0。
mean 的 [A,A,B] 仍为 (2eA+eB)/3，不默认去重。共享表是每个模型内部共享，模型之间参数独立训练。

## XGBoost 输入合同（已确认语义）

| 标量 | 基于截断结果的定义 |
| --- | --- |
| skip_count | 有效 ID 位置数，重复计数 |
| candidate_in_skip | 原始候选 drama ID 是否出现在快划历史 |
| interest_count | 合并后正权重的不同 drama 数 |
| candidate_interest_share | 候选合并权重 / 截断后全部有效权重之和 |
| genre_overlap_count | 候选题材与用户题材的去重交集大小 |

只给原始 features 声明默认值，不给树派生标量配置 default。默认空列表产生的计数/命中为0，
全零兴趣占比为0，不附状态特征。
ID 匹配先于词表/OOV 映射，不把不同未知 ID 当同一实体。
候选 ID 缺失时候选命中/占比为 NaN，不伪造未命中；题材列表使用已声明的 [] 默认值时，overlap 为0。
只启用 i_drama_id、ctx_platform 两个树类别输入，禁止 ID 连续数值解释。
保留与神经网络相同的原始数值字段及缺失状态；不把已编码历史标量自动附加到神经模型。

## 本次实例化的建议项（供审阅，不冒充已确认细节）

1. **已按最新要求修订**：移除 feature_encoding 和 vocabularies 配置；缺失值由
   feature_default_values 声明，变换统一 feature_transformations。history_scalars/tree_inputs
   仍是树模型已提出的字段写法。
2. **已确认的数值编码**：原始数值按 feature_default_values 填0后，计数执行
   log1p 并生成 *_log；quality/heat/时长/集序号/剧集数/active_days 保留原值。
   不再标准化，不给转换结果填默认值，不增加缺失指示输入。
3. **词表内部状态**：不提供独立 vocabularies 配置。神经模型从 embed_name 推导共享关系及
   联合字段，树从 categorical 输入推导类别字典。保留所有训练类别，按原始类型稳定编号；
   映射、OOV 和类型写入 feature state，不能误以为删除配置就是无需保存实际词表。
4. **题材 pooling**：item_tag/user_profile 采用 mean、保留重复位置，不附额外状态标记；
   该选择延续已确认 mean 组件，但题材历史聚合使用 history_pooling: mean，单字段 combiner: concat（Spec015 更新）。FM只取8维 value。
5. **树选模**：独立 Booster 分别用验证 logloss（分类）/RMSE（时长）选择轮次；不把不同轮次
   的独立模型硬拼成统一 epoch。四主任务加权验证 loss 专用于共享训练的神经模型。
6. **资源/初始化**：示例缓存20GiB，embedding uniform[-0.01,0.01]，其他训练超参沿用旧草案。
   这些是可调整初值，无隐式框架默认；实际执行前验证磁盘和容量。

测试集由 split.test_from/test_to 声明；指标由 tasks.*.metrics 声明。
不设置独立 evaluation、test_usage 或 slices。训练/验证与最终测试的执行顺序遵循用例，
测试集不参与拟合或选模。需要分组指标时再通过任务指标 filter 显式声明，本轮不默认添加。
真实数据成熟性仍按 design.md 的未决证据处理；本轮没有访问 PROD 或启动实验。

## 本轮检查边界

检查 YAML 可解析、三份文件独立、原始字段属于 v4、共享截断相同、任务数及配置引用等。
运行时 schema 已实现；独立合成配置与执行验证见 [验证记录](verification.md)。

## 最新确认：遵循 rerank_prom 的配置形式

Source: 用户要求使用 []、feature_default_values、feature_transformations 与 feature_list；
Finder rerank_prom/rerank_model.py:55–73、config.schema.yml:147、204–224。

Finder 是 expr.replace("[*]", feat) 的模板替换，不是正则匹配字段名。不增加另一套 regex 语法。
所有标量列表和配对列表采用 [] 排版；transformations 等对象列表仍用 -，与用户示例一致。

```yaml
feature_default_values:
  u_expose_count_7d_all: 0
feature_transformations:
  - expr: '[*]_log = log1p([*])'
    feature_list: ['u_expose_count_7d_all']
```

处理顺序：原始读取 → 按 feature_default_values 填充 → 按声明顺序执行转换。
默认值的键必须属于 features 原始字段，派生列/标签列不能声明默认值。
未声明的 NULL 保留，不擅自填0。标签资格不因默认值改变；不新增缺失指示输入。
历史默认空列表在 YAML 中显式声明，配对校验仍检查原始列表结构。
非负计数本轮默认0，log1p(0)=0；若未来改为-1，必须显式调整表达式，不能直接 log1p(-1)。

模板按 feature_list 顺序展开，变换按声明顺序执行。允许显式原地转换（例如 [*] = log1p([*])），
读取上一阶段同名值；保存不可变原始列供审计与任务标签使用，禁止变换覆盖标签语义。
本轮示例使用派生名避免覆盖；*_log 派生列在 input_features/tree_inputs 中显式引用。
有依赖时前置定义必须存在，禁止前向引用，不再按任意拓扑重排导致原地转换语义改变。

文本值（字段名、表达式、枚举）统一用单引号；数字/布尔保留 YAML 原类型。
loss_weight 属于表达式，因此写 '1.0'/'0.2'；num_outputs、scale、sigma 等仍为数值。
保持扁平特征空间，不新增 features[item]、apply_to 或 from。Finder 的域选择在本数据结构中不需要。
层 norms 无归一化写 ['']（每层一个空字符串）；不使用 dropout 则省略对应字段。

神经时长任务采用 Finder 形式 soft_label: {func: 'normal', scale: 5, sigma: 2}，
num_outputs=61；label 用 clip(watch_seconds, 0, 300)，metric_label 用原始 watch_seconds。
软标签以 label/scale 为桶坐标，sigma/scale 为桶标准差；expectation 为
sum(prob[k] * k) * scale，明确输出秒而非桶索引。使用稳定的 log-probability 归一化。
Source: Finder comm/nn_utils.py::get_soft_label、pred_expectation（307行）。

最新决定：不自动扩展缺失指示输入。feature_default_values 中填0、-1或其他合法值由用户配置，
模型直接使用该值；审计切片与内部校验不自动成为模型特征。


## 独立搜参配置合同

search.yaml 是 Doku 新增的编排设计，不是 Finder 原有字段；不属于单模型运行配置。
base_configs 引用神经模型，baseline_config引用只执行一次的树基线。
search.method=grid；parameters 为实际配置路径→候选列表，fixed 为神经trial共同覆盖项。
每条路径必须存在于基准配置且新值符合该插件schema，不能静默创建拼错字段。
展开顺序采用声明顺序，生成稳定组合编号、独立run_id和完整resolved config。

comparison.split=validation；tasks列出对齐的指标及方向。分类严格双改善判胜，时长MSE降低
判胜；不提供训练门禁、隐式阈值或跨任务自动总分。三份模型示例均增加watch_seconds.metrics.mse。
搜索结束先出验证报告，再决定一个模型是否精搜；测试日期继续使用各模型split，
不恢复evaluation/test_usage/slices。完整比较和报告规则见design与R438～R441。


## 实施授权与模型目录（最新确认）

DNN 与 DeepFM 都使用 model_type: 'rerank'，不创建 models/deepfm。
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

## 2026-10-10 已确认 numeric 分桶 embedding（配置先行）

此决策替代神经模型此前将计数 `_log` 和三个剧集字段直接作为连续输入的方案：

- `i_episode_dur_s` 上界为 [63, 71, 77, 84, 89, 93, 102]，共 8 桶。
- `i_episode_ord` 上界为 [1, 2, 4, 6]，共 5 桶。
- `i_ep_cnt` 上界为 [10, 15, 24, 40]，共 5 桶；≤9 与 10 合并。
- 25 个计数统计字段按 `clip(round(log2(maximum(x+1,1))),0,31)` 转 int64，
  命名 `_log_bucket`；`u_active_days_7d` 使用 0～7 整数桶。
- 上界均包含等号，最高桶向上开放；原始默认 0 进入最低桶，不新增 missing 桶。
- 共 29 个派生字段各有独立 `embed_name`，每表 4 维；原有 ID/历史表保留 8 维。
  `i_quality_score`、`i_heat_score` 仍直接作为连续值。FM 交互对保持原配置。
- 表达式保持扁平结构，使用已有 `where/clip/maximum/cast` 名称，不新增 apply_to。
- 两份神经配置及 examples 同步；XGBoost 基线不改，历史运行归档不改。

当前状态：后续已授权实现派生 embedding 类型支持，并各训练一次 DNN/DeepFM。
类型从显式 cast（或已知类型字段别名）解析并保存到 Feature State，不向源 Parquet 请求派生列。
词表仅拟合训练集，未见桶沿用 OOV；不新增默认值或隐式桶词表。每轮保存各 task 验证指标。
本次不执行搜参、不读取测试数据；实际运行证据见 verification.md。
Source: 用户确认 numeric 分桶方案、4 维 embedding，并要求先改 YAML，不改代码。

## 2026-10-11 完播单目标消融

用户要求 DNN/DeepFM 仅完播 loss_weight=1，其余六个 task 全部为 0。
当前两份神经 YAML 及 examples 已同步，selection 仅使用 completion loss，
防止未训练目标影响 checkpoint 选择。search 固定项中的 watch_seconds 权重同步为 0，
本轮不执行 search。其他特征、分桶 embedding、网络、优化器及日期保持不变。
保留其他输出头与指标用于运行完整性记录，但这些头没有对应监督损失，
不能作为有效的多目标效果与旧模型比较。仅本机各训练一次，测试集不读取。
Source: 用户本轮明确要求仅使用完播 loss，重新训练 DNN 和 DeepFM。

## 2026-10-11 DeepFM 完整 20 轮诊断

用户要求仅 DeepFM 将 early_stopping_patience 从 3 改为 20，max_epochs 保持 20，
仅完播权重和其他配置保持不变；本机单次运行并保存完整每轮训练 loss、验证 loss/AUC。
DNN 配置不变；不搜参、不读取测试集。与上次仅完播 DeepFM 和 XGBoost 比较。

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

## 2026-10-13 特征范围更新

Source: 用户确认当前配置加入 14/28 天统计及完播、进入沉浸观看的 drama 序列。
此决定取代此前首轮排除这两条历史及仅选 7 天统计的实验范围；不修改已完成实验的冻结配置。
三模型本地 YAML/examples 各从 38 个原始特征扩展到 84 个：新增 44 个统计字段及
`u_finished__drama_id`、`u_immersive_enter__drama_id`。
统计范围为用户七类行为的 all/fyp 14/28 天计数、item 七类行为 fyp 14/28 天计数，
以及用户 14/28 天活跃天数。仅选合同存在字段。

编码沿用各模型现有约定：神经计数 log2 分桶且独立 4 维 embedding；活跃天数按窗口
截断为整数桶；树计数 log1p、活跃天数原值。原始统计默认 0，原始历史默认 []，
派生特征不声明默认值。新增历史 tail 100、保留重复，神经 mean pooling 并共享 drama 表；
DeepFM 增加候选与每条新历史的交互，树增加每条历史的长度及候选命中标量。

配置已加入不等于已验证收益。迟上线字段的空值不能解释成真实无行为；下一轮训练前核查
逐日覆盖率。此次仅改配置与文档，不改 Python 实现、不训练，测试集保持未读取。
三份示例保留原 v4 Manifest、日期和任务权重；v5 已运行目录仅用于历史复现，不覆盖。

### 冻结日更输入替换（2026-10-13）

Source: 用户指定 daily-migration-20261013/frozen-input/manifest.json 更新 dataset。
当前三份本地 YAML/examples 改用上述不可变 Manifest，替代上一节保留 v4 的说明；
实际版本 frozen-33b251cf811004b523d5db41de7f366a4ec79f8326f6e6dce78635c9f18db80d，
feature schema v5，label definition fixed-5m-playback-stream-v4。Manifest 逻辑 SHA256
为 320ede137bede5f3642326339bc8b311621f4eac23cadd51da329797c4533908，已重算核对。
允许读取冻结目录中的合同和 daily/environment=prod/ 下精确分片；不以可变日期模板训练。
数据覆盖 10-01～10-11（1,167,111 行），本次不修改 split、任务、特征或已完成 Run。
验证：三份配置 CLI validate 成功，未读取样本、未训练。

## 2026-10-13 Dataset 简化（当前规则）

Source: 用户确认单 Manifest 地址方案。dataset 只声明 manifest_uri；旧版本/摘要、adapter、
aws_profile、allowed_prefixes 和训练内 daily_prefix 说明被本规则取代。
版本与摘要从 Manifest 读取并校验，AWS 身份由启动环境提供，无生产默认值。
每次 Run 冻结 Manifest/合同原文及摘要，resume/test 使用原冻结记录；不重新解析日更指针。
