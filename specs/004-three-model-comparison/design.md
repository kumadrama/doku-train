# Spec 004：声明式配置与三模型实验设计

状态：声明式运行时及三模型功能已实现；2026-10-12 整理源码交付。实验配置与运行产物仅保留本地。

Source: 本对话 Q6～Q27 及 weighted_mean 组件接口确认；本 Spec requirements。

## 当前状态与目标状态

历史实现的 schema 是 Pydantic 导出的快照，配置加载器不读取它，且固定四分类任务与
27/1/28 生命周期。这些路径已移除；当前实现为 schema 驱动的特征、网络和任务声明，
本次 YAML 选择 6/1/1 与不 refit。旧格式和旧行为兼容路径全部移除，
不将 27/1、四分类或时长比完播保留为隐式默认；6/1/1 和新任务也不得重新写死。

外部 YAML 可扁平书写，内部解析后按公共运行时、特征计划、模型计划分层。
公共 schema 管 Manifest、切分、资源及公共编码声明；模型 schema 管网络与任务头。
合成时拒绝字段所有权冲突，模型不能覆盖公共资源与数据约束。解析器校验未知字段、
特征依赖、实体表共享、权重列配对和 FM 维度；表达式使用受限语法与函数注册表，
不得执行任意 Python eval。表达式与字段清单见 [配置合同](config-contract.md)。

## 公共 pooling 组件

以下为已确认接口片段；完整共享表、状态及截断声明见 [配置合同](config-contract.md)。

```yaml
input_features:
  hist_drama_skip:
    features: [u_skipped__drama_id]
    combiner: mean
  hist_drama_interest:
    features: [u_drama_interests__id]
    combiner: weighted_mean
    weight_feature: u_drama_interests__weight
```

组件位于 comm 的特征编码层；DNN/DeepFM 使用同一实现，模型不自行解释原始列表。
drama 词表由训练集候选与两类历史联合拟合。ID 仅查一次共享表；权重列不查 embedding。
weighted_mean 计算 sum(w_i * e_i) / sum(w_i)。先验证并用 float64 合并重复 ID 权重，
再除行最大值及总和，最后转换为模型 dtype；权重归一化不参与学习，embedding 保持梯度。
padding/零权重排除，非法权重报错。全零权重走无有效权重分支，不能除零。
缺失与空历史保留独立状态，不能用默认 ID=0 伪造历史；OOV 查 OOV embedding 并保留权重。
不做额外 L2 归一化，不按列表顺序声称“最近历史”。

已确认：按有效长度屏蔽 padding；缺失、空列表、全零权重均输出零向量，
不再生成额外状态输入，OOV 历史仍参与 pooling。参考 Finder MP2 masked mean，
weighted_mean 为已确认的 Doku 扩展，缺失指示输入已撤销。

已确认：mean 保留重复 ID；快划取交付末尾100个，兴趣合并同 ID 权重后取最高100个，
同权重按原始 ID 升序。三模型基于同样截断结果编码。不自动拼接任何历史状态输入。
字段级细节和待审阅的实例化建议见配置合同。

## 首轮输入分组

item_id 为候选 drama；两组 hist 为快划与带权兴趣；item_tag/user_profile 分别为
item v2 题材/onboarding 题材；context 为平台；item_numeric 为 quality、heat、
当前集时长、集序号及剧集数；item_statistics 为已选内容侧 7 天统计。
删除 u_fyp_confidence，不保留只含该字段的 user_basic 空组。

增加 user_statistics：
`u_{expose,completion,fast_swipe,comment,share,subscribe,immersive}_count_7d_{all,fyp}`
共 14 项，加 `u_active_days_7d`。计数 log1p 并使用 _log 后缀，其他连续值保留原值；原始字段缺失按
feature_default_values 在转换前填充，不生成标记输入。统计时间字段仅用于审计，不直接输入模型。
本轮不默认配置统计存在/缺失分组指标；需要时用任务 metrics.filter 声明。填充值由 YAML 明确选择，允许0。
14/28 天统计本轮不启用，不能因本次计数相等就认定未来永远等价。

## 网络与任务

共同编码 → DNN → 公共表示；DeepFM 另外计算声明的同维向量配对点积，
将每对的标量拼到公共表示，再进入任务塔。不同维度拒绝，不照搬静默截断。
首轮四组配对及共享表关系见 requirements R409/R411。
两模型统一 8 维 embedding、公共层 [128,64]、任务层 [32]、ReLU，
首轮不加 normalization 或 dropout。

分类头输出 logit，用 sigmoid 与 logloss；时长头输出 61 维，桶代表秒数为
0,5,...,300，以 sigma=2 秒高斯软标签训练 softmax_soft，预测为概率加权秒数期望。
应稳定生成软标签，测试覆盖 0、桶中心、两桶之间、300 和越界值；不得发生密度下溢除零。
已确认：仅软标签生成阶段将超过 300 秒的值裁到 300；
评估必须保留原始秒值并报告越界率。时长原始 NULL 对主时长和 play10/30/50 均 mask。

各主任务基准权重 1、辅助 0.2。不能靠减去软标签熵让显示 loss 相近，误称已校准梯度。
先按初始权重训练一次，记录各任务 loss，再讨论调整；不在首次训练前自动校准权重。
验证选模仍只包含四个主任务，使用该次运行固定的权重。调整权重须新建运行，
不能用最终测试集选择权重。XGBoost 时长使用平方误差回归，保留原始秒数。

## 数据与执行边界

主任务人口筛选 video_finish_seen > 0，与版本化外流数据合同共同限定场景。
completion 使用上游进度口径，fast_swipe 使用 1-non_fast_swipe，immersive_click 的
零值按已确认主面板人口作为负例，不使用 positive-only valid。
时长非空等价 exact 的判断绑定 fixed-5m-playback-stream-v4 源码证据，不泛化到未知版本。

版本化 Manifest adapter 必须校验源逻辑摘要、分片与合同；当前 as_of_ms 不是处理完成时间，
成熟性证据仍待落实。成功读取本地副本并不等于完成生产数据验证。
训练、验证、测试分离；仅训练 fit，验证选 checkpoint。首次训练及调权只看训练/验证，
权重和模型选择结束后才评估测试集（用户已确认）；输出 create-only。
训练产物携带配置、编码组件与词表状态、数据身份及未提交源码身份，禁止发布或修改生产。

## 尚未完成的设计工作

完整字段提案、三模型独立示例、表达式、历史状态和树历史标量已整理到
[配置合同](config-contract.md)。该文档末尾六项实例化细节待审阅；数据成熟性仍待证据。
旧格式移除与全仓处置已确认，不再将整个方案标记为未确认。


## 配置权威与代码职责（用户已确认）

schema 定义允许的结构、类型、范围和能力；YAML 声明本次行为；代码负责执行组件、
算子和跨字段语义校验。不得在运行时绕开配置按固定目标名或日期分支。
日期范围、任务数量/名称、标签表达式/来源、有效性、权重、loss、预测变换、指标、
特征组、pooling、embedding维度、FM配对、网络层及训练阶段都使用配置值。
CPU 边界、时间切分隔离、仅训练集fit、有限值检查和create-only等正确性约束继续强制执行。
未知/未实现的能力拒绝，不把“配置驱动”实现成静默忽略字段或任意代码执行。
旧格式直接给出迁移错误，不保留双解析器或旧训练分支。旧测试应按新合同替换，
仍保留对随机种子、泄漏、mask、失败语义与checkpoint正确性的验证。

## 全仓审查整合方案（2026-10-10，供最终审阅）

本节与 R423～R430 覆盖此前遗漏的工程范围；冲突时以本节和用户最新决定为准。
这是目标方案，不表示当前代码已符合。当前代码、YAML、旧测试尚未迁移。

### Finder 参考边界与差异来源

参考 Finder 的运行时 schema 加载、input_features、共享 embedding、transformations、
item_filter、dnn_inputs、fm_layers 和 tasks 声明。Finder 不提供本次 Doku 的 S3 adapter、
CPU XGBoost 插件、weighted_mean 或下述指标质量门禁；不得将 Doku 自加策略称为 Finder 要求。
Finder 的 TF 队列流水线、远程 checkpoint 同步、feature_check 分组检查也不能直接等同于
本仓 epoch 生命周期、训练恢复或 point-in-time 验证。保留 CPU 与跨仓合同边界，重新实现组件。

Source: 本地 finder-train/recommend/train/comm/model_params.py、comm/config.schema.yml、
models 下 rerank/MP2/live/business 的 schema 与交互实现；本仓全仓只读审查。

### 审查处置清单

| 范围 | 移除的旧逻辑/限制 | 目标实现 |
| --- | --- | --- |
| 配置权威 | Pydantic 生成快照作为事实权威、公共配置混入 DNN 参数、基数先必填后覆盖 | 公共与插件 schema 合成并校验 YAML，派生基数只来自 feature state；类型层不另定义规则 |
| 生命周期 | 固定 28 天、27/1、强制 refit、固定 24/2/2 回测和按测试天数步进 | 显式日期与阶段计划；本次 train/validate/final test，不实现无需求的 refit 或回测分支 |
| 模型装配 | CLI 按模型重复业务流程、comm 引用 rerank Batch/TARGETS | 公共类型、模型注册与统一用例，插件负责本模型能力 |
| 任务 | 固定四分类、时长比完播、固定 BCE/头数、目标名分支 | YAML 声明来源、表达式、mask、头、loss、预测及指标；树模型按任务 objective 构造 Booster |
| 特征 | 默认词表 10000、统一字符串化、固定 zscore、至少一个数值字段、全缺失直接拒绝 | schema 声明字段类型、词表策略、变换和缺失策略；支持纯类别输入；共享状态仅训练 fit |
| 历史 | 仅标量处理或模型私有 pooling | comm 提供 mean/weighted_mean，神经模型共享编码；树模型显式声明标量派生 |
| 时间点 | 写死 7 天、按字段后缀或固定 window_start/end 命名检查 | 使用版本化合同的窗口、时间戳和字段映射；无证据不声称已证明 PIT |
| 神经网络 | 固定 concat、单层同宽塔、embedding≤256、深度≤5、宽度≤4096 等人为上限 | 按 input groups、层和任务头构图；保留维度匹配、合法值及实际组件支持检查 |
| 树模型 | 四个固定模型文件、binary/logloss、轮次/深度/max_bin 任意上限 | 任务描述驱动目标与序列化；参数遵循后端有效域；独立配置 early stopping |
| 随机性 | 按任务下标加 seed、refit 加 10000/1 | 稳定命名的随机流由显式种子派生，任务重排不改同任务随机性；不依赖不存在的 refit |
| 损失汇总 | batch 均值再平均、随有效任务集合变化的权重归一化 | 每任务累加 loss sum/valid count 后聚合；权重固定；无有效样本记录并跳过该任务贡献 |
| 指标门禁 | BLOCK/WARN/PASS、不可评估阻断、AUC floor、默认下降 0.02 | 直接删除门禁接口与调用，不新增可选门禁；记录指标事实与不可计算原因 |
| 指标实现 | 二分类专属公共结构、单类导致全部指标失效、所有预测永久驻内存 | 注册分类/回归指标；逐项有效性；可流式统计，精确 AUC 使用受预算约束的落盘/排序方案 |
| 对比 | 仅目标名和切片就认定可比 | 对齐数据切片、标签、mask 和指标语义；不兼容时标记不可比，不自动告警或阻断训练 |
| 输入合同 | 内部旧 Manifest 是唯一格式、猜成熟小时、合同只校验字节 | 版本化 adapter + 合同语义解析 + 源摘要/分片校验；保留可追溯源身份 |
| S3 | 父目录隐式授权、禁止 key 含 latest、合并双斜杠、默认环境身份 | 显式允许范围与身份、保留原始 key；禁止可变 latest 发现机制而非禁字符串 |
| 资源 | workers=0/prefetch=1、AdamW 固定、4MiB/16MiB/2GiB 隐藏限制 | 资源预算和已实现组件显式配置；取消任意上限，保留格式/后端真实限制与预算检查 |
| 读取 | 每轮反复全分片下载、无磁盘预算、全列读取、S3 隐式线程 | 按依赖投影列、受控传输/Arrow 线程、校验缓存和临时目录预算；不承诺未验证的外存能力 |
| Checkpoint | evaluation/production 固定角色、仅回读即宣称续训 | 通用阶段/进度元数据；恢复模型、优化器、RNG 和采样进度；验证一致性，失败不部分修改状态 |
| 产物 | 固定四任务 codec、仅检查压缩大小、写版本不检查兼容 | 任务清单驱动 codec、按预算限制解压大小并验证结构与版本；保持确定性序列化 |
| lineage | cwd/HEAD/unknown、只覆盖少数摘要 | 精确仓库源码与 dirty 内容摘要、依赖、配置、源合同、adapter、feature state、任务语义 |
| 存储正确性 | 部分写出即成功风险、412 误处理、流未关闭、错误 multipart checksum 假设 | 明确 create-only 冲突、完整性校验、写出完成声明与失败状态；修复本地复制校验竞态 |
| 路径/摘要 | 未拒绝产物 . 或 ..、run_id 路径校验不完整、default=str/NaN 摘要 | 明确路径段合同，确定性可序列化值与有限数校验，避免隐式转换 |
| 观测 | 无限 step loss、manifest 行数冒充实际吞吐、进程历史 RSS 冒充本 run | 有界/落盘轨迹，实际处理样本数及测量范围，区分 CPU 数、线程数与运行峰值 |
| 测试/文档 | 固定四目标/27天/refit、强制 YAML 草案、源码字符串禁词、假恢复/容量宣称 | 配置变体驱动合同测试、真实依赖检查；区分 fixture、恢复、实际容量与未完成能力 |

实现组件支持范围必须在 schema 中准确反映，不因“可配置”而承诺尚未实现的优化器、激活或
执行模式。不把 Python/依赖锁版本等工程选择塞入实验 YAML。原始用户标识禁用按合同语义
识别，不能只检查字面 user_id；数值有限性、资源预算和数据完整性不属于被删除的指标门禁。

### 无门禁的评估与选模

每个指标保存 value、计算状态、原因和样本统计。单类 AUC 为 null，但继续计算 logloss。
零有效样本时记录原因；指标偏低不改变 run 成败，不触发预设质量警告。
验证选模按配置的四主任务加权 loss，任务损失按有效样本聚合；不混入辅助任务。
若配置的选模量不可计算，该次候选没有选择分数；其他可计算候选仍可参与选择。
所有候选均不可选择时保留最后完成的 checkpoint，并记录 selection unavailable；不把它
标记为 best，不自动改用其他指标。训练完成状态、选模状态、评估状态分别记录。
本次最终测试依赖模型选择冻结；未选出模型时不悄悄用最后 checkpoint 完成对比。
读取失败、合同损坏或数值计算异常继续按执行错误处理，不能以无门禁掩盖执行失败。

### 数据成熟性：尚待查证，不能用模型门禁替代

已知上游标签窗口为 5 分钟；本地源码显示归档成功标记与导出时 watermark 的关系不足以
单独证明冻结样本的完整性。Manifest 的 as_of_ms 为机械日期边界；S3 LastModified 和
当前 watermark 也不能替代历史导出时的覆盖证据。源 Manifest 没有保存完整 source run 链接。
真实实验前需要绑定源 run 的归档/构建记录与当时 complete_through/covered_from 等证据，
同时确认发布数据对应的部署版本。未发现证据不等于已证明数据不成熟。
此问题不阻止本地实现和 fixture 验证，但目前不能声明真实实验输入已经验收通过。

Source: 本轮 doku-offline / impression-label 源码核查、指定 S3 发布前缀只读核查；
详细证据见 verification.md。不创建虚构 label_maturity_hours 或固定等待天数。

### 实施顺序与验收

1. 定稿配置合同：补齐完整字段/默认值、共享表语法、表达式白名单、状态布局、mean 重复 ID
   语义及 XGBoost 历史标量清单。保留当前用户 YAML 作迁移输入，不把示例字段当真实数据字段。
2. 建立 schema 加载、公共计划和版本化 Manifest adapter；迁移三份独立 YAML，提供离线配置
   校验，删除旧解析及固定语义。用 fixture 验证日期隔离与合同错误。
3. 实现共享特征状态、历史 pooling、任务/表达式与指标注册；验证训练独占 fit、缺失/OOV、
   权重稳定性、mask 和正确损失聚合。
4. 改造 DNN/XGBoost 插件，在 rerank 内增加 FM 分支；统一训练用例，删除门禁、强制 refit 和无配置回测。
   验证显式 FM 点积、共享表梯度、动态任务、树回归和三模型 CPU smoke。
5. 修复资源、checkpoint、lineage 和存储问题；验证真正恢复、失败/部分写出语义、create-only
   和有界读取。同步旧 architecture、README 与测试，标注历史 Spec 已被取代。
6. make check 通过后，解决成熟性证据并验证冻结真实输入；先训练/验证，再冻结选择并评估
   测试集。报告编码差异、已配置指标、loss 尺度与实际资源成本，不上传或发布模型。

整体方案已获用户同意；本轮仅补齐字段合同与 specs 下设计示例，不修改实现或 configs 下实验 YAML。


## 配置形式修订（用户最新确认）

采用用户示例的 [] 列表排版。删除 feature_encoding/vocabularies 配置入口；
缺失填充值统一 feature_default_values，变换统一 feature_transformations。
批量转换沿用 Finder rerank_prom 的 expr 中 [*] + feature_list 模板替换，不另造正则系统。
神经词表共享关系从 embed_name 推导，树词表从类别输入推导；拟合结果仍保存到 feature state。
默认填充不改变标签有效性；不生成缺失或历史状态输入。
详细顺序与完整示例见 [字段合同](config-contract.md)。

## 缺失处理修订（最新用户决定）

撤销自动 *_missing 和历史 state_features 输入，统一 feature_default_values；默认值可由
用户设为0或-1等。内部有效长度、配对校验和标签 mask 不等于新增模型特征，仍保留。


## 最新数值与 YAML 形式确认

计数仅 log1p 并生成 _log，其他连续值保留原值；本轮移除 standardize。
feature_default_values 只允许 features 中原始字段，先填默认值再转换；派生字段无默认值。
文本值加单引号，数值/布尔保留原类型；loss_weight 使用字符串表达式。
norms 无归一化用空字符串，省略 dropout；soft_label 使用 Finder 的 func/scale/sigma，
裁剪放在 label 表达式，metric_label 保留原始秒数，expectation 按 scale 换算回秒。
保持扁平特征结构，不增加 apply_to/from 或 item 域。本条替代先前标准化及派生默认值建议。


## 首轮4+4+1搜参执行设计

独立 search.yaml（本地实验记录，不随源码发布） 引用两份神经基准和一份树基准；路径相对搜参文件。
search.parameters 按字段路径做笛卡尔积，search.fixed 仅覆盖神经模型，得到每模型4组。
XGBoost只运行原基线一次，不应用神经batch_size或学习率覆盖。每次保存展开后的完整配置。

每个trial在同一训练/验证切分上运行，继承seed和最大epoch/早停设置；CPU首轮串行执行。
从基准run_id派生含模型和组合编号的独立trial身份，确保create-only；不会9次复用同一路径。
首次测速直接计入这9组，不另加隐式实验。试验失败记录事实，不用质量门禁淘汰低分试验。

selection仍负责单次运行内部checkpoint选择，原有神经四主任务验证loss系数固定各1。
comparison只负责跨trial/跨模型报告，不能取代selection，不修改训练状态，也不自动开启精搜。
分类task同人口同mask比较logloss/AUC，双严格改善才标win，双严格变差标loss，方向冲突标tradeoff；
完全相等标tie，部分相等标partial，指标缺失标unavailable；不加入隐含最小提升阈值。
时长以原始秒级MSE比较，神经预测由概率期望×scale得到秒数，不裁剪评估标签。
小幅改善只描述该固定seed的观测结果，不宣称统计显著性。

报告包含9组配置、所选epoch/round、每task验证指标与相对基线差值、耗时/内存/吞吐，
以及训练loss（与比较指标分栏）。不对不同task的最佳trial拼装checkpoint；不强制给整体胜者。
报告后决定是否且仅对DNN或DeepFM之一精搜，搜索空间届时明确，不预先自动执行。
最终冻结选择后报告选定DNN、DeepFM和XGBoost的测试结果，保持每task loss/AUC口径。
训练报告、验证比较和最终测试报告分阶段产生，不能把尚未运行的测试结果写成已有结果。


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
