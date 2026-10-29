# Spec 024：Finder 当前路径迁移设计

状态：用户已批准实施。当前实现与验证证据见 tasks.md、verification.md；本文描述目标及已批准边界。
Source: requirements.md、migration-scope.md；用户最终确认采用 Finder loss_weight 与 weight decay 链路，撤回 PyTorch 耦合 L2 兼容方案。

## 1. 基准与目标

只迁移当前 e1 实验所需功能及完整依赖，忽略旧 TF 实现。基础模型来源为 finder-models 的 models/rerank；连续值使用 models/rerank_prom 的 scalar_features。模型 revision 为 0abdcc692a84319833d871e30c0bfd7ee6a22080，框架 revision 为 cbd00cdc32f9e843204eda52d52480d71fde0105。

保留函数边界、变量作用域、数学公式与配置执行顺序；只适配本地 CPU、Doku 数据和明确批准的扩展。CPU Python3.10/TF2.11，独立依赖。本文是目标状态，不代表功能已经存在。

## 2. 目录与职责

```text
recommend_tf/
  pyproject.toml                 # 独立依赖及锁文件，具体锁工具沿用仓库约定
  train/
    run.py                      # 参数解析与装配
    train_rerank.py              # 用例入口
    comm/
      datasvr/                  # Manifest/Parquet、基础输入分片、batch读取
      common_utils.py           # 所需图表达式
      vocab_manager.py          # 本地固定映射与可训练embedding后端
      graph_helper.py           # Finder优化器装配、单机梯度更新
      weight_decay_optimizers.py # Finder所需衰减实现
      nn_utils.py               # Finder loss/prediction
      training_pipeline.py
      checkpoint_agent.py
      eval/
    models/rerank/
      config.schema.yml
      model_params.py
      data_utils.py
      rerank_model.py
      modules/utils.py
    tests/
```

comm 不依赖具体模型；模型不读取存储、不启动训练任务。私有 native、WePS/PSStor、Horovod、MPI、GPU、平台管理代码不迁入。源码来源与必要修改逐函数登记，不把独立实现称为源码复用。

## 3. 数据生命周期

1. 冻结 Manifest、合同、配置、日期与摘要；train/validation/test 按声明日期隔离。
2. 分块读取数据；按显式默认值填充，执行 item_filter；历史按已确认的重复合并、排序、截断规则整理。长度在整理后计算。
3. 写 run-scoped 基础输入分片，保留标量/变长列表、历史权重、length、原始标签字段。特征和标签共用一条链路，不另建标签缓存。
4. 仅 train 拟合固定类别映射；使用同一图表达式实现生成派生类别，确定性排序编号；不在分片保存最终表达式值和类别索引。首版统一扫描拟合，不另建已知桶范围捷径。
5. 每个训练epoch以seed与epoch确定shuffle，随后组batch、动态padding；验证和测试确定顺序。图内执行特征与标签表达式及模型运算。

基础分片建议使用已存在 Arrow/Parquet 依赖；分片清单记录行数、schema、内容摘要、输入与配置摘要、准备代码版本。临时写入完成后发布完成标记，未完成分片不能复用。限定内存/磁盘预算，失败清理本run临时文件，不修改其他run。仅相同冻结输入及准备语义的resume允许复用；不建跨run共享缓存系统。

shuffle设计采用显式配置容量的有界样本buffer，顺序可复现但不声称与Torch全量permutation相同；不打乱样本内部历史。每个epoch覆盖每个准备样本一次，保留尾batch。恢复以epoch边界为首版验收点，不承诺任意batch精确续训。

padding使用batch内最大真实长度，空历史batch分配至少一列零占位，length保持0。ID与权重同步padding；mask由length生成，不能由原始ID为0推断。历史point-in-time正确性是输入合同要求；本轮尚无真实数据审计完成的声明。

## 4. 模型图

```text
raw_features
  → transform_features
  → scalar_features + embedded_features
  → 合并inputs
  → attention_layer / 配置驱动weighted_mean
  → rerank_core_model
  → target_layer → predictions
原始或转换后的字段 → label表达式 → get_loss(loss_weight)
```

model_fn构图；每batch执行图，不重新构图。TRAIN/EVAL共享特征与预测语义；PREDICT不要求标签，基础PREDICT图验收不等于交付线上exporter。

item_numeric拆成scalar组（quality、heat）和embed组（3个桶），在dnn_inputs原位置依次拼接，保持宽度与顺序。scalar_features来自rerank_prom:586，使用input_type分流，不重写连续值网络。

保留共享128/64 DNN、5对FM、7个独立32维任务塔和当前激活/dropout；未用的LHUC、FiLM、额外tower、HSTU等显式不支持。图表达式补充所需的Doku算子映射，保留Finder AST图运算而非Python逐样本解释。

兴趣weighted_mean是明确扩展：配置声明历史权重字段；图内按有效位置求sum(w*embedding)/sum(w)，空历史或零总权重返回零向量。三路attention保留Finder dot/softmax主体；全空历史显式零输出以维持基础输入合同，不把此边界处理隐称原样源码。mean路径按实际元素数归约。

## 5. 固定lookup后端

仅train收集类别，按共享embed_name合并，按固定类型规范确定性排序。索引0为padding、1为OOV、2起为类别；原始类别0正常映射，不等于padding。validation/test不扩表。

图内使用固定lookup资源把转换后的类别映射为索引，再lookup可训练embedding；维度和初始化遵循已选配置。padding采用常量零行加可训练其余行，不参与梯度/衰减。词表、类型规范、映射摘要随feature state保存，恢复拒绝不匹配状态。

替代vocab_manager私有后端，不改变embedded_features上层分组接口。不能声称固定词表等价于私有动态表容量、淘汰和更新策略。

## 6. 标签、loss与指标

用户已指定：默认值按配置填充；label只定义目标，loss_weight控制样本贡献；不新增自动nullable mask或按有效标签数归约。

迁移实验配置显式增加watch_duration_ms默认0；移除旧Doku时长任务的nullable mask，不暗中将其转成隐藏权重。其他业务筛选须写入loss_weight。原始字段若保留null，其用途由配置与类型合同决定，不能自动排除训练样本。

保留Finder get_loss归约：loss_by_weight=false为mean(weight*loss)；true为sum(weight*loss)/(count_nonzero(weight)+1e-10)。不改成权重和分母；不把标量weight擅自扩成向量来改变true分支语义。目标表达式必须在参与图运算的输入上有定义，weight=0不能修复NaN*0。

保留logloss、normal soft label、softmax_soft、sigmoid/expectation；任务权重按配置。评估过滤与训练loss_weight是不同配置用途，不默认用权重替代全部指标过滤。指标空集记录null及原因；评估、选模与产物协议沿用Doku显式配置，不迁私有evaluator服务。

## 7. Finder优化器与weight decay

撤回统一全参数耦合L2适配。保留Finder optimizer/optimizer_args和独立sparse_optimizer/sparse_optimizer_args装配关系；迁移本实验所需Adam及Finder AdamW实现，不为了兼容Torch改公式。

源码证据：
- graph_helper.py:45–52区分adam与adamw；:1346–1390创建dense/sparse优化器，并存在私有DynamicEmbeddingOptimizer包装。
- optimize_loss:704先对传入loss求梯度，按变量注册分组；:878附近apply_gradients，最后更新global_step。
- weight_decay_optimizers.py:186–224：dense以var -= wd*var衰减；sparse仅对indices命中的行衰减，随后执行基础优化器更新。wd不是自动乘学习率后的系数，不能套用Torch的同名参数。
- graph_helper.py:772/787的dense/sparse_use_weight_decay分支计算正则梯度和summary；该局部结果没有在此函数后续直接加到更新梯度。因此不能仅打开开关就声称衰减生效。
- rerank_model.py:523、548通过层regs指定kernel regularizer；这与optimizer的解耦衰减是两条路径，不能自动相加或混称。

本地适配保留变量分组与优化器参数传递，移除通信和私有表包装，改用本地Variable/IndexedSlices。保留稀疏索引去重语义，不把稀疏梯度无条件变成整表dense梯度。私有动态表包装与本地TF optimizer的slot/未命中行更新差异必须以测试和证据标注，不承诺私有服务逐参数等价。

原实验的adam + weight_decay=0.0001不能自动等价映射。设计接口保留两种明确选择：adam不隐式增加衰减；需要Finder解耦衰减时显式选adamw及其参数。具体实验采用哪种配置、wd数值和dense/sparse作用范围在运行配置审阅时确认，不阻塞迁移这条能力，但阻止声称原实验正则已等价复现。禁止自动启用AdamW默认wd=0.004。

## 8. checkpoint与运行边界

本地checkpoint保存模型变量、optimizer slots、global_step、完成epoch、seed及配置/词表/基础输入摘要；只恢复同一逻辑图合同。产物run-scoped、create-only，失败不得宣称完整成功。模型checkpoint与推理artifact分离。本轮不提供线上部署或旧TF checkpoint兼容。

准备、词表拟合、batch输入、forward/loss/update、验证分别计时，记录行数、吞吐、RSS、线程和软件版本。所有性能结论基于同一冻结切片与环境，不复用旧P1/P2数字，不通过排除准备耗时宣称提速。

## 9. 验收与剩余配置审阅

- 对照源函数固定输入/权重验证转换、scalar、embedding、attention、core、logits、loss和梯度；必要适配独立断言。
- loss覆盖0/非0权重、两种loss_by_weight分母、标签默认0，不存在自动隐藏mask。
- Adam/AdamW验证dense与稀疏一步及多步：命中、未命中、重复索引、padding、slot恢复；验证wd不隐式乘lr，正则无重复添加。
- 基础分片覆盖一致性、EOF/尾batch、padding与length、shared vocab/OOV、故障中断不可复用及epoch恢复。
- 最小CPU七任务训练/评估/恢复；文档/索引、lint和适用测试通过后才宣称实现完成。
- 设计中有界shuffle、epoch边界恢复、分片格式是供本次整体审阅的工程方案，不声称此前逐项批准；实验optimizer参数仍需在启动实验前明确。

设计确认后才生成tasks并实施；本次不启动云训练、读生产样本或发布模型。


## 实施补充

Source: 本轮迁移代码验证。

- 当前图返回本地字典协议；保留源函数边界，不保留Estimator私有平台运行器。
- 为epoch边界恢复重建同一图，以seed+epoch固定该轮随机流，恢复上一checkpoint的参数与optimizer slots；Finder层内dropout公式不改为自定义无状态实现。
- 指标AUC使用本地SQLite排序聚合，避免把全量预测留在Python列表；指标过滤不隐式绑定loss_weight。
- Doku reader的hashlib.file_digest改为流式SHA256以兼容Python3.10；不导入Torch特征准备。
- 当前计时区分准备、词表、图构建/恢复、训练含输入、验证含输入。更细的input/forward/update拆分及真实数据性能对照仍属后续验收，不据此发布提速结论。


## 首次正式训练入口与持久化（2026-10-29 用户确认）

每 epoch 保存训练/验证 task loss、配置声明的验证指标及阶段耗时到 training/epoch-metrics/NNNN.json；TensorBoard 以 epoch 为 step 写入 training/tensorboard，单轮事件文件 flush/close 后发布。不可计算指标保留 JSON null/reason，TensorBoard 不伪造为0。训练集暂只统计 loss，不额外重扫训练集计算 AUC。

experiment.execute 统一 train → 验证集选模 → 所选 checkpoint 独立 test。选择不可用时保留训练产物并记录 trained_without_selection / test not_run，不使用 test 选模或隐式降级。

每轮 checkpoint、指标完成后同步上传 S3 run-scoped 路径；上传 input 冻结合同、training、test 和根 JSON，排除 prepared 原始输入、缓存、临时 SQLite。事件文件上传前关闭；已发送未变化文件跳过。上传失败使训练失败，并尽力上传失败状态，不能把未持久化轮次标为已完成的云端运行。状态文件最后上传。最终 metrics.json/lineage.json 仍只在完整训练成功后写出。

SageMaker cloud_train 校验配置、源码及锁摘要；Dockerfile.train 提供正式入口及 DOKU_MODE=verify 的同镜像验收路径。首次40 epoch配置沿用已review的 extended40-lr004：lr0.004、batch2048、Finder AdamW0.0001、seed42；单机CPU，复用现有AWS设施。用户已授权验证通过后启动一次正式任务。


## 本地输入列投影（2026-10-29 用户确认）

在 published reader 完成事件时间切分后，基础分片仅保存显式 dtypes 输入列及历史length；不额外持久化reader附带的impression_time审计列。已有基础分片的读取增加列投影，统一供词表拟合、train、validation及独立test使用。列集来自声明的输入dtype与历史组，而非全局硬编码删除某个字段；未来显式声明需要的字段仍保留。

分片完整性摘要、行数验证、shuffle顺序、padding和模型图语义不变。该修改仅位于本地worktree，未重建或推送镜像，正在运行的首次40 epoch任务保持原实现。


## 列式输入与RSS检查（2026-10-29 用户确认，本地实现）

Prepared读取将Arrow列转成NumPy数组，通过只读Mapping行引用参与现有shuffle；数值变长历史保持NumPy数组视图，不逐元素构造Python列表。字段包含null或不支持的类型时回退该列的to_pylist以保留None/类型语义。行引用持有列缓冲区，保证跨块shuffle生命周期。

不改变random.Random调用次数、buffer append/pop时机、batch边界和尾部排空顺序。RSS同步检查改为每1024输入样本以及每个输出batch边界，后台50ms峰值采样保持不变；这改变资源超限检测时延，不改变训练数学语义。来源分片摘要和行数检查保留。

本次只修改独立worktree并本地验证，现有SageMaker任务和镜像不变。

## 一次性准备与词表拟合列式优化（2026-10-29，本地实现）

Source: 用户要求对同一切片拆分准备/词表拟合耗时后优化，保持语义且不影响在跑任务。

PublishedDataset 对带时区的 Arrow timestamp 直接按原始时间单位批量验证分区上下界及 null；未请求的审计时间列不转成 Python datetime。显式请求时间列仍解码返回。字符串、数字、无时区时间沿用原校验路径。Manifest/合同、完整分片 SHA256 校验不省略。

train_rerank 传递列块，普通字段按列填默认值；仅历史相关字段构造小字典并复用 organize 原算法。仍在过滤前验证与整理历史，避免复制 PyTorch filter-first 的不同错误处理语义。筛选后按列构造 Arrow Table、保存基础输入。没有缓存表达式结果或最终类别索引。

词表拟合使用不打乱的 column_batches，跨分片拼满原 batch_size，尾批保留。按列构造与旧 tensor_batch 一致的标量/历史 padding 张量，之后复用原 transform_features、历史 length mask、集合收集及排序。空历史分片可为 list<null>；拼接按列处理，禁止通过统一 Arrow 数值类型提升丢失整数精度。读取仍校验每片 SHA256、行数和总行数。训练/验证/test 的 shuffle 算法与 tensor_batch 不改。

本轮没有合并准备与拟合扫描，没有改表达式执行位置，没有提前编码最终特征，也没有改每块过滤图的构建方式。实测先处理原始解码和张量组装热点；跨阶段扫描合并需单独证明收益及一致性。

## TF单因素搜参扩展（2026-10-29）

后续 Adam 的 Dense 与命中 Embedding 正则提案见 [正则迁移设计](regularization-design.md)，用户已确认并完成本地实施与合成验收，未进入当前七组实验。

六组用户指定实验及Finder标准LN接入设计见 [搜参设计](hyperparameter-experiments.md)。用户已确认纯Adam（同时取消衰减的例外）并增加同资源baseline，共七个任务。公共LN路径的实现范围由用户本轮明确授权。

## 可复现性修复（2026-10-29，用户授权实施）

Source: [一致性排查](consistency-investigation.md) 与用户“修复这两处”。

training_pipeline.seed_graph 在每个训练 epoch 的新图内调用 random.seed、np.random.seed、tf.set_random_seed，使用既有 seed+epoch；checkpoint 评估构图也显式固定 seed，随后恢复参数。恢复训练仍重建该 epoch 的随机状态，与不中断训练一致。不改变批次 shuffle 自己的 random.Random(seed) 协议。

graph_helper.optimize_loss 保留原梯度计算和 IndexedSlices 去重/求和，在所有优化器实际使用的 dense gradient、sparse values/indices 上构建 all_gradients_ready 屏障。所有 apply_gradients（包括创建的 Finder AdamW decay assignments）都在此屏障的控制依赖下构造，global step 在全部更新之后递增。屏障跨 dense/sparse 优化器，不能仅等待各变量自己的梯度而允许其他参数提前修改。

Finder weight_decay_optimizers.py 保持原样；更新次序为同一旧参数状态计算全部梯度，然后执行原衰减及Adam更新。对旧run的源摘要校验不绕过；此次不迁移旧checkpoint以混用新运行时。

回归包含图可达性检查（所有decay/Adam写操作依赖屏障）、可手算的 dense+sparse 梯度及衰减、独立子进程的完整初值/两轮训练/恢复checkpoint逐字节一致；改变seed必须改变初始化。诊断依据及范围见一致性排查文档。
