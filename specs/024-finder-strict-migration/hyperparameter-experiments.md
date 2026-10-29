# TF 六组单因素实验（2026-10-29）

Source: 用户要求学习率0.001/0.002、Adam、weight_decay=0/0.00003、Finder LN，共六组，相对当前baseline单变量；最新耗时优化代码，16 vCPU、4计算线程。

## 已确定范围

共同配置：40 epoch、batch2048、seed42、保留率0.9、ReLU、原七任务及loss_weight、同一冻结Manifest/日期、shuffle8192、原选模定义与test入口均保留。新建独立任务与输出，不影响在跑任务。16 vCPU使用与既有m5系列一致的ml.m5.4xlarge，intra_op_threads=4，inter_op_threads=1。按性能优化版准备、词表拟合和输入读取路径运行。

| 组 | 学习率 | 优化器 | 衰减/L2系数 | norm | 相对baseline变化 |
| --- | --- | --- | --- | --- | --- |
| 对照 | 0.004 | Finder AdamW | 0.0001 | 无 | 无 |
| LR001 | 0.001 | Finder AdamW | 0.0001 | 无 | 学习率 |
| LR002 | 0.002 | Finder AdamW | 0.0001 | 无 | 学习率 |
| ADAM | 0.004 | 纯Adam | 无 | 无 | 优化器并同时取消衰减，用户确认例外 |
| WD0 | 0.004 | Finder AdamW | 0 | 无 | weight_decay |
| WD3E5 | 0.004 | Finder AdamW | 0.00003 | 无 | weight_decay |
| LN | 0.004 | Finder AdamW | 0.0001 | ln | norm |

## 已确认的两点（用户追加答复）

1. 用户选择纯Adam，不传weight_decay、不新增L2。ADAM组同时改变优化器并取消衰减，明确标注非严格单变量；与WD0的数学路径可能基本重合，仍按用户要求保留两组。
2. 用户批准增加16 vCPU、4线程同资源baseline，六个变体加一个对照，总共七个任务。

用户已确认纯Adam，并增加同资源baseline，总共七个任务。Adam组明确标注非严格单变量例外，其他五个变体相对新baseline只改指定参数。原始加权task loss、selection_score定义不在本轮修改，避免引入额外变量；报告以相同TF口径逐epoch比较各任务AUC/logloss/MSE及总选模指标，不以test挑选超参。

## Finder LN设计

来源：finder-train/recommend/train/comm/layernorm.py的LayerNorm，以及finder-models/recommend/train/models/rerank_reddot/rerank_model.py的_get_norm("ln")。不改Finder源仓。

使用axis=-1、epsilon=1e-6、center=True、scale=True、remove_batch_mean=True。gamma初值1、beta初值0，按每样本最后一维求mean/variance；没有运行均值，TRAIN/EVAL公式相同。调用tf.nn.batch_normalization只是使用其算术实现，不是BatchNorm统计。

LN应用于共享隐藏层[128,64]和全部任务塔隐藏层[32]，顺序Dense→LN→ReLU→Dropout，最终logits层不加LN。只迁移此已授权路径，不引入tensorflow_recommenders_addons动态Embedding依赖、BN、RMSNorm或其他norm变体。

验收：关闭norm时原路径不变；LN前向/输入与gamma、beta梯度对照Finder源类；TRAIN/EVAL共享变量、checkpoint恢复；非法norm与长度拒绝；合成完整训练。所有组使用同一新镜像，原生Linux验收通过后才提交。

执行状态：同镜像Linux25项验收及七配置差异检查通过，七个独立任务已提交；不使用Adam+L2。具体任务及镜像见formal-training.md。

## Adam L2 五组扩展（2026-10-29 用户授权）

以 `doku-exp-f024-adam-20261029-072135` 的冻结配置为基准。保持 Adam、lr=0.004、无 norm、batch=2048、seed=42、dropout、日期切分、shuffle、任务及选模配置。每组40 epoch，ml.m5.4xlarge（16 vCPU），intra-op=4、inter-op=1。先通过新镜像原生 Linux amd64 合成验收和五组配置差异验证，再提交独立任务；不修改已有任务。

| 组 | Dense L2 | Embedding L2 | 对照 |
| --- | --- | --- | --- |
| D1 | 0.000015 | 0 | 单独 Dense |
| D3 | 0.00015 | 0 | 单独 Dense |
| E1 | 0 | 0.000001 | 单独 Embedding |
| E3 | 0 | 0.00001 | 单独 Embedding |
| D2+E2 | 0.00005 | 0.000003 | 组合，不称单变量 |

Dense 系数应用于共享及所有任务的非空隐藏层 block，排除最终 logits、bias 和 norm。Embedding 表选择为 []（全部输入表），按有效出现位置计数。实验文件仅保留 `runs/finder-adam-l2-five-20261029/`，不随源码发布。新五组使用同一新镜像；旧纯 Adam 基线来自之前镜像，关闭新增正则一致性已有本地验收，保留代码版本差异记录。
