# Spec 035：TF 训练计时与对照实验设计

状态：实现及两epoch/40epoch实验已完成；2026-11-06用户授权提交PR并合并main。
Source: 本会话只读云调查、recommend_tf/train/comm/training_pipeline.py、Spec 024 verification.md。

## 当前实现与启用方式

推荐在TF实验配置顶层显式声明：

```json
"input_pipeline": {
  "batch_encoding": "columns",
  "prefetch_batches": 2,
  "encoded_cache_bytes": 0
}
```

不声明时保留rows编码、缓存0、预取0。批量编码保持dtype、历史padding、样本顺序及batch边界；FIFO后台线程仅准备输入，TF计算仍在主线程。实验缓存能力保留为独立可选项，但当前推荐组合不启用缓存。可选计时和本地基准入口不改变配置逻辑身份；input_pipeline本身进入冻结配置。

两epoch组合和b4096完整40epoch已完成；40轮原始非计时指标逐项一致。单轮训练缩短51.6%、包含准备和逐轮上传的训练流程缩短47.0%，尚不声称完整流程达到50%或跨机器SLA。具体证据见verification.md。源码合并不自动改变运行配置、发布镜像或切换线上模型。

## 历史基线调查（2026-11-05初始快照）

候选主基准为 doku-exp-latest31-1105-tf-lr002；历史 doku-exp-ln001-1103-tf-lr002 仅供参考，两者配置和输入未证明相同。原始实验产物仅本地 /tmp/timing-*，不随源码发布。

| 阶段 | 历史40轮合计（秒） | 当前已采25轮的每轮中位（秒） |
| --- | --- | --- |
| 训练含输入 | 20852.265 | 540.898 |
| validation含输入 | 846.344 | 21.476 |
| 图构建/恢复 | 422.377 | 10.443 |

历史metrics.seconds=22363.950，包含词表163.421秒；减去词表及三阶段剩79.544秒，包括保存/回调等，不是纯checkpoint耗时。SageMaker总训练23501秒，与metrics差1137.050秒，尚不能准确拆分。当前仍在训练，未取得完整全流程基准。

两者单台ml.m5.4xlarge、100GB盘。当前batch2048、shuffle8192、intra/inter=4/1、reader_workers=0、prefetch_batches=1、s3_transfer_threads=1、arrow_threads=1，内存上限16GiB、缓存上限20GiB。历史进程峰值RSS约1.136GiB，不代表整机占用。CloudWatch未返回有效CPU/内存数据，不能推断CPU空闲。

训练含输入占上述三阶段约94%。当前tensor_batch/feed与sess.run同步执行；reader_workers只并行分片下载/校验等准备，受transfer与prefetch限制，不并行每轮编码。历史小切片4→16计算线程仅改善2.29%，不能外推本基准。

## 目标计时

入口到成功产物完成使用monotonic总计时。分别记录下载/校验、特征准备、拟合、图恢复、训练、验证、test、checkpoint与最终落盘；记录未归属残差及失败阶段。

训练内细分读取/shuffle、编码/feed、sess.run和指标聚合，每epoch聚合，不逐步打印。低频采集CPU、RSS、I/O及队列等待；默认关闭，先验证观测开销，目标低于2%。重叠工作报告忙碌时间、等待时间与壁钟关键路径，不直接累加。

## 实验矩阵

| 顺序 | 变量 | 控制方式 |
| --- | --- | --- |
| A | 原配置，计时关闭/开启 | 核实观测开销 |
| B | intra=2/4/8/16，inter=1 | 同机器交错执行，切片至少3次 |
| C | reader=0/2/4/8 | 单独验证transfer、prefetch限制后组合；冷缓存 |
| D | Arrow threads=1/2/4/8 | 与计算线程分开测试，核实参数实际生效 |
| E | 批量列式编码、有界预取与计算重叠 | 单项验证后组合，保持顺序及batch边界 |
| F | 同架构16/32/64 vCPU候选 | 精确规格另核实；先固定软件线程，再调线程 |
| G | 最佳组合 vs 原始基准 | 冷缓存全量至少3组配对重复 |

B–F先用固定代表性切片筛选，包含首轮和稳态。收益停滞或资源竞争时停止扩展，不预设核数越多越快。基础设施变更仍归iac。

[INFERRED] 优先减少编码成本，评估跨epoch复用与顺序无关的确定性编码；缓存身份绑定输入、配置和特征状态，遵守资源上限，不缓存随机训练运算。预取必须按原顺序交付、传播异常且能关闭。

[INFERRED] 按历史粗计时训练占约89%，其他部分不变时需要训练段约2.3倍加速才能总耗时减半。仅输入/计算重叠未必足够，不能提前承诺50%。

## 一致性和进入实施条件

顺序摘要、batch边界、标签/mask和离散特征完全一致；浮点特征、梯度、loss及指标容差用基准重复校准，并在正式优化实验前确认。检查选模和早停变化，不能将少跑epoch当加速。

设计已确认，实施任务见tasks.md。真实基准实验前冻结最新基准完整配置、Manifest、镜像及源码身份，核对当前工作区差异。后续用户已授权本Spec两epoch筛选实验；独立任务设3小时上限，不引入GPU、多机或训练语义调整。

## 本地实施接口（Q11）

复用comm/timing.py，训练函数仅插入可关闭的span/decorator。观测开关通过benchmark入口的--profile提供，不写入训练config；同一源码下开关不会改变checkpoint逻辑身份。训练/验证分开聚合input、tensor_batch、feed和compute；已有stage_seconds继续保留graph_restore。prepare为包含读取的总时间，下载/校验是其子项，不能相加。

入口：`python -m recommend_tf.train.benchmark --configs /absolute/baseline.json /absolute/candidate.json --local-root /absolute/mirror --output /absolute/new-output --repeats 3 --profile`。使用TF Python 3.10环境，从仓库根目录执行；local-root为Manifest同前缀内容的本地镜像根目录。每个配置用完整实验配置，不是resolved-config。每次新进程，轮次交错配置顺序；输出create-only，不删除任何现有缓存。去掉--profile可作观测开销对照。

benchmark.json记录全部成功/失败状态、execution_seconds及process_wall_seconds、中位数和范围。process_wall_seconds包含解释器/TF导入和进程退出；execution_seconds从execute前到计时会话结束，包含产物保存，不含导入；二者明确分列，不相互冒充。测量结果本身的落盘不属于训练产物。timing/中的inclusive父子与worker耗时不能直接求和。

每秒记录进程CPU百分比、RSS、可用时的进程I/O及主机CPU；进程CPU可超过100%，主机值包含其他进程。平台不支持I/O时为null。统计process_cpu_seconds和process_peak_rss_bytes；短任务可能没有采样点。输出仅本地、不带publish回调，不提交云任务。

缓存状态统一标为uncontrolled，不能用此报告直接声称通过冷缓存全量验收；工具不自动推断配置之间只差资源参数，也不自动判定指标一致性。后续已授权真实输入的两epoch筛选；正式50%全量验收不属于本轮完成声明。初始计时阶段未实现队列预取；后续InputPipeline已提供prefetch_wait观测。

## 三项优化实施设计与两epoch筛选

Source: 用户明确授权按批量编码、确定性编码复用、有界预取三方向改代码并做一致性测试及平台筛选。

input_pipeline显式可选配置：batch_encoding=rows（默认）或columns；encoded_cache_bytes=0（默认关闭）；prefetch_batches=0（默认关闭）。缓存需columns编码。配置非法fail closed。选项包含在冻结配置/lineage中，不改变模型计算、样本顺序和batch边界。

1. columns编码：按ColumnRow的列块分组，NumPy批量取行，再恢复原batch顺序，复用tensor_columns；普通Mapping输入保留原路径。
2. 编码缓存：每Run私有有界LRU，键绑定prepared identity、不可变分片摘要、块序号；值为确定性dtype转换及历史padding的数组。每次读仍校验文件摘要。batch取行后按该batch最大历史长度裁剪/补零，保持原张量形状。缓存只复用输入编码，不缓存TF变换、embedding或随机运算；不跨Run共享。超过容量时淘汰或不缓存。
3. 预取：单个有界后台线程，仅执行读取/shuffle/编码，主线程执行sess.run。按FIFO顺序消费；传播生产者异常，显式关闭线程。跨线程计时使用copy_context，worker时间与主线程等待不相加；保留内存预算检查。

资源矩阵：16vCPU基准intra4/inter1；intra8/16；reader2/4/8分别显式匹配transfer/prefetch（作为读取并发组合，而非孤立reader参数）；32/64vCPU保持intra4/inter1以隔离硬件影响。各组新实例冷磁盘与独立输出；OS/网络缓存状态仅记录不保证完全相同。候选项通过本地张量、参数、指标、顺序及失败/恢复测试后各启动两epoch单项实验，再按结果选联合方案。

实际机器矩阵为16/48/96vCPU：SageMaker API拒绝m5.8xlarge/m5.16xlarge（不是该服务支持的训练规格），已替换为支持的m5.12xlarge/m5.24xlarge。没有修改配额或IAM。优化镜像额外运行rows原路径control，与columns/cache/prefetch同镜像比较，排除镜像与公共包装开销；cache组是在columns上增加4GiB编码数组缓存，prefetch组在rows上增加2批FIFO预取。联合优化不预先提交。


## 精确组合验证

Source: 用户要求先验证批量编码＋预取一个组合。配置batch_encoding=columns、prefetch_batches=2、encoded_cache_bytes默认0；其余冻结输入、模型、batch2048、seed、16vCPU、intra4/inter1、reader0与原optcontrol一致。仅补测试，不修改生产实现；复用原优化镜像digest，唯一新增两epoch任务。精确组合已补1/4计算线程下的loss/指标、逐tensor checkpoint、恢复及batch顺序一致性测试；云端组合及后续40epoch结果已回收，见verification.md。
