# Spec 035 验证记录

日期：2026-11-05。首轮为本地实施；后续用户另行授权云实验，已完成12个两epoch任务并回收结果，见末节。

## 已执行

- `TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 PYTHONPATH=. recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests/test_training_timing.py recommend_tf/train/tests/test_benchmark.py -q -p no:cacheprovider --disable-warnings`：5 passed，12.78秒。
- 固定seed两轮合成训练：开启/关闭计时的loss和指标完全一致，checkpoint每个tensor完全一致；从关闭计时的第一轮checkpoint开启计时恢复，第二轮指标一致。
- 人工时钟验证iterator只计生产者时间，包含最后StopIteration收尾，不含消费者时间；失败session落failed，关闭不创建输出。
- 完整合成Manifest→准备→训练→选模→test基准CLI成功；拒绝复用输出；失败子进程保留退出码、不计入成功统计。
- 本地合成基准每组3个独立进程：关闭计时中位2.339844秒，范围2.318276–2.398153；开启中位2.309372秒，范围2.301004–2.379446。缓存未受控，两组范围重叠；不声明真实输入开销低于2%，不声明性能提升。
- 上述基准输出仅本地：`/var/folders/nh/7p0c6_gx5w9cmfd7rxhc9rtw0000gn/T/tf035-overhead-a3oy1ecj`。配置为独立合成fixture，不含真实用户样本。

## 全仓检查

`UV_PROJECT_ENVIRONMENT=/Users/yujiatang/doku/doku-train/.venv make check`：锁检查、格式、lint、mypy通过；非TF测试326 passed、3 skipped。TF全量195 passed，220.14秒；make check退出码0。随后新增的失败子进程统计用例包含在上述5项专项验证中。

首次受限执行因uv缓存目录无写权限未能启动；经工具批准后使用现有缓存重跑，无云操作。

## 尚未验收

- 真实tf-lr002冷缓存完整基准、最终test及保存计时、配置/源码/输入冻结核对。
- 大样本观测开销≤2%、CPU/worker梯度、代码加速及≥50%全流程收益。
- 数值容差需正式实验前确定；本地完全一致结果不等于跨机器浮点结果完全一致。

最终文档检查：`git diff --check`通过；Spec 035四个索引目标存在。最终`ruff check .`及`ruff format --check .`通过（370文件）。


## 后续：三项代码优化与平台筛选（2026-11-05）

用户明确授权两epoch平台基准、资源对照及一致性通过后的代码优化实验。

- `TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 PYTHONPATH=. recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests/test_input_pipeline.py -q -p no:cacheprovider --disable-warnings`：14 passed，29.64秒。三个单项及组合两轮训练loss/指标/checkpoint逐tensor完全一致，各自恢复一致，shuffle/batch边界不变；包括空历史、变长历史权重、Unicode宽度、大整数无float精度损失、有界缓存及预取FIFO/异常/取消。
- 优化前的合并专项组19 passed（41.26秒）；随后对列式gather复用排序索引的小优化已再次通过上述14项。
- CodeBuild基准镜像构建成功：`doku-train:58f93c58-393a-42c1-bc5b-47a6a6c4e72d`；digest `sha256:5d5699c26321f5a262693ebb06c62502fbda16285e283fcd2c9a663bc3ccc9d8`。
- CodeBuild优化镜像构建成功：`doku-train:a1dd679e-75e6-46af-86f9-9802bd239b7f`；verification阶段运行test_input_pipeline.py与test_training_timing.py，通过后才push；digest `sha256:4ecf5c3ee878795643d34ceba71cd3ee49697535bb68bcee6bd851200dc05bf8`。本地检查确认镜像源码归档的三个核心实现文件与当前工作区逐字节相同。
- 合成16384行编码微基准：rows三次0.1178/0.1162/0.1181秒；columns 0.0180/0.0177/0.0168；cache 0.0162/0.0106/0.0099；prefetch 0.1164/0.1174/0.1194。仅编码无TF重叠、不含准备/计算，不外推真实训练加速。

实际运行请求与回执仅本地runs/tf035/，不随源码发布。基准/线程/读取共6组已提交；m5.8xlarge/16xlarge因服务不支持被拒绝，已改为m5.12xlarge/24xlarge（48/96vCPU）并提交成功。固定原输入、batch2048、seed与模型，max_epochs统一2；不配置test，不发布线上模型。所有成功提交任务的上限为10800秒。

资源初始进度：base/intra8/intra16/reader2/reader4/reader8/cpu96进入Training，cpu48先在Downloading。状态是查询时快照，不代表完成。首个基准约361秒时已处理约161.8万行，列解码累计141.0秒、分片下载27.6秒；处于准备阶段，尚无完整epoch分解，不据此判定最终瓶颈。


最终全仓命令：`UV_PROJECT_ENVIRONMENT=/Users/yujiatang/doku/doku-train/.venv make check`退出码0；锁检查/格式/lint/mypy通过；非TF326 passed、3 skipped（179.20秒），TF210 passed（210.76秒）。最终`ruff check .`、`ruff format --check .`（372文件）、`git diff --check`通过。

优化两epoch任务全部提交成功：doku-exp-tf035-1105-173748-{optcontrol,columns,cache,prefetch}。资源与基准：doku-exp-tf035-1105-172449-{base,intra8,intra16,reader2,reader4,reader8}、doku-exp-tf035-1105-172622-{cpu48,cpu96}。总计12个任务；只声明提交和已查询状态，不声明加速比例、完成或全量验收。代码未提交/推送，线上模型和Endpoint未修改。


## 两epoch实验回收与分析

Source: 本会话用户要求回收全部任务的耗时并分析；各任务最终DescribeTrainingJob及独立S3产物。

- `python runs/tf035/collect.py`：12/12训练任务Completed；回收各自的benchmark-resources、run-resources、timing-summary、training/metrics、配置/lineage及输入Manifest。默认关闭优化的任务不生成input-pipeline.json属于预期，其他必需产物齐全。
- `python runs/tf035/analyze.py`：逐组校验完成状态、2个epoch、输入Manifest一致、实际任务配置摘要等于冻结配置；12×2个epoch所有非计时字段与基准完全相同。没有声称云checkpoint逐tensor对比已执行。
- `python runs/tf035/report.py`：生成本地analysis.md、results.csv；阶段核算总和与基准完整执行时间相等（浮点舍入容差）。完整JSON和报告仅本地runs/tf035/，不随源码发布。
- 结论：有界预取和批量编码为有效候选；编码缓存不优于单独批量编码且内存增加。资源扩容未显示规模性收益，reader2/4仅改善准备。单次两epoch不代表独立重复或完整多轮验收，尚未达到完整流程50%目标。
- 未提交新的联合任务，未修改运行配置默认值、线上模型或Endpoint。下一候选是columns+prefetch，再考虑reader2/4；联合收益尚未验证。


## 精确columns＋prefetch组合补测

- `TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 PYTHONPATH=. recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests/test_input_pipeline.py -k columns_prefetch -q -p no:cacheprovider --disable-warnings`：2 passed、14 deselected，16.69秒。分别用1/4计算线程验证缓存关闭的精确组合；两轮loss/指标及checkpoint每个tensor与原路径完全一致，恢复第二轮一致，样本顺序和batch边界一致。
- 原优化镜像源码归档与当前生产Python文件逐字节核对：45个文件一致；本次只补测试，复用已有镜像，无需重构镜像。
- 云端此前12组的全部非计时epoch字段精确一致；这不同于云端checkpoint逐tensor比较，后者尚未执行。新组合的云端一致性与耗时仍需该任务完成后验证。

唯一组合任务已提交：`doku-exp-tf035-1105-200202-columns-prefetch`。本地配置校验确认相对optcontrol仅input_pipeline不同；未启动其他组合或资源实验。


## 精确组合云端结果回收

- `python runs/tf035/collect_combined.py`：组合任务Completed，回收完整计时、指标、输入Manifest与配置产物。
- `python runs/tf035/analyze_combined.py`：断言组合及三个对照全部完成、各2epoch、Manifest一致，全部非计时epoch字段与原路径精确一致。
- 结果保存于本地runs/tf035/combined-analysis.json及combined-analysis.md；组合优于单独预取，为当前候选，但完整两epoch流程尚未减半，40轮推算不作为实测验收。
- 本次仅回收分析，没有新提交云任务或修改默认实现。文档变更执行git diff --check。


## 原b4096任务优化重提

Source: 用户明确要求用最新优化代码重提原b4096任务。

- 原配置下载字节SHA256与原任务环境声明一致；原runner SHA256 f1cfc21bc99d5eb68136ece29a99778f40ebfb685a2b40847fd998958fa49458保持不变。
- 提交该任务时45个生产Python文件与优化镜像归档逐字节一致。新配置去掉input_pipeline后与原配置完全相同，batch4096、max_epochs40、TF线程4/1、原日期与模型不变。
- 唯一配置增量：batch_encoding=columns、prefetch_batches=2，encoded_cache_bytes默认0。
- 提交成功：doku-exp-tf035-b4096-opt-1105-204310；原任务未停止/修改；完整请求、配置与回执仅本地runs/tf035-b4096/。不复用两epoch runner，不修改原24小时MaxRuntime。
- 提交时完整训练尚未完成；完成后的指标一致性证据见下节。


## 2026-11-06 b4096 40epoch TensorBoard对照

Source: 用户要求优化任务与对应baseline的逐epoch指标放入TensorBoard并对比一致性。

- `python runs/tf035-b4096/tensorboard-comparison/collect.py`：两任务均Completed，各40轮；下载各40个原始TensorBoard event文件，保留原始JSON。
- `python runs/tf035-b4096/tensorboard-comparison/validate.py`：输入Manifest一致、配置除input_pipeline外相同；40轮4200个非计时字段逐项完全一致，最大数值差0；最佳epoch均8、score一致。
- EventAccumulator逐tag/step核对两run各49标签、1960点，完整覆盖epoch1–40，数值与原始JSON对应float32完全一致；高精度一致性来自JSON比较，不依赖TensorBoard的显示精度。
- 本地TensorBoard使用现有2.21.0环境提供scalar视图，127.0.0.1:6006；HTTP检查两run及主要loss/AUC/耗时标签的40轮数据。运行产物及详细报告仅本地runs/tf035-b4096/tensorboard-comparison/，不随源码提交。
- 本次未修改训练实现、配置默认值或云资源；没有声明云checkpoint逐tensor已比对。


## 2026-11-06 PR最终验证

- 已同步main `1178eac8b5a7329a4485c35fd649193879877a02`，保留优化的显式开关与原默认值。
- `UV_PROJECT_ENVIRONMENT=/Users/yujiatang/doku/doku-train/.venv make check`：退出码0；锁文件、格式、lint、类型检查通过，非TF 329 passed/3 skipped，TF 213 passed，inference 8 passed。
- 提交范围仅16个源码、测试及文档文件；实验配置、数据、模型和TensorBoard产物不提交。`git diff --cached --check`通过。
- 云端40轮平均训练耗时516.962→250.397秒（缩短51.6%），完整训练流水线23274.582→12327.361秒（缩短47.0%）。此为该冻结数据与机器上的单次配对结果；完整流程50%及多次独立重复仍为后续目标。
