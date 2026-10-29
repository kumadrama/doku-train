# opt40 / prep40 一致性排查（2026-10-29）

Source: 用户授权按真实切片、同一初始 checkpoint、单步/短程训练、原生 Linux 的顺序排查，不影响现有任务。

## 范围与隔离

使用两次实际上传的 source.tar.gz 中 recommend_tf 源码，分别独立进程加载。生产 worktree 的模型、训练运行时和优化器没有修改；诊断代码、报告仅在 gitignored `runs/finder-parity-investigation-20261029/`。云端只运行独立 CodeBuild，不提交新40轮训练，不更新已有镜像/任务或IAM。

原始切片取冻结 Manifest 的10月1/12/23日各前32768行，共98304原始行，过滤后93582行；每轮46批，batch2048。使用正式训练的完整冻结词表。两版分别执行各自 reader、prepare、fit_vocabulary；比较全字段逻辑样本、词表、无shuffle及seed42/43/44的输入dtype、shape及字节。Parquet文件字节不作为逻辑样本一致性的替代。

初始化以 opt40 构建的同一 checkpoint 为准，包含模型、AdamW slots、beta powers和global step；每次训练在独立进程、独立图中恢复，更新前全部变量摘要逐项校验。固定Python、NumPy、TF seed和PYTHONHASHSEED。原实验4/1线程。

## 诊断采集

只在隔离运行时包装历史 graph_helper.optimize_loss，暴露它已经生成且真正提交给优化器的梯度，不另构造一套反向图。采集 TRAIN 中间输出、logits、任务loss、实际更新梯度，以及单次 sess.run 完成后的全部参数/slots；不把 EVAL forward 混入训练更新 fetch。每次完整切片训练后，对同一切片执行 EVAL，比较预测/loss字节。这是数值一致性诊断，不是独立验证集质量评估。

初版额外构造梯度图/同时读取EVAL debug，存在观察干扰风险，local-output及worker-v1.py仅作排查过程记录，不用于最终结论。最终原调用链对照使用 worker.py/local-output-v2，以及对应Linux归档。

## 已确认结果

数据层本地及Linux均通过：93582行逻辑样本、全部词表、4种顺序的全部46个batch一致。

在同checkpoint下，原调用链仍不满足严格训练一致性。Linux第一轮关闭dropout时跨版本46批最大loss差0.00044602155685424805；开启dropout为0.0008788108825683594。两版自身重复也不同，不用“处于自身波动范围”作为通过标准。

额外诊断仅修改隔离调用链：

[] 零衰减：weight_decay=0，用于隔离衰减影响，不作为正式修复。
[] 有序更新：保留原衰减系数、学习率、数据和数学公式，在创建 apply_gradients 及其 decay op 前，加上对全部 dense gradients 和 sparse gradients values/indices 的控制依赖。

本地诊断结果：零衰减自身重复，以及有序更新的自身/跨版本、dropout关闭/开启，全部46批的输入、logits、loss、梯度、参数/slots和切片EVAL字节精确一致，最大loss差0。Linux有序更新及零衰减诊断的7组比较也全部精确一致，最大loss差0；包括dropout开启/关闭、跨版本和自身重复。

## 原因与限制

原迁移版调用链直接创建 apply_gradients；AdamW 的 decay assign 与梯度计算之间缺少完整执行依赖。它保证decay在Adam参数更新前，却没有保证decay发生在所有梯度计算后，可能在本batch仍读权重时改写权重。诊断中只增加此依赖即可消除本地差异，零衰减对照进一步支持这一归因。不能将此结论推广成“Finder所有部署均有相同表现”，其原分布式执行环境不在本次审计范围。

此前另已证明输出层初始化仅固定TF seed仍不足。初始化缺口与本次执行顺序缺口是两个独立问题；同checkpoint对照排除了前者后仍复现后者。缺少两次真实云训练第0步权重，不能量化它们分别造成多少最终AUC差异。

完整3294879条源样本尚未扫描；本轮已在首个训练差异处定位原因，未将切片通过声明为全量或40轮轨迹通过。正式修复、回归测试和全量扫描作为下一阶段，现有运行实验保持原样。

## 证据

运行脚本：run_audit.py、worker.py、run_diagnostics.py、diagnostic_worker.py。样本和checkpoint只用于隔离计算，不写入源码仓库；日志只输出摘要。

本地：local-output-v2/{report,diagnostic-report}.json。Linux原调用链：CodeBuild b69c7bcc-17ad-4219-bfb8-a6efaec0ad3d，linux-baseline-evidence.json。计算已完成，但后续S3报告上传缺少KMS GenerateDataKey权限，构建状态为FAILED；结果从CloudWatch恢复，未修改权限。第一次原始分片直读403后，使用现有已授权身份的短期只读下载链接；没有公开数据或更改访问策略。


## 最终 Linux 证据与下一阶段

第二轮 CodeBuild `doku-train:b8977218-af13-482a-8874-46ec2d764c6e`，源包摘要 `8b72e53df736dbb218f937385b461f0ef0b314fb4e9c990a616244a2a794fb34`，Linux x86_64、TF2.11.1，使用 opt40 不可变镜像依赖。完整执行8次原调用链短程训练和10次诊断短程训练，每次46批；最终7组诊断比较全部exact=true、max_loss_abs=0、切片EVAL完全一致。证据：linux-final-evidence.json与linux-final.log。S3报告导出同样被KMS权限拒绝，构建收尾FAILED；计算结果均从CloudWatch读取并本地断言验证，未将构建状态伪称成功。

本轮第1～5步定位完成。候选正式修复边界：统一设置Python/NumPy/TF随机状态；在本地graph_helper创建任一参数更新前，等待全部实际dense/sparse梯度完成；保留Finder AdamW衰减公式、系数和稀疏命中行范围。应增加图依赖结构检查和独立进程严格训练对照作为回归。此为已实证的隔离诊断改动，尚未修改生产实现；下一阶段正式修复后进行全量输入扫描，不重启已有任务。

## 正式实现（2026-10-29 用户追加授权“修复这两处”）

上述“尚未修改生产实现”描述为排查结束时状态。本轮已在独立worktree正式修改 training_pipeline 与 graph_helper，未部署：seed_graph统一设置Python/NumPy/TF；all_gradients_ready屏障等待全部实际梯度，所有optimizer apply操作在其控制依赖下构造。Finder weight_decay_optimizers.py未改动。已新增结构依赖及数学检查、独立进程完整训练与恢复回归。本地41项测试通过；正式代码原生Linux验收记录见 verification.md。

## 全量 15 epoch 重跑（2026-10-29 用户授权）

用户确认不再比较历史base，分别重跑opt40和prep40，各15个epoch，检验修复后的跨版本一致性。隔离目录：`runs/finder-parity15-20261029/`（仅本地，不随源码发布）。从两份历史源码快照分别构建，只修改training_pipeline随机种子与graph_helper更新依赖，加入复现回归测试；保留两版原有输入/拟合差异，不引入后续L2功能。使用相同不可变opt40依赖镜像，配置仅将max_epochs从40改成15，沿用ml.m5.2xlarge（8 vCPU）、4/1线程、seed=42、batch=2048与8192有界shuffle。

先分别进行原生Linux验收，通过后启动独立SageMaker任务；不修改或停止既有任务。完成后比较全部15轮逻辑指标、冻结输入证据、词表及checkpoint内所有变量（含optimizer slots/global step）的名称、形状、dtype与数值；时间、路径、source digest等运行元数据分别列示，不纳入数值相等判定。不能仅凭AUC显示值一致声明完全相同。全量训练一致性仍待实际结果。

## 16 vCPU 线程基准（2026-10-29 用户授权）

在独立ml.m5.4xlarge任务中使用已修复prep15不可变镜像，固定此前93582行切片、完整冻结词表、同一初始checkpoint、seed=42、batch=2048和shuffle buffer=8192。不修改两组15轮任务。inter_op=1，intra_op按[4,8,16]、[8,16,4]、[16,4,8]交错执行，每次独立进程恢复同一checkpoint。记录读取/shuffle、tensor_batch+feed组装、正常训练fetch的sess.run耗时，以及全部46批与排除首5批的稳态时间；不在计时循环内抓取梯度或参数。基础数据准备仅一次，独立于每轮输入处理时间；记录切片逻辑摘要并与前次一致性验收摘要核对。每次训练前后在计时区间外校验变量摘要，并记录跨线程loss差异，不假定浮点归约跨线程逐位一致。

运行脚本及结果仅存本地 `runs/finder-threads16-20261029/`；本地两批smoke已通过，正式16 vCPU结果待返回。该基准检验训练输入/计算吞吐，不直接等同完整实验总耗时或训练前拟合耗时。
