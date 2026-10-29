# 首次 Finder TF 40 epoch 训练

Source: 2026-10-29 用户授权「验证通过后启动第一次40 epoch训练」。

## 配置

基于已确认 extended40-lr004：train 10/01–23、validation 10/24、test 10/25（UTC）；40 epoch、batch2048、seed42、lr0.004、Finder AdamW decay0.0001。七任务及模型配置见本地 runs/finder-e1-extended40-lr004-review/config.json；无PyTorch正则等价声明。

复用已有 CodeBuild/ECR/SageMaker角色及网络；单机 ml.m5.2xlarge、100GB卷、最长86400秒。训练需要读取S3和写出run-scoped指标，网络隔离关闭，仍使用既有VPC。

## 保存与入口

- 每轮 training/epoch-metrics/NNNN.json 保存 train loss、validation loss和配置指标、阶段耗时。
- training/tensorboard 记录同样的数值，横轴为epoch；null指标只在JSON保留原因。
- 每轮checkpoint、指标、关闭的event文件通过S3增量上传；prepared/cache不上传。
- 正式入口校验源码/锁/config摘要；train完成后以validation选模，再独立test。
- 训练集TensorBoard目前保存loss，未额外计算训练集AUC；validation包含AUC等配置指标。
- 中断时保留已上传轮次。最终完整metrics/lineage及成功状态不会在未完成40轮时提前写出。

## 已执行验证

- `UV_CACHE_DIR=/tmp/spec024-uv make tf-check`：14 passed，包含epoch event解析、null指标、上传失败重试/范围、真实入口选模test、模拟中断轮次保留。
- 完整extended40-lr004配置的本地合成验证：2 epoch、7任务、AdamW、dropout恢复一致性、独立test通过。正式配置仍为40 epoch。
- `ruff check recommend_tf --no-cache`、`git diff --check`通过。

构建、同镜像Linux验证及正式任务状态以本地 runs/finder-first40-20261029 下的构建日志、request、submission、status为准，后续追加实际结果。


## 原生Linux验证与任务提交结果

- 同镜像原生Linux测试：14 passed，21.08秒；TF2.11.1、Arrow17.0.0、Python3.10.22。
- 任务：[doku-exp-finder024-40ep-20261028-164625](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/doku-exp-finder024-40ep-20261028-164625)。
- 提交后查询状态：InProgress / Starting，尚未宣称完成40轮。
- 镜像：`703844615824.dkr.ecr.us-east-2.amazonaws.com/doku-train/doku-reco-train@sha256:821b31052b5175956224969db8628c7fd669002ec7a8dd4f586fdb55d4a01462`。
- 源码摘要：`4aa02c39563d134003ecd1763f2ab81de82d8b46b661f9243502d486f5041fb7`；配置摘要：`596ab4293e65655df0dd51fb14d0843f24b0931247c4300bb6478e618d525d1e`。
- 每epoch持久化路径：`s3://prod-data-lake-20260901071345278900000001/ml/sagemaker-experiments/environment=prod/runs/finder024-40ep-20261028-164625/live/`，TensorBoard位于该路径的 `training/tensorboard/`。
- 仅启动一个正式SageMaker任务；CodeBuild测试未额外启动训练任务。


## 输入优化版第二次40 epoch训练（2026-10-29）

用户明确授权启动独立任务验证耗时，旧任务不变。新镜像基于首次任务不可变镜像构建，配置字节摘要、锁文件、实例规格、VPC、时限均逐项核对一致；仅输入列投影、列式行引用、RSS检查降频及对应测试变化。

- 任务：[doku-exp-finder024-opt40-20261029-032036](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/doku-exp-finder024-opt40-20261029-032036)。
- 同镜像原生Linux测试：17 passed，20.19秒；源码摘要 `e527374249b0a0c1f8d068544f5ec4c963b6f1c85f9618d4795534d95c5ef2a5`。
- 镜像：`703844615824.dkr.ecr.us-east-2.amazonaws.com/doku-train/doku-reco-train@sha256:7bce4d63f895a27eadb1a1626b1e308bf1697de289db6d9100259da53d905b66`。
- 提交成功，首次查询 InProgress / Starting；尚未取得新任务完整epoch耗时，不宣称已验证云端提速。
- 逐epoch指标、TensorBoard、checkpoint存入独立路径：`s3://prod-data-lake-20260901071345278900000001/ml/sagemaker-experiments/environment=prod/runs/finder024-opt40-20261029-032036/live/`。
- 只读对比脚本本地 `runs/finder-optimized40-20261029/compare_timing.py`，记录准备、词表、train_with_input、validation_with_input和epoch wall；结果 `timing-comparison.json`。完整40轮及test待任务完成后验收。

## 准备与词表优化版第三次40 epoch训练（2026-10-29）

用户明确授权将本次一次性准备/词表拟合优化加入同配置40 epoch，启动独立任务验证云端耗时，不影响在跑任务。

- 任务：[doku-exp-finder024-prep40-20261029-040027](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/doku-exp-finder024-prep40-20261029-040027)。
- 直接对照第二次输入优化任务 `doku-exp-finder024-opt40-20261029-032036`；包含上一轮输入优化及本轮批量时间校验、列式准备、列式词表拟合。40 epoch、batch2048、seed42、4 intra/1 inter线程、shuffle8192、ml.m5.2xlarge单实例/100GB卷/24h上限不变。
- 配置摘要仍为 `596ab4293e65655df0dd51fb14d0843f24b0931247c4300bb6478e618d525d1e`；锁摘要仍为 `b2763acbc6cb48f18bd9326b605e9bf7634a6d3a4ec05355023c8f8c983d0bd5`；请求中角色、VPC、网络/Spot选项、资源和时限与第二次逐项核对一致。
- 源码摘要 `8eaeae2b3d0d92cf214751f264b1b4726252f1f59fb10c7723acbfc96ef12c7a`。
- 新镜像 `703844615824.dkr.ecr.us-east-2.amazonaws.com/doku-train/doku-reco-train@sha256:5ccb5dd082b21f499bb84c3aa8cea64a008798a2ea8a656eebd5f6df0b097fa4`。
- CodeBuild `doku-train:bcf55427-822f-4b61-ab07-87695ea0b959` 成功；同镜像原生Linux amd64 / Python3.10.22 / TensorFlow2.11.1 / Arrow17.0.0：22 passed，21.06秒。验证源码和锁摘要均与提交身份匹配。
- 新任务已提交，独立持久化路径 `s3://prod-data-lake-20260901071345278900000001/ml/sagemaker-experiments/environment=prod/runs/finder024-prep40-20261029-040027/live/`；逐epoch TensorBoard、指标、checkpoint及最终选模/test入口沿用。
- 提交前只读确认已有两个任务仍为InProgress/Training；没有调用StopTrainingJob或更新它们的配置、镜像、输出路径。全部操作记录在本地 `runs/finder-pretrain40-20261029/`。
- 云端实际耗时尚待日志产生，未将本地37.5%作为云端结果。三任务只读对照脚本 `compare_timing.py` 比较prepare、fit_vocabulary、epoch阶段耗时；本次完整40轮/test验收仍待完成。

## 七任务TF搜参提交（2026-10-29）

用户确认纯Adam（同时取消衰减）及额外同资源baseline。以下七任务已提交，全部使用最新准备/词表拟合/输入优化代码，40 epoch、ml.m5.4xlarge（16 vCPU）、4 intra/1 inter线程、100GB卷、24h上限。其他基础配置、冻结数据及逐epoch保存/选模/test入口不变。

| 组 | 独立SageMaker任务 |
| --- | --- |
| baseline | [doku-exp-f024-baseline-20261029-072135](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/doku-exp-f024-baseline-20261029-072135) |
| lr001 | [doku-exp-f024-lr001-20261029-072135](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/doku-exp-f024-lr001-20261029-072135) |
| lr002 | [doku-exp-f024-lr002-20261029-072135](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/doku-exp-f024-lr002-20261029-072135) |
| adam | [doku-exp-f024-adam-20261029-072135](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/doku-exp-f024-adam-20261029-072135) |
| wd0 | [doku-exp-f024-wd0-20261029-072135](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/doku-exp-f024-wd0-20261029-072135) |
| wd3e5 | [doku-exp-f024-wd3e5-20261029-072135](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/doku-exp-f024-wd3e5-20261029-072135) |
| ln | [doku-exp-f024-ln-20261029-072135](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/doku-exp-f024-ln-20261029-072135) |

统一不可变镜像：`703844615824.dkr.ecr.us-east-2.amazonaws.com/doku-train/doku-reco-train@sha256:0d356e76c448f3ff45181380409efacf5b60c8eaadef482c04900c65c0527161`。
源码摘要：`96a5ec1b5f15edc2ef076e441d277f158cb67e4c4c7df6a88d67437a4b2e6fb0`；锁摘要：`b2763acbc6cb48f18bd9326b605e9bf7634a6d3a4ec05355023c8f8c983d0bd5`。
CodeBuild `doku-train:88e6abde-a473-4205-a8e4-24f749043b3c`成功：原生Linux amd64同镜像25 passed，31.68秒；七份配置差异及优化器实例化校验通过。
五个标准变体仅修改各自指定参数；纯Adam组同时修改优化器并删除weight_decay，按用户确认作为例外，不标成严格单变量。LN迁移Finder公共affine实现，gamma/beta开启，epsilon=1e-6，隐藏层Dense后、ReLU前应用。
七任务使用同一镜像摘要、七个独立S3 live/artifacts前缀。没有停止、修改旧任务。请求、响应、配置摘要、原生验收证据保存在gitignored `runs/finder-six-experiments-20261029/`。
当前仅确认验收和提交成功，完整训练、性能与指标结果尚待完成。
