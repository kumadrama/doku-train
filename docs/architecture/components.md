# 组件与依赖

Source: Spec004 与本轮实现。当前入口已切换为声明式配置，不兼容旧固定任务/日期格式。

- `run.py`：validate/train/test/search/report 参数解析。
- `train_rerank.py`：模型延迟装配与独立进程调用；仅在 composition root 选择模型实现。
- `comm/model_params_validator.py`：加载公共及插件schema；结构和跨字段校验。
- `comm/expressions.py`：受限AST、[*]与feature_list展开；不执行任意Python。
- `comm/datasvr/published_dataset.py`：v1 adapter、合同版本、摘要、分片和UTC日期校验。
- `comm/feature_plan.py`：默认值、log、训练词表、共享历史截断和树标量；编码结果落盘。
- `comm/training_pipeline.py`：数据隔离、训练与最终评估、create-only输出及lineage。
- `comm/task_metrics.py`：任务指标，不可计算指标记录原因，没有质量门禁。
- `comm/search.py`：配置路径覆盖、声明式网格、逐task比较和报告。
- `models/rerank/rerank_model.py`：共享embedding、DNN、可选FM与任务塔。
- `models/rerank/runner.py`：CPU优化、验证选择、epoch checkpoint恢复。
- `models/rerank_tree/runner.py`：CPU hist独立Booster、目标与轮次按task配置。

不创建models/deepfm。神经与树运行时分进程加载，避免本机OpenMP冲突；两者之间不共享可训练参数。

## SageMaker 平台适配（已部署，单个 DNN 云训练通过）

Source: [Spec 008 需求](../../specs/008-sagemaker-dnn-training/requirements.md) 与
[设计](../../specs/008-sagemaker-dnn-training/design.md)，2026-10-19 用户会话确认。

当前状态：已增加 SageMaker 专用入口、冻结提交包、产物校验和 step 事件同步。
目标状态 [INFERRED]：在 recommend/train/platforms/sagemaker 增加提交、运行与归档适配，
调用现有训练用例；comm/models 不依赖 SageMaker。云端运行配置独立于实验 YAML。
IAM、专用 SG、网络/存储授权与 binding 归 doku-iac；不另建训练仓库。
首期仅实验训练，不切换线上模型；2026-10-20 需求与设计已确认，实施顺序见 [Spec008 任务](../../specs/008-sagemaker-dnn-training/tasks.md)，本地及原生 amd64 合成容器验收已通过。

Source: 2026-10-19 用户后续确认：Spec008 首期不创建 Studio Domain，本地提交，
通过 SageMaker 控制台 / CloudWatch 查看 AUC/loss 曲线；指标适配属于平台层，
后期 Studio 接入复用训练接口与产物，首期平台适配已实现，Studio 接入仍为后续目标。

Source: 2026-10-20 Q20/Q21 确认：step 曲线使用本地 TensorBoard（S3 同步），
AWS 控制台/CloudWatch 继续用于任务、日志与时间指标；SageMaker 服务时限显式设为 86400 秒。

Source: 2026-10-20 用户授权单任务云端验收。当前状态：专用 IAM/网络已部署，真实 DNN train/test/归档及 33 条 step 指标数据验证通过；浏览器视觉验证受 Work 设置阻塞。DeepFM 仅验证原生合成容器，15 并发和每日自动训练未执行。详见 [Spec008 验证](../../specs/008-sagemaker-dnn-training/verification.md)。

## 公共 loss（Spec 016，当前实现）

Source: 2026-10-23 用户确认的 [Spec 016](../../specs/016-shared-nn-loss/design.md)。

沿用 Finder comm/nn_utils.py:get_loss 的公共入口与按任务类型分派结构，
使用 PyTorch CPU 实现。神经网络训练、验证和 logloss 指标共用该入口；
树评估传入原始 margin。task_metrics 不再维护独立 logloss 公式。
标签、mask 与表达式由调用方解析，训练保留加权均值，验证保留无权均值；
指标按自身标签与 mask 计算。极端预测直接按 logits 计算，不再概率截断。

### Spec 016 公共 NN loss（当前状态）

NN 训练、验证、测试的 loss 与 logloss 指标共用 `comm/nn_utils.py:get_loss`，按 Finder 的公共入口与任务分派结构组织。树的 logloss 位于 `models/rerank_tree/loss_utils.py`，由树 runner 注入指标层；comm 不反向依赖树，NN 不依赖该目录。底层运行时仍为 PyTorch CPU。

Source: Spec 016；2026-10-23 用户确认及树实现独立保留的补充要求。

## NN 字段组合与历史聚合

Source: [Spec 015](../../specs/015-parameter-free-attention/design.md)，2026-10-21 用户确认拆分语义。
当前本地 rerank 先用 combiner（concat/sum/avg）组合字段，再用 history_pooling（mean/weighted_mean/attention/none）归约历史位置。多字段历史和标量 query 均先组合，attention 为固定尺度点积、masked softmax、加权和。DNN/DeepFM 共用，none 保留序列并拒绝直接送入 DNN/FM。
公共 FeaturePlan 负责对齐检查与同步 tail 截断，模型负责组合、聚合和维度验证；不引入公共层反向模型依赖。旧 pooling 配置需要显式迁移，树模型不支持神经聚合。尚未部署新训练镜像。


## 配置驱动的树阈值前置训练

Source: [Spec 017](../../specs/017-tree-threshold-binning/design.md)，2026-10-23 用户确认。

train_rerank composition root 装配公共前置编排与模型专属阈值导出；comm 保持不依赖具体模型。NN 特征拟合前加载 train-fitted 的冻结树阈值，全桶域编码进入 Feature State。目标和验证状态见 Spec 017。


## DeepFM 耗时观测（Spec 019）

Source: [Spec 019](../../specs/019-deepfm-timing/design.md)，2026-10-27 用户确认。

当前状态：公共 timing collector 以上下文 session 装配，公共数据组件不依赖具体模型。
rerank schema/validator 限制仅 DeepFM 可开启 profiling，默认关闭；train/test/归档各有
独立 invocation。stdout 和 Run JSONL/summary 共用累计事件，普通失败保留诊断。
profiling 不参与训练逻辑 identity，不改变模型、特征或选模；SageMaker 收录计时产物。
目标状态：本地验证后可另行授权更新镜像和云端诊断；当前未部署，不声明生产性能收益。

## 批量特征准备（Spec 022，当前本地实现）

Source: [Spec 022 设计](../../specs/022-batched-feature-preparation/design.md)，2026-10-27 用户确认并授权本地实施。

`train_rerank` / `eval_rerank` 在 composition root 为神经网络注入 BatchFeaturePlan；
公共 pipeline 只依赖 fit_dataset / prepare_dataset 接口，独立树继续 FeaturePlan。
批计划在初始化时构建，复用旧历史算法；批读取保持分片/时间校验，训练拟合携带完整宽度以省去
重复扫描，validation/test 仍扫描。编码按块写 memmap，原词表编号与状态合同保持。
通用表达式采用保留 Python 标量语义的列内核，对数桶提供受保护的 NumPy 快路径；
不等同于所有表达式已成为原生算子。当前仅本地验证，不声明云端性能或已部署。


## Spec 024：独立 Finder TF CPU 路径

Source: [Spec 024 设计](../../specs/024-finder-strict-migration/design.md)，用户批准实施。

当前状态：独立 `recommend_tf/train` 已建立本地迁移实现，依赖与 `recommend` 隔离；
图表达式与主模型来自 Finder，Doku Manifest/合同读取适配到独立命名空间。
训练前基础分片与固定词表、图内标签/loss_weight、本地优化器与epoch恢复均有合成测试。
目标状态及完整边界以 Spec024 为准。原生 Linux amd64 完整合成链路已通过 SageMaker 验收；真实数据性能尚待验收，
本地合成测试不代表云端部署或当前 e1 实验数值等价。

### Spec 024 训练前批量准备（当前本地实现，2026-10-29）

Source: [Spec 024 设计](../../specs/024-finder-strict-migration/design.md)。

TF PublishedDataset 批量验证带时区的审计时间，train_rerank 保留列块，prepared 只对历史字段复用逐行整理；train-only 词表拟合直接读取列批并组装张量。模型图中的表达式、lookup 与训练 shuffle 不变。本地代码变化不代表正在运行的云任务已经更新，性能证据见 Spec 024 verification。

### Finder affine LN（当前本地实现，2026-10-29）

Source: [Spec 024搜参扩展](../../specs/024-finder-strict-migration/hyperparameter-experiments.md)。公共comm/layernorm.py迁移Finder LayerNorm affine路径；模型层在Dense与ReLU之间通过norms="ln"启用，gamma/beta参与训练并保存checkpoint。默认空norm不变，无动态Embedding依赖。仅本地验收，本轮搜参尚未提交。

### TF Adam loss 正则（2026-10-29，本地实现）

Source: Spec 024 regularization-design.md 与用户确认。comm/regularization.py 汇总配置层 Dense L2 与有效 lookup 位置的 Embedding L2；model_fn 将其加入 TRAIN 目标。输入、选模和验证 task 指标不变；关闭时不增加正则。原生 Linux 合成验收38项通过，用户授权新增五组实验；真实数据结果待完成，已有云任务保持原镜像。

### TF 可复现性（2026-10-29，已修复待部署）

Source: Spec024 consistency-investigation.md、design.md、verification.md及用户修复授权。每轮构图前固定Python/NumPy/TF随机状态，沿用seed+epoch；全部实际梯度计算完成后才允许任一优化器更新参数，消除本地Finder AdamW衰减的执行次序竞态。Finder衰减公式不变。本地及原生Linux各41项测试通过。已有训练任务仍使用原镜像，未部署本修复。


### Spec 024 合入状态（2026-10-29）

Source: 用户确认 opt15/prep15 逐 epoch 一致并授权 PR 合入；[合入验证](../../specs/024-finder-strict-migration/verification.md)。

当前交付包含独立 Finder TF 路径、列投影/列数组行视图与 RSS 检查降频、列式基础输入准备与 train-only 词表拟合，以及统一随机状态/跨优化器梯度屏障。上述历史阶段中的“待验收”不代表当前状态。两项输入优化保持模型图、loss、shuffle 随机调用顺序和 batch 边界；RSS 检查降频改变超限检测时延。现有 PyTorch/XGBoost 路径不变；云端实验镜像不因源码合并自动更新。


## Spec 027：TF 日更与 Endpoint 发布（目标）

Source: 2026-10-30 用户确认模型提供方边界与 Q12 B；[需求](../../specs/027-tf-daily-serving/requirements.md)、[设计](../../specs/027-tf-daily-serving/design.md)。

当前状态：新版 TF 已有训练/validation/选模和离线 test；Spec 027 已新增本地两阶段日更、独立导出和平台发布代码，验证见 [Spec 027 记录](../../specs/027-tf-daily-serving/verification.md)；未部署云端。

目标状态：`train/daily_train_model.py` 根据 `online_config` 执行冻结的 30+1 训练，复用公共数据准备及 `training_pipeline.run()`；离线实验三阶段不变。`train/export_model.py` 从成功 run 导出推理包；`platforms/sagemaker` 在上游发布成功触发的一次性 ECS 中编排训练、导出及固定 Endpoint 更新。新实例就绪后一次性切换，失败保留旧版本。

本仓不新增 `serving/` 或自研 C++ 服务。独立 serving 仓库交付 TensorFlow Serving 推理镜像与版本化合同；本仓通过固定镜像和制品部署，不跨仓 import 实现。IAM/网络/触发属 iac，监控属 monitoring，调用方实验与业务排序不属于本次工作。
