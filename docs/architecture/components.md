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

## SageMaker 平台适配（本地已实现，云端未部署）

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

## NN 字段组合与历史聚合

Source: [Spec 015](../../specs/015-parameter-free-attention/design.md)，2026-10-21 用户确认拆分语义。
当前本地 rerank 先用 combiner（concat/sum/avg）组合字段，再用 history_pooling（mean/weighted_mean/attention/none）归约历史位置。多字段历史和标量 query 均先组合，attention 为固定尺度点积、masked softmax、加权和。DNN/DeepFM 共用，none 保留序列并拒绝直接送入 DNN/FM。
公共 FeaturePlan 负责对齐检查与同步 tail 截断，模型负责组合、聚合和维度验证；不引入公共层反向模型依赖。旧 pooling 配置需要显式迁移，树模型不支持神经聚合。尚未部署新训练镜像。
