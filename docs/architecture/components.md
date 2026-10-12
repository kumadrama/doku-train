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
