# 验证

Source: 2026-10-23 本地执行证据及用户授权的四任务 SageMaker 重训、只读导出证据。

- `UV_PROJECT_ENVIRONMENT=/Users/yujiatang/doku/doku-train/.venv make check`：通过。锁文件、format、lint、mypy 全部通过；**233 passed, 3 skipped**，128.28 秒。
- `git diff --check`：通过。
- `tests/unit/test_tree_buckets.py`：float32 精度、边界相等右桶、首尾区间、空阈值、null/default、非法表达式/阈值、缺失特征报错、未见桶完整索引、状态恢复、相对路径、同数据切分校验。
- `tests/smoke/test_tree_binning.py`：真实合成树→导出→DeepFM→test；树 YAML 删除后 test 正常；同一冻结 Manifest；中断后不重训树/不读 YAML；恢复与连续运行模型权重一致。
- `tests/unit/test_sagemaker_platform.py::test_tree_yaml_is_frozen_in_submission_and_run`：树 YAML 随提交内容冻结，删除本机配置并改原 Manifest 后仍以冻结输入训练，归档及 checksum 通过。
- 正式 `tools.checkpoint.export_tree_buckets` 对已完成 `tree-rawstats-full-20261022` checkpoint 做只读导出：84 个数值输入均有条目。67 个指定统计特征合计 **9,322 桶**，最大 **254 桶**；与原试导出逐特征桶数一致。`u_comment_count_28d_fyp` 无分裂，导出 []，一个桶。
- 实验导出文件与结果仅保留原工作区本地 `runs/attention-comparison-20261022/tree-rawstats/`，不随源码提交。

集成中发现同进程加载 XGBoost 后初始化 PyTorch 会导致本机阻塞；最终实现将树训练和 checkpoint 阈值导出均放在独立子进程，参数数组执行并校验退出码。上述全套验证基于修复后实现。

- 四任务 `doku-exp-tree-four-task-binning-20261023` 已 Completed；下载产物校验通过后正式导出。67 个统计特征共 **8,378 桶**、最多 **248 桶**（`u_expose_count_14d_fyp`、`u_completion_count_7d_all`）；全部 84 个数值特征共 **9,510 桶**、最多 **253 桶**（`i_quality_score`）。云训练使用已有镜像，验证的是实际 checkpoint 导出，不代表新的自动预训练流程已部署到云端。实验配置、模型及统计文件仅保留本地 `runs/tree-four-task-binning-20261023/`。
