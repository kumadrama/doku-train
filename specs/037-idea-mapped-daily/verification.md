# 2026-11-13 生产验收

首发使用 SageMaker 作业 doku-exp-idea-mapped-08-r1-v05-w10-1113-a59331bd 的 epoch 7。等权 AUC 为 0.6874079315669585。两次 InvokeEndpoint 与本地 checkpoint 的最大绝对误差为 2.24e-8，Endpoint 为 doku-prod-idea-mapped。

首轮 doku-exp-idea-mapped-daily-20261112-c474ee88 在独立控制记录中进入 training，冻结日期为 11 月 2–12 日。两条 EventBridge 规则均已回读为 ENABLED，实际目标为 doku-prod-idea-mapped-daily:1。terraform apply 新增 10 项资源。

本地 pytest 运行 test_daily_publication.py、test_official.py 和 test_idea_daily_input.py，共 41 项通过。对密封的真实 08 训练产物重走标准导出和 TF Serving 封包入口已通过。此次证据覆盖首发、日更启用与首轮训练接管；首轮新模型发布结果以独立状态日志为准。

## 本地导出配置与日更复用

- `recommend_tf/train/tests/test_inference_contract.py::test_export_only_selected_task_preserves_training_heads`
  用 7 个训练头完成训练和导出，逐头确认 checkpoint 变量存在，加载 SavedModel 检查仅有
  `completion` 一个签名输出。非法、空和重复 task 列表另有参数化测试。
- `recommend_tf/train/tests/test_daily_publication.py::test_daily_reuses_training_and_exports_without_labels`
  验证历史配置的四头导出、checkpoint 数值一致和可恢复发布。
- `recommend_tf/train/tests/test_idea_daily_input.py` 验证窗口从首个成熟日增长至 30 个训练日、冻结后可重现原窗口、运行中的操作复用原 ready，以及缺少发布指针时等待。
- 使用本地密封 08 历史实验产物及原 binding 重新执行 export_model.execute，加载签名核对
  三个输出和单位均与原 binding 一致。验证仅在本地执行。

## mapped/v1 原地回刷及生产切换

- 上游 [PR 64](https://github.com/kumadrama/doku-rec-offline/pull/64) 和[生产回刷作业](https://github.com/kumadrama/doku-rec-offline/actions/runs/43429615668)。对 2026-11-02 至 11-12 的 11 个 UTC 日逐行回读，共 698,823 行。用原始 map 的有序 keys 对照两列 list<int64>（包括 null 和空列表），其余列逐列调用 Arrow equals。核对新旧 Manifest 摘要和各分片行数。
- 用实际发布指针执行 freeze 并通过 PublishedDataset 打开冻结窗口，得到 10 天 train、1 天 validation。训练角色对实际不可变分片的 GetObject 和对应 Bucket Key 的 KMS Decrypt 策略模拟均为 allowed。
- ECS 实例任务 6c52f161eacc40fe9ddb46aea12c164f 从公共入口启动，退出码为 0。Terraform 两次 apply 合计新增一条训练读权限、更新两个目标。通过 DescribeRule 和 ListTargetsByRule 回读，两个触发器均已启用并指向 doku-prod-idea-mapped-daily:2。DescribeEndpoint 返回 InService。确认发布进度为 2026-11-12，active 为空。
- 本次 make check 通过。公共训练测试 329 项通过、3 项跳过，TF 训练测试 224 项通过，推理测试 8 项通过。启用后独立 stack 再次执行 terraform plan -detailed-exitcode，返回 0，无待应用差异。
