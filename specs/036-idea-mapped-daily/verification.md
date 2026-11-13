# 2026-11-13 生产验收

首发使用 SageMaker 作业 doku-exp-idea-mapped-08-r1-v05-w10-1113-a59331bd 的 epoch 7。等权 AUC 为 0.6874079315669585。两次 InvokeEndpoint 与本地 checkpoint 的最大绝对误差为 2.24e-8，Endpoint 为 doku-prod-idea-mapped。

首轮 doku-exp-idea-mapped-daily-20261112-c474ee88 在独立控制记录中进入 training，冻结日期为 11 月 2–12 日。两条 EventBridge 规则均已回读为 ENABLED，实际目标为 doku-prod-idea-mapped-daily:1。terraform apply 新增 10 项资源。

本地 pytest 运行 test_daily_publication.py、test_official.py 和 test_idea_daily_input.py，共 41 项通过。对密封的真实 08 训练产物重走标准导出和 TF Serving 封包入口已通过。此次证据覆盖首发、日更启用与首轮训练接管；首轮新模型发布结果以独立状态日志为准。
