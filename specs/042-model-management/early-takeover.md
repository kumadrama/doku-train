# 提前接管：旧执行版本隔离

Source: 2026-11-26 用户明确要求“提前接管”。本补充取代必须等待完整 24 小时的对应要求，但不允许省略旧执行路径隔离与线上核验。

## 当前状态

现有控制器只支持投递窗口排空。Idea 与 FYP LHUC 的既有 plan 和 ownership 日志必须保留，不直接编辑 drain_after，也不另建 plan 绕过 CAS。官方 base 已按用户要求停止，不能恢复日更；其余三个 FYP 模型保持现有日更。

## 目标与验收条件

提前接管采用显式操作和可恢复隔离日志，完成以下条件后才允许同一原 plan 提前激活：

1. 根据旧任务定义的冻结 config/binding 确认模型身份，盘点所有可能执行该模型的旧 revision。无法识别的版本或不完整分页不得视为隔离完成。
2. 暂停该模型全部旧入口，保留其他模型的触发目标。仅停用目标模型的旧任务定义版本，不停用共享任务族内其他模型版本。
3. 记录隔离开始时间、revision 集合与摘要；等待至少 600 秒传播窗口，再回读所有目标 revision 均 INACTIVE。AWS 说明 deregistration 限制可能最多需要 10 分钟生效；既有 task/service 不受 deregistration 影响，因此仅 INACTIVE 不足以证明旧执行路径已消失。
4. 核验没有引用旧 revision 的 service；同时扫描 desiredStatus=RUNNING 和 STOPPED 的任务，逐个确认实际 lastStatus=STOPPED。特别防止 desiredStatus 已 STOPPED 但容器仍在运行的竞态。已有旧 SageMaker 训练、发布状态和未消费请求也必须解决，不能伪造成功或丢弃请求。
5. 切换前重新核验入口、隔离集、旧 journal、EndpointConfig 与原计划证据；有变化则 fail closed。沿用原 registry 发布与 ownership CAS，直到 managed；不更新 Endpoint 权重或调用方配置。
6. 隔离后原 revision 不能直接重新激活，abort 不能简单重新开启原入口；必须明确报告需要正常发布新的 legacy revision 才能恢复旧流程。

Source: [AWS DeregisterTaskDefinition API](https://docs.aws.amazon.com/AmazonECS/latest/APIReference/API_DeregisterTaskDefinition.html)。600 秒是 AWS 文档描述的传播上限，不是可由调用方任意缩短的训练参数。

## 实施状态

- 已核验当前控制器排空实现及 AWS 官方语义。
- 已创建独立实现分支；AWS SSO 已恢复，云端旧版本盘点已完成。LHUC 有 :35–:40，Idea 有 :2–:9；另有三个无标准配置字段的历史版本需要先核对无引用后隔离。
- 未停用任何任务定义，未修改接管日志或部署新的控制器；线上仍遵循既有排空流程。
- API 新增 adopt_fence，CLI 新增 fence-adoption；服务端记录观察时间和 600 秒窗口。隔离操作由获授权操作员执行，控制器只增加 ECS inventory/service 只读权限。合成合同与既有流程测试通过后，经 DKD 发布，先验证未隔离会被拒绝，再隔离和接管；既有 DTE 端到端证据不替代本次生产隔离验证。
- 本地已修复旧执行器空闲检查：同时查询 desired RUNNING/STOPPED，并验证实际退出。`PYTHONPATH=. /Users/yujiatang/doku/doku-train/recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests/test_model_management.py -q -k adoption` 为 6 passed；覆盖正在退出任务不可提前接管、STOPPED 分页和实际退出后放行。修改文件 Ruff 与 `git diff --check` 通过；尚未部署。

本轮本地管理测试：33 passed，包含拒绝 ACTIVE 目标/未知 revision、服务引用、分页扫描、传播窗口及原计划保持不变；IaC 模块 4 passed，CLI 6 passed。云端尚未部署此变更。
