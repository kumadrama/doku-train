# Spec 028 实施任务

Source: 用户确认 requirements/design 并授权代码实施；随后明确最小推理先放 doku-train 的 recommend_tf 下，不放 train。

- [x] T1 历史 run 审核工具：绑定真实任务与原始配置，校验选模及产物，export/parity 成功后提交 evidence。
- [x] T2 ECS 启动适配与独立编排镜像上下文；固定文件摘要与真实执行者身份。
- [x] T3 recommend_tf/inference 最小加载/特征适配/响应/HTTP 桥接及 TensorFlow Serving 镜像。
- [x] T4 发布器传递显式推理预算；补充原始历史伴随字段导出。
- [x] T5 doku-iac 新增独立权限、ECS/上游事件触发、执行告警与绑定生成模块；不改既有未提交环境文件。
- [x] T6 本地合同/数值/容器/IaC 测试与 CI 接入，执行证据见 verification.md。
- [x] 原生 amd64 编排镜像运行验收：CodeBuild SUCCEEDED，真实 ECS exporter/发布运行通过。
- [x] 外部装配：实际环境绑定、角色、实例、部署告警与上游 ECS 成功→ready→日更 ECS 事件接线已部署；发布心跳告警已启用。数据截止日专用监控仍属 monitoring 后续工作。
- [x] 云验收阶段①②③：真实指定 run 核验/导出、镜像上传、首次 Endpoint 实际请求与数值一致性通过。
- [x] 云验收阶段④⑤：28+1 训练完成，补齐推理角色权限后复用原模型发布；重复执行、编排恢复、实际回滚和恢复全部通过，两个自动触发规则已启用。未重新训练。

剩余事项属于云端集成与验收，不以本地代码/合成测试替代。首次 Endpoint 和编排中断恢复已实际通过；日更训练与后续验收执行中。

Source: 2026-10-30 用户最新指示：第④⑤步继续实施，窗口改为 28 天训练 + 1 天 validation，取代此前等待 31 天的安排。

2026-10-31 最终验收通过；以 verification.md 的最终记录与 final-acceptance-20261031.json 为准。
