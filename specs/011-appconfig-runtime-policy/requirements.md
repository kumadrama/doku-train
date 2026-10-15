# 011 — 云端训练运行开关

Source: 用户确认 #7–9 评估方案，授权 doku-train / doku-iac 分别提 PR，禁止线上部署。

- AppConfig `doku-train/tree` 分别控制 training、publication、cleanup、redis。
- training=false 跳过新的 daily 任务；publication=false 保留训练/评估产物，不推进版本。
- redis=false 禁止新的发布/分发/回滚推进，不删除 Redis 绑定、不降级 S3-only。
- cleanup 独立控制，继续使用协调清理和 consumer/distribution readiness 保护。
- 训练的独立 timer 继续关闭，保留样本成功事件触发，禁止新增重复触发路径。
- 参数及故障 fail closed；只读失败不是业务关闭，不伪造成功指标。
