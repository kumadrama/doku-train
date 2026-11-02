# Spec 031 实施任务

- [x] 先部署 PR #30 TF 修复：pipeline/train 不可变镜像，ECS revision 6，原 Endpoint 模型保持。
- [x] doku-tools：受信 task target 解析、同 PSM 发布锁、注册后激活、漂移检查、回退和版本记录。
- [x] doku-train：AppConfig 准入、持久 ready/执行意图、有界恢复、配方核验、独立 pipeline/ready 发布单元和本地制品生成。
- [x] doku-monitoring：独立策略/打包、暂停持久转存、统一事件预算、分页及 journal 内事件恢复。
- [x] doku-iac：TD 无删除脱管、精确字段忽略、Lambda owner 标记、最小权限、profiles、恢复调度和受信绑定生成。
- [x] 本地测试、重复构建及只读 TF 迁移 plan。
- [ ] 按 operations.md 发布平台并迁移 Spec 031 生产所有权。
- [ ] AppConfig 内容发布、受控暂停恢复及一次自然日更验收。

后两项属于下一次生产迁移，不以本地测试代替。原工作区 Spec 029 内容未修改。
