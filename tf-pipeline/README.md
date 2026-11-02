# TF 每日编排发布单元

独立于根 deploy/prod 的树训练任务。当前 release.json 记录 PR #30 已验收的 bootstrap；Spec 031 实施不代表这些镜像已包含新代码。
发布前从审查过的源码构建新的 pipeline/train digest，生成不可变 binding，更新 release.json 和 task-definition.yml 中的四个 DOKU 输入坐标。
训练配方和推理镜像保持冻结；禁止使用 latest。服务 YAML 供 dkd 的 --path tf-pipeline 入口使用。

Task 发布的 task-target.yml 依赖 doku-tools 的受信 ecs_task_targets binding 和激活实现。
先确认平台上线、旧任务结束、保存旧 target；再执行 IaC 无删除交接，最后发布任务。
交接期间保留完整 legacy_command；兼容 task 激活后单独切换到只传 region/ready 参数。此后config/binding 来自 task 环境，避免 Terraform 回写应用版本。
发布必须核对真实 target revision，不能把 RegisterTaskDefinition 当成已激活。

本地使用 `scripts/prepare_tf_release.py` 生成 release/binding/task-definition，禁止手工拼摘要；具体迁移顺序见 [Spec 031](../specs/031-runtime-release-ownership/operations.md)。
