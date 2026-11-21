# TF 每日编排发布单元

独立于根 deploy/prod 的树训练任务。当前 release.json 记录 PR #30 已验收的 bootstrap；Spec 031 实施不代表这些镜像已包含新代码。
发布前从审查过的源码构建新的 pipeline/train digest，生成不可变 binding，更新 release.json 和 task-definition.yml 中的四个 DOKU 输入坐标。
训练配方和推理镜像保持冻结；禁止使用 latest。服务 YAML 供 dkd 的 --path tf-pipeline 入口使用。

Task 发布的 task-target.yml 依赖 doku-tools 的受信 ecs_task_targets binding 和激活实现。
先确认平台上线、旧任务结束、保存旧 target；再执行 IaC 无删除交接，最后发布任务。
交接期间保留完整 legacy_command；兼容 task 激活后单独切换到只传 region/ready 参数。此后config/binding 来自 task 环境，避免 Terraform 回写应用版本。
发布必须核对真实 target revision，不能把 RegisterTaskDefinition 当成已激活。

本地使用 `scripts/prepare_tf_release.py` 生成 release/binding/task-definition，禁止手工拼摘要；同时传入已应用的 model_monitoring 输出（`--monitoring-outputs`），见[任务族标准入口](../recommend_tf/README.md#tf-日更任务部署)。具体迁移顺序见 [Spec 031](../specs/031-runtime-release-ownership/operations.md)。

Discovery 日更联动沿用该任务族。部署前应用现有 worker 的 Discovery profile 专用发布权限，将 IaC 的 `discovery_online_release_binding` 输出通过 `scripts/prepare_tf_daily_deployment.py --online-release-outputs` 合并到现有 binding，再按原发布流程上传冻结文档、更新任务四个输入坐标并激活新 worker。训练配置、推理镜像、endpoint 和原任务族保持原值；只追加 `discovery_appconfig` 资源标识。不能只部署新代码却遗漏 binding。

共享日更任务导出 `serving_with_model_version` 和 `serving_default`。启用固定 Endpoint 前先发布支持该签名的在线引擎。日更通过原有精确版本探测和固定 Endpoint 身份探测后，在各业务 profile 内首次启用固定地址。之后合同兼容的日更复用配置。实验模型通过各自的日更绑定执行同一合同。详见 [Spec 040](../specs/040-discovery-fyp-daily-publication/design.md)。
