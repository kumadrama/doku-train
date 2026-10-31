# Spec 031 生产迁移操作与门槛

下列为迁移顺序；2026-11-01 已开始执行，当前证据见 verification.md。PR #30 已部署 TF daily revision 6，保持现有 Endpoint 模型不变。每一阶段单独记录输出，禁止把版本注册成功当作激活成功。

1. 发布 doku-tools 的 task-target 扩展并验证隔离目标：旧字段保留、部分失败回退、并发漂移拒绝。保存当前 target 全量 JSON、TD/镜像、两个 Lambda code hash、binding/recipe、Endpoint config 与 Terraform state。
2. 在维护发布冻结下应用无资源删除的 IaC 交接。确认 removed destroy=false 保留 revision 6、target 仍指向 6、legacy_command 与原参数等值；Lambda code hash 保持，添加 doku:code-owner=dkd。恢复规则 DISABLED，runtime policy 为 null。新增 deferred 权限不依赖策略启用。生成 bindings 前必须实际 apply 完成。
3. 从两个独立 stack 执行 `terraform output -json`。`scripts/gen-runtime-release-binding.py` 合并当前 binding，并输出根 stack 所需最小发布权限 tfvars；审查、应用权限，再发布生成的环境 binding。部署角色仅能更新指定函数代码和指定规则 target，不能变更函数配置。
4. 构建兼容 pipeline/train 镜像。用 train 的 `scripts/prepare_tf_release.py`（base binding、recipe、已应用的 tf_runtime_binding 输出和镜像 digest）生成冻结的 binding/release/task definition；保留已部署模型配方、推理镜像。上传不可变 JSON 后更新 tf-pipeline 发布声明，经 dkd 注册并激活新 task。
5. 此时 target 仍保留 legacy_command，确认新镜像兼容旧完整参数。随后单独将 IaC legacy_command 设为 null，回读 target 仅传 ready 参数，新 task 从环境读取新的 config/binding。验证合成 ready 的输入解析，不发起真实训练或 Endpoint 更新。
6. 用同一 release/config 打包 tf-ready（`package_tf_ready.py --release ... --config ... --output ...`），两次摘要一致；上传真实制品，填写 lambda.yml 的 key/object_version，经 dkd lambda 发布。它开始持久化带配方摘要的 ready intent。必须在第 5 步完成后才切此生产者。旧 28+1 settings 暂留作旧版本回退，正常新代码不使用它们。
7. 分别构建/发布 model-monitoring Lambda，先保持 runtime_policy=null 验证现有采集。根监控规则、notifier 均无需重新发布。记下 CloudWatch delivery ACK、catalog 及原 Grafana UID。
8. 在既有 AppConfig application/environment 注册业务仓 Draft 4 schema profiles；以业务仓 defaults 为初值，通过受权 AppConfig 控制台创建 hosted version、发布并等 COMPLETE。记录 profile ID、配置版本、deployment ID；不得由 Terraform 每次改写内容。
9. 分别接入 policy binding（监控由环境装配，TF 用新不可变 binding/task release，并同批构建匹配 ready release）。先确认开关均为等值行为、recovery=false；再开启恢复，max_items=1。新旧 binding 的 policy 身份不同会拒绝旧 deferred 自动回放；切换前必须排空旧 pending 或按冻结证据人工迁移，不能静默换配方/策略身份。
10. 受控验证暂停训练不创建新 Job；暂停发布保存既有制品、恢复不重训；关闭监控 dashboard_sync 仍采集；collection 暂停先持久化再 ACK，恢复不重复指标。真实训练/Endpoint 操作须纳入该次明确授权。
11. 至少观察一次自然日更、补偿、告警后，清理 legacy_command/旧窗口/旧扫描预算配置。再次只读 plan 不回写应用代码/target。保留旧不可变制品和版本，不清空 journal/cursor。

回滚：使用发布平台恢复已保存的 task/ZIP。若回退到不支持环境默认值的 revision 6，必须先暂停物理触发并恢复完整 legacy_command，再切旧 revision。旧程序不识别 AppConfig，业务暂停期间不能只依赖 AppConfig；先硬暂停，再回退，核查 pending 事件后恢复触发。遇到外部 target 漂移或无法证明旧 executor 已停止，停止自动接管，保留证据。

生产验收清单：同一 model_name、Endpoint 名/当前模型版本、Grafana UID；target 激活回读、Lambda code hash、AppConfig runtime 版本、28+1 配方、四任务指标、失败告警均有证据。生产迁移尚未完成时，不能勾选该清单。
