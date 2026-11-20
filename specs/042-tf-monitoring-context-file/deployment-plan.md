# 本次生产恢复计划（待发布）

Source: 2026-11-20 现场只读 AWS 查询；用户确认的文件传递修复方案。尚未执行下述生产写入。

## 范围

- 首次接线只修复 `tf_rerank_ln_lr002_3feat_us_or_unknown`，任务族仍为 `doku-prod-tf-rerank-daily`，四个导出任务及选模策略不变。
- 保留当前 Endpoint 及 cutoff 2026-11-18 成功版本；等待恢复训练自然完成后走既有导出、探测与发布流程。
- 生产 worker 基底：`703844615824.dkr.ecr.us-east-2.amazonaws.com/doku-train/tf-pipeline@sha256:1f6639476d87ace7e56b308183bc6a49ac2435d7cbdad30b74c62f0ab2a3fbe8`。
- 生产训练基底：`703844615824.dkr.ecr.us-east-2.amazonaws.com/doku-train/doku-reco-train@sha256:5e9ef04832e66350684c086061ec4c5dd9c748e51df05dd2c28269e9ff20da94`。
- 原 operation：`565d9f282f8f3aedf4b34e015a911c141aeb1438ad1377ca067aaf5df92e4ffc`；原训练名：`doku-daily-565d9f282f8f3aedf4b34e015a911c141aeb1438`。

## 顺序与验收

1. 推送独立修复分支至已有 origin `github.com/kumadrama/doku-train`，通过 PR 审查与 CI。禁止直接更新 main。
2. 使用已有 `doku-train` CodeBuild 项目构建补丁镜像。以已验证基底为基础，只交付本次运行时代码差异；先核对基底对应文件与补丁前版本，差异不一致时停止并审查。镜像内计算真实训练源码摘要和锁摘要，验证 CPU TensorFlow 版本及上下文读写/摘要异常。不可直接替换整个 train 目录夹带其他差异。
3. 为后续日期准备不可变 binding：仅更新训练镜像及源码摘要。worker 更新编排镜像和新 binding 引用。沿用原 recipe 字节、角色、网络、资源与模型名。
4. 准备该模型 ready ZIP：沿用线上 ZIP，只替换 release binding 引用。通过原 tf_endpoint 发布入口生成并审查计划；预期仅该模型 worker target 和 ready Lambda 制品更新，不改 IAM/调度开关或其他模型。
5. 使用原 binding、原 operation 和已验证的新训练镜像运行 repair_monitoring_context 的只读计划。确认旧 ECS 已 STOPPED、同名训练仍不存在、冻结 config 摘要一致；候选请求仅改变传输字段和镜像来源。若任务已存在则停止该修复路径，按实际状态恢复。
6. CAS 应用 journal 修复并回读，确认 operation、原输入、run URI、job 名和 online/high_water 不变；保持原请求审计证据。更新后的 worker 复用原 deferred envelope，它仍解析原绑定，因此 operation 不变。
7. 验收 SageMaker 同名任务被接受，实际 Environment 和 channel 对应修复后的冻结请求，日志通过源码/配置/上下文摘要校验；训练后验证 COMMITTED、四项监控事件和既有 Endpoint 发布结果。任何导出/发布失败复用成功训练产物，不另建 retry-id 重训。

生产写入前重新读取状态，不把本计划中的时间点快照当作实时事实。
