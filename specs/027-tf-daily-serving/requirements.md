# Spec 027：TF 每日训练与 SageMaker 发布需求

状态：2026-10-30 用户确认本文与 design 并授权本地代码实施；云部署与生产操作未授权。

Source: 2026-10-30 用户在 grilling 会话中确认 Q1–Q12、两个训练入口复用结构，并要求整理 requirements/design，先不写代码。

## 1. 目标与边界

在独立 `recommend_tf` 路径为已选 TF 模型建立每日训练、导出、S3 发布及固定 SageMaker Endpoint 更新能力。保留现有 `recommend` 树模型日更和离线 TF 实验行为。

首次发布候选为 `doku-exp-f024-ln-lr001-20261029-134714`。本地保存的实验记录显示已选 epoch 21，原实验采用 23 天训练、1 天 validation、1 天 test。该记录不是远端制品完整性证明；首次导出必须核验成功状态、选模记录和实际产物。不要求为首次发布重训 30+1。

- `doku-train`：训练、导出、版本化制品合同、每日编排、S3 发布、Endpoint 更新、部署确认及回滚。
- 独立 serving 仓库：请求解析、特征编码执行、模型加载、响应封装及推理镜像；复用 TensorFlow Serving，不自研 C++ 推理引擎。
- `doku-iac`：IAM、网络、ECS 编排任务、上游触发连接和环境资源装配。
- `doku-monitoring`：训练/部署失败、模型数据新鲜度、Endpoint 错误和延迟监控。
- 消费方：特征获取、业务实验分流、排序与调用失败降级。本次不修改 `doku-mono`。

本 Spec 的已确认目标在 TF 范围内扩展既有“仅训练产物、不切换线上版本”的限制；不扩展为本仓承担 serving 实现。代码实施和云操作仍需后续授权。

## 2. 配置与日更训练

R1. 增加 `online_config`，字段表示自然日数量，必须为正整数：

```yaml
online_config:
  training_date: 30
  validation_date: 1
```

该配置仅日更入口使用；普通离线实验继续使用显式日期。不得将窗口值硬编码进训练框架，也不得向 dataset 恢复 daily_prefix 等已移除字段。

R2. 截止日期 D 来自已发布上游数据及其日分区合同，不由执行机器当前时间猜测。以本次配置解析 train=[D−30,D−1]，validation=[D]；自然日时区沿用上游合同，不擅自改为客户端时区。31 天必须齐全且满足上游成熟性证据；缺日不提交训练、不缩短窗口，记录原因并告警，补齐后允许重试。

R3. 日更每次从头训练，固定配置配方，不自动搜参、不默认续训昨天模型。日期、输入摘要、run_id 随任务变化；配置、源码及镜像版本显式固定，升级需明确发布变更。不得因新增早停能力而自动改变指定实验配方。

R4. `daily_train_model.py` 只执行 train + validation。复用现有数据准备和 `training_pipeline.run()`：train-only 词表拟合、训练、每轮 validation、checkpoint 及配置选模。不要求独立 test 日，不将 validation 指标称为 test 指标。无法选模时记录状态并保留训练结果，但不发布未经选定模型。

R5. 离线 `experiment.py` → `train_rerank.py` 保持原行为。允许小范围提取公共数据准备函数；禁止复制训练循环或另建模型实现。相同冻结输入、配方和环境下，两入口的训练和选模结果应一致。

## 3. 数据 ready 与编排

R6. 沿用上游样本发布成功触发一次性 ECS 编排任务，不新增独立常驻训练 timer。树模型与 TF 为独立下游流程，不要求树模型成功才能训练 TF。触发必须携带或可解析精确日期、不可变 Manifest 引用及摘要；单个 Parquet 写入不是 ready。

R7. 编排验证窗口交付证据并冻结实际输入后提交 SageMaker CPU Training Job；训练入口再次验证冻结输入。ECS 负责跟踪、导出、发布和更新 Endpoint，不在 Endpoint 中训练。按明确身份和环境执行，无 PROD 默认值。

R8. 重复事件、任务重启和 API 超时不得引起重复训练或重复发布。训练、导出、上传、部署分阶段持久记录；已完成训练可单独重试导出或部署。旧数据版本不得覆盖较新版本，显式回滚除外。

## 4. 导出与 serving 合同

R9. 新增独立 `export_model.py`，只读取成功 run、真实选模记录及配套状态，交付 SavedModel、特征状态、特征配置、输入输出签名、来源 lineage 和校验和。导出失败不修改训练成功状态。不得把训练 checkpoint 或 SageMaker 训练归档直接宣称为可部署推理包。

R10. 模型相关编码由 serving 侧执行，与模型共同版本化；能安全入图的变换可由导出包承载，不能声称全部变换已验证可入图。本仓不新增 `serving/`、HTTP server 或线上特征读取逻辑。与独立仓库仅通过版本化制品和镜像合同协作，不跨仓 import 内部实现。

R11. 推理请求为无标签、具名且有类型的批量特征，不继承训练样本过滤条件。响应提供四个主任务 completion、fast_swipe、immersive_click、watch_seconds 的结果及实际模型版本，保持候选顺序；辅助任务不默认对外暴露。精确字段、维度、单位和变换随签名明确，不能靠调用方猜测。

R12. 验证选定 checkpoint 与导出模型最终输出一致；覆盖默认值、OOV、空历史、截断、padding 和批量输入。冻结的词表等状态不得在 serving 重新拟合。真实用户样本不得进入仓库、推理包或测试日志。

## 5. 发布、切换与恢复

R13. 每日自动发布不可变 S3 版本，使用独立 serving 仓库提供的明确镜像版本创建 SageMaker Model/EndpointConfig，并更新名称固定的 TF Endpoint。独立 TF Endpoint 与原树模型服务并存；消费方管理树/TF 业务实验。首次指定模型部署完成后按日更新 TF，不等待业务实验结束。

R14. 新实例就绪后一次性切换，不设置小流量 canary；部署后观察运行状态，异常回滚。保留旧版本及镜像/配置回滚材料。部署观察参数需显式配置，不能把未配置告警宣称为已具备自动回滚。

R15. 数据、训练、导出或部署失败时保留旧服务。分别记录“上传完成”“部署完成”“实际在线版本”；仅 S3 上传成功不代表上线。服务实际版本应以部署核验与响应为证据。

R16. 不新增隐式 AUC 下限或效果下降门禁；正确性、完整性和执行错误仍须阻止发布。容量指标目前未给定，先完成正确性与吞吐测试，记录机器、批量大小、并发、延迟分位数和峰值 RSS，不提前声明生产 SLA。

## 6. 验收与部署前依赖

本地验收应覆盖：日期解析和缺日；入口隔离与训练一致性；无 test 日更；产物完整性和预测一致性；编排幂等/恢复；Endpoint 超时对账、失败和回滚；并发发布与清理保护。实现后在 verification.md 给出实际命令和结果，本轮不生成虚构验证证据。

待外部确认项不阻塞通用本地实现，但阻塞对应联调/生产验收：

- [NEEDS CLARIFICATION: serving 仓库地址、负责人、制品目录/签名合同及 CPU 推理镜像版本；不得假设 TF 2.11 与任意现有镜像兼容。]
- [NEEDS CLARIFICATION: 环境、区域、角色、网络、S3/KMS、Endpoint 名称、实例规格及实例数量。]
- [NEEDS CLARIFICATION: 消费方候选批量、峰值请求量和延迟预算；由压测与消费方共同确认容量。]
- [NEEDS CLARIFICATION: 部署观察时长、执行健康告警与回滚绑定；清理保留期和恢复时限。]

## 7. 参考

- [设计](design.md)、[任务](tasks.md)、[运行合同](operations.md)、[验证](verification.md)
- [Spec 024](../024-finder-strict-migration/design.md)：新版 TF 训练边界。
- [Spec 009](../009-daily-tree-model-publication/design.md)：仅参考训练/导出/发布分层，不继承树模型 test、Redis 或质量阈值语义。
- [AWS Pipeline 调度](https://docs.aws.amazon.com/sagemaker/latest/dg/pipeline-eventbridge.html)
- [AWS 蓝绿部署](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-guardrails-blue-green.html)
