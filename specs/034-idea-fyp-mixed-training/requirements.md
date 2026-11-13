# Spec 034：TF Idea / FYP 配置驱动混训

状态：需求已整体确认；用户授权设计后直接实施；本地实现及验证见 verification.md。
Source: 2026-11-04 本会话 Q1–Q19 用户决策；原始需求为支持新 Idea 数据、双流混训、shuffle、FYP 1/50 采样与缺特征补 0。

## 范围与输入

- 仅 TF 公共训练输入与执行路径；不扩展 Torch/树模型。初始实施不包含云操作；2026-11-05 用户另行授权合并部署、两天历史数据重产及 SageMaker smoke，随后明确授权 Idea 只读 IAM 和稳定哈希重跑；不含模型/Endpoint 发布。
- 消费 doku-offline 的 FYP 产物与 doku-rec-offline/idea 的 request-training **每日投影产物**，不直接消费原始小时快照，不 import 上游实现。
- 每日出口已提供 doku-train-dataset-manifest/v1、扁平 Parquet 与版本化特征/标签合同；复用基础 reader，适配多源与每日解析。不得将缺少 FYP 特有元数据等同于 Idea 合同损坏。
- data_sources 逐来源声明 name、稳定 source_id、format 和不可变 manifest_uri。来源 ID 不依赖数组位置，读取层注入 datasvr_source；不自动作为模型特征。
- 支持配置单来源或多来源；每次运行冻结实际使用的 Manifest、合同、摘要、来源映射及跳过记录。恢复复用冻结输入，不重新发现日更指针。

## 采样、字段和混合

- item_filter 为总过滤，适用于进入后续训练/评估的数据。FYP 对用户标识做稳定哈希后取模，保留 stable_hash(u_account_id) % 50 == 1；Idea 全量。实际字段须从源合同核实映射，不引入 uin 私有命名，不改为曝光采样。2026-11-05 用户明确授权此调整，取代此前直接取模、不使用 hash 的限制。
- 两来源特征取配置所需字段，对方不存在的字段通过显式 feature_default_values 补 0；保持 dtype/shape 合同。标签占位不等价于允许参与 loss。
- 同名字段语义与类型须核对，不盲目合并。来源间历史等结构化特征按已声明合同对齐。
- 采样后自然混合，不强制来源配比，不重复 Idea。每 epoch 每条保留样本使用一次。
- 允许在整个 train 日期范围跨天跨源 shuffle，不跨 train/validation/test 时间边界。
- 首次批量读取、采样、特征准备并缓存；每 epoch 按来源剩余样本数比例随机交错样本引用，再通过公共有界 shuffle buffer，最后组 batch。固定 seed、epoch 顺序可复现；实现后观测耗时，必要时再优化。

## 目标、loss 与评估

- Idea 主目标仅 read_more、watch_now、vote。
- FYP 使用现有 TF 精排四个主目标作为 drama 辅助目标，各目标权重为对应原权重的 0.1 倍；实际目标名称在设计中依据基线配置列明。
- tasks.loss_weight 支持表达式与按来源门控：Idea 样本 drama 权重 0，FYP 样本 Idea 权重 0。
- 混训复用 FYP 任务 loss。默认每头的加权单样本 loss 之和除以实际 batch 行数。显式配置 loss_by_weight 时按非零权重行数归一化。
- 接受 Idea 每日 idea-visible-ten-minutes-observed/v1 标签口径，包括每日投影将未观测标签填 0。vote 标签包含成功非重复的 Vote/Notify 操作，watch_now 独立。
- 上游 placement 原值透传每日 Parquet/特征合同，不新增标签。用户接受本期以卡片类型近似按钮资格：WATCH_NOW 参与 watch_now，REGULAR 参与 vote（含 Notify）。缺失 placement 只参与 read_more，不参与另外两个 Idea 目标。
- placement 不保证等价于客户端最终按钮类型；本期不采集 primary_action，不使用事后状态重建按钮资格。
- eval_tasks 定义指标与各自 filter，validation/test 共用；不新增 test_task 或全局 validation_filter。
- validation/test 指标仅统计 Idea；watch_now/vote 指标分别采用同样 placement 资格条件。
- loss_weight=0 不隐式过滤指标。每个 metric 独立过滤并统计分母、有效行数；无可计算指标记录 null 与原因。

## 静默跳过与空输入

Source: Q18/Q19 用户明确同意，取代本 Spec 范围内来源异常必须终止整次训练的旧要求。

- 声明来源未产出、声明日期无数据、读取/权限错误、文件丢失、校验失败或训练过滤后为空时，静默跳过该来源，使用其他可用来源继续；不报错、不告警，原因仅保留运行记录。
- 跳过来源的范围与处理顺序由设计明确，保证校验失败的数据不参与训练，不混入来源的半成品数据。
- 两来源 train 都不可用：静默跳过本次训练，不生成成功训练产物。
- Idea validation 为空：继续训练，跳过依赖验证指标的早停与选优，保留最终 checkpoint，不声称选出最佳模型，不用 FYP 替代。
- Idea test 为空：跳过该评估，指标记录 null 与原因。
- 上述静默策略针对数据来源不可用，不将模型执行错误、配置错误等任意故障伪装为成功。

## 验收要求

1. 合成双来源数据验证单流/混训、稳定 source_id、用户取模采样与跨源缺字段默认值。
2. 验证来源/placement loss 门控、批内归一化与梯度；验证三个 Idea 目标及 drama 辅助目标。
3. 验证 metric 独立过滤、各自计数/AUC、空样本 null，不混入 FYP 指标。
4. 固定种子重现 epoch 顺序，无丢样本/重复样本，无跨 split 泄漏，有界读取与 shuffle；记录吞吐和内存证据。
5. 验证来源缺失/读取失败/校验失败/过滤为空的静默跳过、全部为空、验证集为空、测试集为空。
6. 验证输入冻结、缓存身份和 resume 不重新解析可变输入；失败来源无部分数据泄漏。
7. 同步架构设计、任务与可执行验证记录；训练侧实施须在需求及设计确认后进行。

## 已完成的独立上游改动

本会话已按用户明确指令在 doku-rec-offline 本地允许 column.placement 投影 v1/v2，加入每日特征配置并更新每日说明；原配置已有未提交改动被保留。
验证：test_request_consumer.py 9 项通过，git diff --check 通过。未部署、未重产数据。

## 2026-11-05 单入口兼容补充

Source: 用户要求原 manifest_uri 能力兼容到多源。每来源 manifest_uri 支持精确 Manifest 或显式 `{date}` 模板。原 TF 单源 reader 是精确读取，模板解析为多源新增适配，不声称旧 reader 原生发现每日产物。解析日期来自已有 split 配置；支持 FYP/Idea _SUCCESS 的 manifest_uri+manifest_sha256 公共合同，不要求 Idea 补 FYP 专有字段。解析及加载期间验证指针未变化、摘要一致、时间范围正确；冻结 URI 列表与指针原文，重放不解析可变入口。
