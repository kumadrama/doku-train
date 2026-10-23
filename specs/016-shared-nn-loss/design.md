# 设计：公共 loss 按 Finder 分层

状态：需求与设计已确认，实施及完整验证完成。

Source: finder-train/recommend/train/comm/nn_utils.py:get_loss；finder-models/recommend/train/models/rerank/rerank_model.py 的 STEP7 Build Losses。2026-10-23 本地源码核查。

## 当前状态

Doku 训练与验证 loss 已共用模型文件中的 task_loss；task_metrics 另有概率版 NumPy logloss。树评估也使用该指标函数。Finder 在 comm/nn_utils.py 定义 get_loss，由模型建图的训练/评估公共路径调用；logloss 使用 logits 交叉熵。

## 目标状态

1. comm/nn_utils.py 提供 get_loss，按任务 loss_type 分派。logloss 唯一底层实现使用 PyTorch binary_cross_entropy_with_logits；迁入现有其他任务 loss，模型文件只保留网络实现。按 Finder 公共入口组织，不建立另一套注册器或框架。
2. get_loss 支持逐样本结果及公共加权归约，以满足梯度训练、按样本累计评估和指标自身 mask。训练仍 mean(weight * loss)，任务间求和；验证 loss 仍对有效标签无权平均。不能直接照搬 Finder 默认训练权重到验证，否则单任务零权重头的评估语义会改变。
3. runner 的 train/evaluate 调用同一 get_loss。评估传递原始 logits 给指标层；logloss 指标按自己的标签与 mask 调用同一入口，不能无条件复用 validation loss 标量（配置允许不同的 metric_label/mask）。概率仅供 AUC 等概率指标使用。
4. 按用户补充，树模型将原始 margin 交给 rerank_tree/loss_utils.py 的独立 logloss。metric_values 接受可选 logloss_func，由树 runner 注入；公共层不 import 树模型，NN 默认调用公共 get_loss。树原生训练目标不变。
5. 删除 task_metrics 独立概率截断/对数公式，不保留静默概率回退。历史极端预测的 logloss 可能变化：旧指标将概率裁剪到 1e-15，新实现按真实 logits 稳定计算；这是统一实现的明确数值变化。

## 验证方案

- 公共入口对二分类普通/极端 logits 的值和梯度验证；检查权重与归约。
- 相同标签/mask 下 train loss、validation loss、metric logloss 一致；不同 mask/metric_label 保持分别计算。
- 零权重任务仍报告有效评估指标，空有效集仍返回 null 与原因。
- 树 margin 的 logloss 与神经网络公共实现一致；验证无二次 sigmoid。
- 运行相关单元测试及 make check，更新 architecture 与本 Spec 验证证据。

Source: 2026-10-23 用户明确确认本设计并要求实施及完整验证。

Source: 2026-10-23 用户补充确认：树 func 不迁入 comm，保留在模型目录，未来可删除 rerank_tree。
