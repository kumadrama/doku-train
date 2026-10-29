# extended40-lr004 → Finder TF：配置 Review

状态：供用户 review，未提交云训练。实际读取的基线是 PyTorch `e1-extended40-lr004-20261027/prepared/package.json`，并非手抄配置。来源及配置摘要见本地 provenance.json。

## 保持不变

- 冻结 Manifest URI、训练 2026-10-01～23、验证 10-24、测试 10-25（UTC）。训练输入 2,978,900 行，验证 153,757 行，测试 162,222 行，均为过滤前 Manifest 行数。
- 40 epoch、batch size 2048、seed 42、学习率 0.004（固定，无新增学习率调度）。
- 七个任务及 label、loss_weight、指标配置、主干 128/64、任务塔 32、dropout keep probability 0.9。
- Embedding 名称、共享关系、维度、uniform(-0.01, 0.01) 范围；五个 FM 交互对。
- 特征表达式、视频曝光过滤、历史去重/排序/截断规则和选模配置。
- intra_op=4、inter_op=1、读取线程和资源预算保持基线配置。

## 必要差异

| 项目 | 原 PyTorch | 本次 Finder TF |
|---|---|---|
| 优化器 | Adam，耦合 L2=0.0001 | Finder AdamW，暂列 weight_decay=0.0001；dense 衰减变量，sparse 只衰减命中行；系数不再乘学习率 |
| Adam 参数 | PyTorch 默认 beta/epsilon | 显式 beta1=0.9、beta2=0.999、epsilon=1e-8；未单独启用 sparse optimizer，共用同一个优化器 |
| 混合输入组 | item_numeric 同时含连续值和类别 embedding | 拆成 item_numeric_scalar_0（quality/heat）和 item_numeric_embed_1（三个分桶），保持拼接顺序和宽度 |
| 表达式配置 | feature_list | apply_to_features + apply_to=[item]；表达式文本不变，在模型图中按 batch 执行 |
| FM 配置 | fm_type/features 外层结构 | 转为 Finder 的特征组配对列表，交互对不变 |
| 标签缺失 | watch_seconds 相关任务有 nullable mask | watch_duration_ms 默认 0，移除 mask；按原 label 和 loss_weight 计算，缺失观看时长会参与对应任务的 loss |
| loss 分母 | 原 Torch loss 实现 | 显式 loss_by_weight=false，Finder mean(weight*loss)，不增加有效样本 mask |
| 特征与词表 | 训练前保存最终编码数组 | 训练前保存基础输入，train-only 冻结词表；图内表达式和 lookup 每 batch 执行 |
| shuffle/padding | 全量索引 permutation / 预处理数组宽度 | 每 epoch seed+epoch 有界 shuffle，缓冲 8192 行；每 batch 按真实 length padding |
| 初始化与随机数 | PyTorch Linear 初始化、Torch RNG | Finder VarianceScaling/TF dense 默认初始化及 TF RNG；seed 相同不保证跨框架相同权重或轨迹 |
| 配置入口 | 原 YAML 统一入口 | 新 TF run 读取 config.json；config.yaml 为同内容可读副本；dataset_config 只保留读取字段，避免残留 Torch optimizer 造成歧义 |
| 训练产物 | Torch checkpoint | TF Saver checkpoint + feature-state；不兼容旧 Torch/TF checkpoint |

**重点审阅：优化器衰减系数。** 0.0001 仅保留原数值作为待 review 的实验值，不是 Adam L2 到 Finder AdamW 的等价换算。学习率 0.004 下，不能据此声称正则强度一致；没有为了模拟 PyTorch 而改动 Finder 实现。

## 文件和使用

实验文件按仓库规则只保留本地 runs/，不提交真实实验配置：

- config.json：新 TF 入口实际读取的配置。
- config.yaml：按 pipeline 顺序平铺的 review 视图；训练入口仍读取 config.json，命名空间还原后值完全一致。
- source-config.yaml：原提交配置备份。
- frozen-input/：原提交的 Manifest 与合同元数据，不含训练样本。
- provenance.json：来源、摘要、dtype 适配说明。
- prepare_config.py：可重复生成；不访问 AWS。
- verify_config.py / verification.json：完整配置的本地合成验证与结果。

从此 worktree 根目录执行本地验证：

```sh
PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 recommend_tf/.venv/bin/python runs/finder-e1-extended40-lr004-review/verify_config.py
```

实际训练入口（本次未执行；输出目录须不存在；云端仍需封装训练镜像/提交参数）：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python -m recommend_tf.train.run --config runs/finder-e1-extended40-lr004-review/config.json --output /opt/ml/work/runs/finder-e1-extended40-lr004
```

2026-10-29 更新：run 的 Manifest 入口已接入 train → 验证集选模 → 所选checkpoint独立test；cloud_train 负责正式云端启动及每epoch上传。

## 验证范围

使用完整网络、全部实际特征表达式、七任务、AdamW 和原 batch size；仅把本地 smoke 的 max_epochs 改为 2，并生成每个 split 12 条合成数据。覆盖空/重复历史、观看时长缺失、两轮训练、checkpoint 恢复一致性、独立 test。40 epoch review 配置不被 smoke 改写。

本地结果不证明真实数据容量、吞吐、模型效果或与 PyTorch 数值等价。未执行 S3 数据读取、上传、构建或云训练。

Source: 用户选择方案②并要求兼容 Finder；原提交 package.json；Spec024 已确认设计；当前 data_utils.py、graph_helper.py、rerank_model.py、training_pipeline.py。

验收结果（2026-10-29）：上述 verify_config.py 命令通过；2 epoch、7任务、恢复 validation loss rtol=1e-6、独立 test 全部通过。另核对 config.yaml 还原命名空间后与 config.json 内容完全一致，正式配置仍为40 epoch / batch2048 / lr0.004；git diff --check 通过。

格式调整：特征处理超参 → features → dtypes/defaults → feature_transformations → input_features → 模型 → 训练。列表统一使用 []；format_config.py 校验还原后与运行时 JSON 相等。
