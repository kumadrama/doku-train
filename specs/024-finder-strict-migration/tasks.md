# Spec 024 实施任务

Source: 用户批准按设计实施。以下为本地实施状态，不代表云训练上线。

- [x] 独立 recommend_tf/train 包、Python3.10/TF2.11 依赖与 uv.lock。
- [x] 来源清单；transform_features/dnn_layer/target_layer/scalar_features 与源 AST 对照一致。
- [x] 当前路径图表达式、scalar、embedding、attention、core、target及loss。
- [x] Finder Adam/AdamW本地后端，dense/稀疏衰减与padding梯度验证。
- [x] 基础输入分片、固定词表、shuffle/batch/padding。
- [x] Manifest/合同入口、合成配置映射、训练/验证/独立test、checkpoint/epoch恢复。
- [x] 图中间值、七任务训练、dropout恢复、损坏输入、无标签PREDICT测试。
- [x] 根 make check 接入独立 TF 验证，architecture及验证记录同步。
- [ ] 实际 e1 配置显式选择Finder优化器/衰减参数后，进行真实冻结切片耗时与容量验收。
- [x] 原生Linux amd64完整Arrow输入到训练验收：SageMaker合成任务11项通过，见cloud-verification.md。

真实e1性能不以合成测试替代。已按授权运行一次SageMaker合成验收；未部署线上模型、未提交或推送。


## 首次正式训练

- [x] 逐epoch JSON / TensorBoard 与完整入口实现。
- [x] S3 run-scoped 增量上传，排除输入分片/缓存，失败状态不声明成功。
- [x] 原生Linux同镜像验证及首次40 epoch任务启动；证据见 [正式训练](formal-training.md)。
- [ ] 首次40 epoch训练完成及真实数据结果验收。

## 一次性准备与词表拟合优化

- [x] 固定三日各32768原始行切片，保存原实现快照并拆分准备/拟合耗时。
- [x] 批量验证审计时间、列式普通字段准备、仅历史复用逐行整理。
- [x] 词表拟合列批读取、直接组装张量；保持跨分片 batch 边界。
- [x] 新增列式/逐行一致性、空历史跨片与尾批、过滤前错误、时间边界/null测试。
- [ ] 全量云端性能验收：本轮未启动任务或更新镜像。
- [x] 关闭profiler的三次对照：准备9.754→6.074秒、拟合1.826→1.158秒；真实切片基础输入/词表/训练及拟合张量一致性通过，22项测试通过，详见verification。
- [x] 用户追加授权后完成原生Linux同镜像22项验收，并独立提交准备/拟合优化版40 epoch；任务及摘要见formal-training。
- [ ] 第三次任务准备、词表拟合、40轮训练及test的云端实际耗时验收。

## 六组单因素搜参

- [x] 设计矩阵与五组已确定配置差异验证；详见hyperparameter-experiments。
- [x] Finder affine LN迁移、原类前向/梯度精确对照、完整训练/恢复；本地25项测试通过。
- [x] 用户确认纯Adam、取消衰减，并增加16 vCPU同资源baseline，共七个任务。
- [x] 同镜像Linux25项测试、七配置校验通过；七个16 vCPU/4线程独立任务提交成功。
- [ ] 七任务完整40轮/test结果与性能汇总。

## 2026-10-29 Adam L2（本地实施）

[已完成] 接入开关、Dense regs、Embedding 系数与表名筛选，禁止隐式双重衰减。
[已完成] 采集 pooling 前有效 lookup 输出，按位置累计；训练总 loss 接入两项正则。
[已完成] 单独保存逐 epoch batch 平均优化目标与正则，并写入 TensorBoard。
[已完成] 验证重复 ID、共享表、padding、系数、梯度、关闭配置一致性与训练恢复。
[已完成] 原生 Linux 合成验收38项通过；用户授权提交五组云实验，真实数据性能与质量结果待训练完成。

## 可复现性修复（2026-10-29）

[已实施] 统一 epoch 构图随机状态，并固定评估图随机状态。
[已实施] 在全部梯度计算完成与任何优化器写入之间建立跨优化器屏障。
[已完成] 图依赖、数学更新、独立进程训练/恢复一致性与完整TF测试；本地和原生Linux各41项通过。
[不在本次变更中] 不修改已有任务、不提交新的40轮训练、不绕过旧checkpoint源码身份校验。


## PR 合入验收（2026-10-29）

Source: 用户确认 opt15/prep15 一致，授权将当前 TF 迁移和两项耗时优化通过 PR 合入 main 并同步本地。

- [x] 隔离合并分支，只纳入当前 `recommend_tf/train`、Spec 024、文档与验证入口；不带入旧 TF 原型或无关生产配置。
- [x] 全量 15 轮 opt/prep 指标逐项一致，327 个 checkpoint 变量（含模型/优化器状态）每轮逐字节一致；冻结 Manifest/record/Feature State 一致。
- [x] 当前 TF 代码测试 41 passed；格式/lint/mypy 与构建脚本检查通过。
- [x] 完整根 `make check`：318 passed / 3 skipped，TF 41 passed；远端 PR CI 作为合并前最后检查。
- [ ] 经 PR CI 通过后合并并同步本地 main。

原始实验配置、真实数据、checkpoint 与完整报告继续只保留在本地 runs/，不随源码提交。上述为本次合入状态，前文按时间记录的待验收项不覆盖该结论。
