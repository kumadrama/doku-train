# 系统概览

Source: 用户确认的Spec004与本轮CPU实现。

本仓负责不可变发布数据到本地训练产物，不负责样本生产、线上推理或基础设施。
DNN与Finder式DeepFM共用rerank插件，以fm_layers区分；XGBoost使用rerank_tree。
任务、日期、特征、默认值、log变换、网络和loss由schema与独立YAML声明。

首轮计划：4个DNN+4个DeepFM+1个树基线，验证集按task比较loss/AUC，回归看MSE。
报告之后再决定是否精搜一种神经模型；最终测试在选择冻结后显式执行。

当前状态：配置校验、合成fixture训练、搜索和报告实现已接入；真实v4成熟性证据已完成核查，
XGBoost与DNN各完成一次本机真实训练，详见结果（本地实验记录，不随源码发布）。
这不是28天容量验证或搜参后的结论。没有云训练、生产写入、模型发布或exporter实现。

2026-10-10：真实4+4+1验证集搜索已完成，8次神经训练与1次XGBoost结果复用；
详见搜参报告（本地实验记录，不随源码发布）。尚未评估测试集或启动精搜。

## 树模型每日发布：本仓实现与跨仓目标

Source: 2026-10-13 用户确认；[Spec 009 需求](../../specs/009-daily-tree-model-publication/requirements.md)、
[设计](../../specs/009-daily-tree-model-publication/design.md)。

当前状态：本仓已实现 Spec 009 的 file/S3 发布和清理入口；未部署云端，消费者协议尚待 mono 接入。
目标状态：上游训练投影成功后触发单次训练、独立测试、发布门禁、原生 XGBoost 导出与
S3 清单提交；本仓提供回滚及 30 天清理入口，iac 管基础设施，mono 管加载与 RPC。
model_type 和 model_name 均由配置声明；同名配置/合同变化允许发布，文件按日期与不可变
版本隔离。业务调用方可只指定名称/日期，但公共组件必须一次解析并传递同一版本给编码
和 predictor，不允许各自选 latest。保留当前版本和替换后至少 24 小时的旧版本。
新增门禁只影响发布，不推翻成功训练状态；旧文档的首阶段 exporter/本地输出边界由
Spec 009 在树模型日更范围内扩展。

Source: 2026-10-13 用户确认 S3 方案并授权实施。消费者后台下载与加载，在线请求只用内存中
已就绪的固定版本，不逐请求读 S3；发布成功不等于服务就绪，不引入 Redis 模型存储。


## 独立 Finder TensorFlow CPU 路径（Spec 024）

Source: [Spec 024 设计](../../specs/024-finder-strict-migration/design.md) 与 2026-10-29 用户验收及合入授权。

当前实现位于 `recommend_tf/train`，使用独立 Python 3.10 / TF 2.11 锁文件；训练入口、图内表达式/lookup/loss、固定 train-only 词表、按 epoch 恢复、验证选模与独立 test 均与 `recommend` 解耦。保留 Finder AdamW 的实际衰减公式，不承诺 Torch 参数同名即等价。输入读取和准备/词表拟合两项性能优化随迁移代码交付，执行证据见 Spec 024 verification。旧 TF 原型不属于本实现，推理导出及自动发布仍不在本次范围内。
