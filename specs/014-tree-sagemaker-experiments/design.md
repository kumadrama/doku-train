# 设计

Source: 用户已确认本会话实验设计。

平台 runtime 增加 stages，默认 train/test 保持既有调用，实验显式 train。放行 rerank_tree，按模型与阶段验证归档，compare 对未执行 test 返回 null。树验证曲线通过公共事件观察接口输出，不改变选模。每Job单机CPU，先基线测量后有限并发。

分析与实验编排放 experiments/tree_sagemaker，分群阈值与实验列表配置声明。E01只移除直接类别ID，E02只加三个episode命中，E03 top200，E04移除无序集合截断。T00基线，T01深度4，T02深度8，T03 child100，T04 lambda20，T05轮数1600。单任务同步裁剪selection和publication；实验不调用发布。

同一冻结Manifest、训练10/1–15、验证10/16、seed42，测试不执行。分群依据训练曝光频率：未见、1–99、100–999、1000+、缺失。逐日覆盖与截断审计仅训练验证；指标不可计算保留原因。不引入质量门禁。

Source: 2026-10-21 用户要求剩余实验全部并发启动。并发通过启动参数声明，移除首轮4并发硬编码上限；本轮剩余20个Run使用parallel=20，启动前核对区域配额和占用，同名prepared包幂等恢复，不重跑已完成实验。
