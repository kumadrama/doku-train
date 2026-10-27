# Spec 019：DeepFM 耗时观测需求

Source: 2026-10-27 用户经 grilling 确认 Q1–Q7，并授权基于最新 main 开分支先文档后实施。

状态：需求已确认，本地实现及验证完成；未部署。

- 仅为 DeepFM（rerank 且 fm_layers 非空）开启新增观测；普通 DNN、独立树训练不启用。
- profiling.enabled / profiling.print_logs 均默认 false；enabled 控制采集及 Run 文件，print_logs 控制新增 stdout。关闭采集但请求打印必须报错。
- 数据准备记录冻结、下载与校验、Parquet 读取/解码与转行、拟合、长度扫描、编码与数组赋值；模型记录初始化、取 batch、forward/loss/backward/optimizer、验证推理与指标、checkpoint、独立 test、归档。
- 前置树分桶仅记录总耗时，不细拆树内部。
- 大阶段开始/结束输出，长阶段每 60 秒在可用进度边界输出，epoch 结束汇总；batch 细项内存累计，不逐 batch/逐行写日志。
- stdout 与 Run 文件共用测量事件；JSONL 增量 flush，summary 原子写出；普通异常保留失败阶段与已完成证据。重复 test 使用独立 invocation。
- 不改变特征、随机种子、训练更新、loss、评估频率、checkpoint 频率与选模。计时开关不进入训练逻辑 identity。
- 本次不做性能优化、跨实验缓存、云训练或部署。此前 3 小时估计不是本次验收承诺。
