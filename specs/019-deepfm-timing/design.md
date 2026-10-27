# Spec 019：DeepFM 耗时观测设计

Source: 用户确认方案及当前 training_pipeline、FeaturePlan、NeuralRunner、SageMaker 源码。

状态：设计已确认，本地实现及验证完成；未部署。

## 配置与隔离

rerank schema 增加可选 profiling 对象，两个布尔值默认为 false；只允许 fm_layers 非空时开启。保留现有普通训练日志及粗粒度 DOKU_STAGE，开关只控制新增 DOKU_TIMING。公共计时器不依赖模型；模型限制由配置验证完成。

## 收集与输出

comm/timing.py 提供上下文局部 session、stage/span、函数和 iterator 测量。关闭时不读取时钟、不创建目录；保留少量调用开销。使用 perf_counter 测 CPU 同步调用的 elapsed wall seconds。每次 invocation 用独立 UUID 文件，UUID 不使用训练 RNG。

文件为 timing/<phase>-<invocation_id>.jsonl 与对应 -summary.json。事件包含版本、run_id、phase、stage、status、elapsed_seconds、count/units，epoch 汇总包含累计值及本轮增量。仅大阶段开始/结束、60 秒进度和 epoch 汇总输出；JSONL flush，summary 临时文件 replace。失败只记异常类型，不打印样本或异常消息。

阶段名采用层级路径；父时间包含子时间，不相加。并行下载 worker 继承提交时的阶段上下文，累计耗时标注可能重叠；主线程等待独立统计。iterator 仅测 next，不跨 yield 包含消费者耗时。每行操作可累计，但不每行探测 RSS/写日志；周期进度是协作式边界，不承诺不可中断 native 调用期间精确每 60 秒输出。

## 测量边界

- freeze_input：Manifest/合同读取与冻结整体。
- tree_pretraining：独立子进程树训练及阈值处理总时间。
- fit_features / encode_train / encode_validation / encode_test：父阶段；row_transform、vocab_collect、vocab_finalize、width_scan、encode_row、array_allocate、array_assign 为子项。
- shard_download 包含网络和落盘；checksum / metadata_validation 分开；parquet_batch 包含读取解压解码，to_pylist 单列，行校验单列。memmap array_assign 不代表物理落盘，不增加 fsync。
- model_initialize、shuffle、batch_fetch、zero_grad、forward、loss、backward、optimizer、validation_epoch / validation_selected / test_evaluate、metrics、resume_save、checkpoint_serialize / checkpoint_write。
- test 包括恢复冻结输入、编码、模型初始化、checkpoint 校验与加载、评估和报告写入；在 run lock 内创建 invocation。
- SageMaker archive 记录归档复制与校验清单，平台容器退出后的压缩上传不可在进程内测。归档计时使用独立 invocation；完成后将 finalized timing 文件复制进 archive，最后原子发布清单；计时自身收录和最终清单写出不计入 archive 段。失败时 timing 复制到 output/data 供诊断。开启 profiling 的归档必须有 train、配置要求的 test 和 archive 成功 summary 及对应 JSONL，缺失时拒绝发布或验收归档。

## 可复现性与兼容性

profiling 从逻辑配置摘要剔除；计时不进入 Feature State 或指标质量语义。保持来源 revision/code digest 校验，因此只保证同一实现中开关切换的 identity 稳定，不绕过跨源码 resume 身份检查。训练 checkpoint 中配置保留原始观测设置用于追溯。

## 验证

固定时钟验证累计/层级、关闭零时钟、周期输出、失败 summary、iterator 不计消费者时间；配置非法组合；DeepFM 开关前后模型参数/指标/identity 相同；重复 test 文件隔离；epoch/数据处理事件覆盖；归档校验与失败诊断；运行 make check。合成测试不等于生产性能证据。

## Spec 022 批准备路径的计时口径

Source: 2026-10-27 用户确认 [Spec 022](../022-batched-feature-preparation/design.md) 并授权本地实施。

神经网络批准备路径以 column_decode / batch_transform / history_rows / batch_encode
替代旧 to_pylist / row_transform / encode_row 热点，history_scalar_rows 和 expression_fallback
记录保留的标量处理。训练拟合成功并复用完整宽度时不产生 train/encode_train/width_scan；
validation/test 保留 width_scan。array_assign 按批有效行数累计，仍不是物理磁盘落盘耗时。
独立树旧路径和计时开关不变；不同路径子项不能直接按同名总和推断端到端加速。
