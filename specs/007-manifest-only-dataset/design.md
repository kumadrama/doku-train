# 设计

Source: requirements.md 与用户已确认方案。

公共 schema 的 dataset 为仅含 manifest_uri 的封闭对象。读取层根据 manifest_version
识别已支持的 doku-train-dataset-manifest/v1；未知版本失败。数据、特征和标签版本从源读取，
不把具体实验的 v4/v5 标签固定在框架中；特征/标签合同必须与 Manifest 一致。
移除训练入口 daily_prefix 模式，daily_input.resolve_days 保留为独立冻结辅助函数。
此行为取代 Spec 006 的训练入口自动解析 daily 指针设计。

PublishedDataset 在内存中持有首次读取的原文，freeze 写入 run/input/manifest.json、
contracts/{features,labels}.json、record.json；后者记录源 URI、文件 SHA256、逻辑 SHA256
及合同摘要。保留原源 URI 和分片 URI，不因冻结文件目录变化改写训练逻辑身份。
恢复训练从 checkpoint 相邻 input 读取，测试从原 Run 的 input 读取，校验冻结摘要，
不重新读取原 Manifest/合同。精确分片仍从原 URI 读取并核验 SHA256/行数。
resolved-config 仍保留用户源 URI；独立新 train 是新冻结，resume/test 才复用旧冻结。

拒绝不支持的 URI scheme、query/fragment、路径穿越和远程 Manifest 引用本地文件。
本地 fixture 允许 file URI；S3 权限由 IAM 管理。boto3 使用标准环境凭据链。
无预期摘要预填，源首次读取之前的替换不能被独立识别；首次读取之后由冻结记录保证一致性。
源 Manifest 可未提供合同摘要，仍保存实际读取合同的摘要；有上游摘要时必须校验。
