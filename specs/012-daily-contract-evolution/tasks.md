# 实施任务

- [x] 合同集合验证与确定性合并。
- [x] 逐日证据冻结和分片读取默认值。
- [x] lineage 与架构文档同步。
- [x] 合同、恢复、CPU smoke 测试及 make check。

## 2026-10-31 数据集身份兼容

- [x] 同步 requirements/design，明确树/Torch、TF 的独立 adapter 边界。
- [x] 两套跨日合并和冻结复验使用一致身份规则，保留逐日 dataset_id。
- [x] 新增 v5/v6 未选特征与恢复回归、身份/schema/标签不兼容负例。
- [x] 完整 make check 验收与记录。
