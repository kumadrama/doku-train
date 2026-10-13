# Spec 009 实施任务

Source: 2026-10-13 用户确认继续使用 S3 并开始实施代码；不执行云端操作。

- [x] 配置 model_type/model_name 分离、离散日期与发布指标声明。
- [x] regression_auc、独立评估、树 checkpoint 序列化边界。
- [x] 原生树模型导出、特征合同、校验和及重新加载一致性。
- [x] 本地/S3 条件存储、发布清单、去重与同名串行。
- [x] pin、回滚、draining 清理及 dry-run。
- [x] 日投影准备与单次流水线、CLI。
- [x] 合同测试、CPU smoke、make check、文档与验证记录。


## 外部接入（不属于本仓代码实施）

- iac：上游投影成功后触发 daily；调度 cleanup，装配身份、资源和显式 S3 前缀。
- mono：后台下载、内存缓存、固定版本请求、实例 pin 与多副本就绪协调。
- monitoring：消费 published/rejected/failed/skipped 结果与持久 pin 告警。
- 云端 smoke、Go parity 和真实数据容量验证需对应接入及授权；见 verification.md。
