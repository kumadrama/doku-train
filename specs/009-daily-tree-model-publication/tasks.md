# Spec 009 实施任务

Source: 2026-10-13 用户确认继续使用 S3 并开始实施代码；不执行云端操作。

- [x] 配置 model_type/model_name 分离、离散日期与发布指标声明。
- [x] regression_auc、独立评估、树 checkpoint 序列化边界。
- [x] 原生树模型导出、特征合同、校验和及重新加载一致性。
- [x] 本地/S3 条件存储、发布清单、去重与同名串行。
- [x] pin、回滚、draining 清理及 dry-run。
- [x] 日投影准备与单次流水线、CLI。
- [x] 合同测试、CPU smoke、make check、文档与验证记录。

## 2026-10-20 派生规则交付修复

Source: 用户授权按已验证方案修复导出合同、补测试及文档、同步飞书。

- [x] encoding 原样包含 history_scalars，空配置写空对象。
- [x] 新增合成数据端到端导出测试：合同摘要、列顺序、独立重建、五种算子边界。
- [x] 明确旧 Go 消费者两个入口需同步适配及新制品版本切换顺序。
- [x] 运行完整 make check 并记录结果，同步飞书验证与配合事项。
- [ ] 外部实施：服务端升级、发布新制品及线上切换（本次不执行）。


## 外部接入（不属于本仓代码实施）

- iac：上游投影成功后触发 daily；调度 cleanup，装配身份、资源和显式 S3 前缀。
- mono：后台下载、内存缓存、固定版本请求、实例 pin 与多副本就绪协调。
- monitoring：消费 published/rejected/failed/skipped 结果与持久 pin 告警。
- 云端 smoke、Go parity 和真实数据容量验证需对应接入及授权；见 verification.md。

### PROD 编排接入

- [x] 上游成功后独立训练阶段，冻结 cutoff、隔离重试，DTE 不接入。
- [x] PROD 每日 12:00 UTC 清理、按模型范围 IAM、SSE-KMS 写入。
- [x] 固定摘要配置的容器适配器、CPU 镜像构建、PROD task definition、无 Git lineage。
- [x] mock 编排和容器边界测试；云端 apply、配置上传、镜像发布和调度验收尚未执行。
