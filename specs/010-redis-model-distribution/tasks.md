# Spec 010 实施任务

Source: 2026-10-14 用户授权“开始实施代码”；requirements/design 为功能合同。

## 本仓

- [x] 迁移 daily_run 到 run.py；scheduled.py 复用统一编排。
- [x] 新增 publish_model.py，独立发布与日更复用 S3/Redis 发布用例。
- [x] 新增二进制 Redis 传输、TLS/Cluster 显式配置和三层字节预算。
- [x] serving 子集 descriptor、原字节摘要验证、ready-last、catalog Lua CAS。
- [x] registry v2、持久操作所有权、generation/epoch、旧写入者格式隔离。
- [x] 精确版本同步、stage-only、明确接管、未提交上传取消和新 epoch 重建。
- [x] 去重后仍分发；拒绝版本不分发；迟到输入不撤销后续发布/回滚。
- [x] 两侧 current、替换保护、双消费者 pins、24 小时 drain、先 Redis 后 S3 清理。
- [x] 双协议门禁默认关闭；未初始化清理只读；中断恢复、墓碑、按登记精确键删除。
- [x] 故障注入与真实 Redis 集成测试，CI 配置本地 Redis service。
- [x] README、architecture、设计和验证证据同步。

## 跨仓与部署（本轮未执行）

- [ ] iac：专用 Redis 容量、TLS/ACL、noeviction、网络/身份、任务绑定；停止旧 writer 后升级 v2。
- [ ] mono：特征服务和 predictor 版本上下文、精确加载、Go 原生模型验收、持久 pin 与排空。
- [ ] monitoring：分发失败/未知、pending 操作、孤儿 pin、容量、版本不一致监控。
- [ ] PROD stage-only、完整请求、每日同步、清理 dry-run 验收；完成双协议验收后再开门禁。
- [ ] Redis Cluster/TLS/ACL 与 failover 的真实部署验收。DTE 不部署。

- [x] 2026-10-15：请求只传 model_name/model_version，摘要改由加载阶段校验；同步 R3 和设计 3.1。
