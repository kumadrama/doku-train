# Spec 005：任务

- [x] 核对上游 S3、样本保留列与训练投影缺口。
- [x] 在 doku-offline 先写失败测试，再实现 v5 nullable 行为标签投影及合同交付。
- [x] 保持训练通用 reader；增加三模型行为任务 smoke，覆盖已知负例、NULL mask 与无 finish 行。
- [x] 完成两仓 make check 并记录结果，跨仓实际序列化产物读取通过。
- [ ] 发布新构建并按明确日期生成生产 v5 Manifest（需单独的环境与写湖授权，非本地代码验证）。
