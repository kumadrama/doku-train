# 按日投影输入设计

Source: 本 Spec requirements 中已确认方案。实现适配 `doku-train-day/v1` 与既有 manifest/v1，
只依赖数据契约，不 import doku-offline。

dataset 两种互斥输入：

- 固定输入：manifest_uri、manifest_sha256、version。
- 按日输入：daily_prefix、projection_sha256。其余版本、身份和 allowed_prefixes 继续必填。

根据 split 的日期并集读取指针，校验完整文件 SHA、manifest 逻辑 SHA、日期与分片范围、
特征/标签版本、规范 projection SHA、逐日契约一致性。解析完再次检查指针字节，
若并发回补改变指针则失败，重试后读取一致的已发布集合。

冻结到 run/input/manifest.json 及 contracts，manifest 记录所有源 manifest 的文件 SHA、
不可变分片 URI、行数和 SHA；不复制 Parquet。resolved-config 在拟合前写出并改为固定输入模式。
恢复 checkpoint 读取原 run 的 resolved-config 输入，校验请求投影与日期一致。
最终 test 同样使用 resolved-config，不重新读取日期指针。

相同 YAML 仅排版变化不改变规范 SHA。来源、类型、截断或版本变化则阻断混用；暂不自动推断
字段子集兼容性，需回补整个训练日期范围。不可变分片保留期内保证可复现；生命周期删除后不保证可重训。

每日自动化由 doku-offline 日 join 成功后串行投影负责；doku-iac 仅修改现有日任务参数。
本仓不启动云训练或应用生产 IaC。

## 2026-10-13 Dataset 简化（当前规则）

Source: 用户确认单 Manifest 地址方案。dataset 只声明 manifest_uri；旧版本/摘要、adapter、
aws_profile、allowed_prefixes 和训练内 daily_prefix 说明被本规则取代。
版本与摘要从 Manifest 读取并校验，AWS 身份由启动环境提供，无生产默认值。
每次 Run 冻结 Manifest/合同原文及摘要，resume/test 使用原冻结记录；不重新解析日更指针。
