# 跨日合同合并与缺列读取

Source: requirements.md；2026-10-16 用户授权方案。

合并在公共 datasvr 中完成，不增加模型分支或 YAML dataset 字段。逐日独立验证
pointer/Manifest SHA、特征投影摘要与本日合同版本。跨日标签版本取集合并集；
标签合同其他字段保持相等。特征按名字合并，完整同名定义须相等；其他特征合同
元数据（除 feature_spec_version/features）相等。合成特征合同版本由原合同内容
确定性计算，原始版本另存，不能伪称为某一天的上游合同。

冻结 Manifest 新增 daily_contracts_version 标识，source_manifests 内保存原始
Manifest/合同文本、摘要和日期；分片仍保留原始身份。读取器复验这些证据与合并
结果，根据分片 URI 找到本日声明字段。YAML 所选字段在某日未声明时必须有显式
默认值，读取时直接补该值（列表逐行复制）；已声明却物理缺列仍失败。未选字段
新增不要求默认值。普通单 Manifest 保留原有严格读取行为，但标签版本支持集合。

训练 frozen-input 内的 Manifest 自包含上述证据，恢复不再访问源合同或指针。
lineage 增加逐日源版本/合同摘要与按日默认值字段记录。保持分片摘要、时间、行数、
数据集身份、标签配置检查及发布门禁。现有 Spec 006/009 的跨日全等限制由本方案取代。

## 2026-10-31 数据集家族兼容修订（已授权）

Source: 用户确认树/Torch 与 TF 两套数据层同步修复并验收。
已核实生产 v5→v6 只新增未选用的 u_deep_link_drama_id，标签与既有字段定义未变。
两套独立 adapter 使用相同规则，不互相 import：仅识别
`fyp-rank-train/fyp-rank-features-vN`（N 为正整数），要求本日 schema 为
`fyp-rank-features/vN`；其他身份仍严格全等。跨版本同家族合成 Manifest 使用
`fyp-rank-train` 身份，同版本保持原 ID；不同家族和标签配置不兼容仍失败。
冻结复验从原始逐日 Manifest 重新计算期望身份，保留原 ID/schema/摘要，不能信任
合成身份绕过校验。lineage 增加每日 dataset_id。既有字段定义、合同元数据、摘要、
标签及缺列校验不放宽；未选新增字段不要求默认值。普通单 Manifest 读取不变。
覆盖两套真实冻结入口、离线恢复和负例，运行 make check；不包含云部署或重试。

## 发布容量对齐

Source: 2026-10-31 只读 describe-task-definition 确认生产 revision 7 为 CPU 4096 / memory 24576。
本次部署模板保持现行 24 GiB 容量，修正仓库旧 16 GiB，避免代码发布意外回退资源。
