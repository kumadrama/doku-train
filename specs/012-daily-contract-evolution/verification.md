# 验证记录

Source: 本地合成数据验证，2026-10-16；不代表生产部署或真实数据训练。

| 需求 | 可执行证据 |
| --- | --- |
| 跨日标签版本合并；普通 Manifest 多版本校验 | unit/test_daily_contract_evolution.py |
| 标量 -1、列表 [] 默认值；逐行隔离；旧日缺列 | test_merge_defaults_and_restore_without_source_metadata |
| 缺少显式默认值拒绝 | test_missing_explicit_default_fails |
| 已声明特征、标签、时间列缺失拒绝 | test_declared_feature_or_required_column_missing_fails |
| 同名类型/来源/单位、标签语义和其他元数据冲突拒绝 | test_semantic_conflicts_rejected |
| 未选新增字段不要求默认值 | test_unselected_new_feature_needs_no_default |
| 冻结证据、分片和合并版本不一致拒绝 | test_frozen_evidence_cannot_disagree_with_merged_manifest |
| 指针竞态和投影摘要不一致拒绝 | unit/test_daily_input.py |
| 移除源合同和指针后恢复读取 | test_merge_defaults_and_restore_without_source_metadata |
| 混合合同 CPU 树模型完整训练、本地发布、冻结后独立 test、lineage | smoke/test_daily_contract_training.py |

执行环境使用现有锁定依赖；`UV_CACHE_DIR=/tmp/doku-train-uv-cache` 避免默认缓存写权限限制，
`UV_OFFLINE=1` 保证验证不访问网络。验证结果：

- `UV_CACHE_DIR=/tmp/doku-train-uv-cache UV_OFFLINE=1 make check`：通过。锁文件、
  Ruff 格式/lint、Mypy（44 个源文件）全部通过；pytest **133 passed, 3 skipped**，66.06 秒。
- 新增混合合同单测：15 passed；新增 CPU smoke：1 passed（已包含于上述全量检查）。
- `git diff --check`：通过；8 个本次相关 Markdown 文件的本地链接检查无缺失。
- 验证期间仅使用合成样本与本地 ObjectStore，没有启动云训练、上传或部署生产。

## 提交前审查与主线集成

Source: 2026-10-16 用户授权审查、创建 PR、部署及补跑。审查覆盖默认值/声明列边界、
逐日证据与合并结果一致性、冻结恢复和 AppConfig/Redis 既有生产入口。未发现阻塞问题。
已合入主线 d9c1b03；解决文档索引冲突并将本 Spec 编号调整为 012，保留原有
Spec 011 AppConfig 控制。用户预先存在的 SageMaker 草案和索引不纳入本次提交。

再次执行 `UV_CACHE_DIR=/tmp/doku-train-uv-cache UV_OFFLINE=1 make check`：
**157 passed, 3 skipped**，69.41 秒；格式/lint/类型（45 个源文件）和锁文件检查全部通过。
生产部署和补跑授权不代表已经执行成功；结果以发布/工作流记录为准。
