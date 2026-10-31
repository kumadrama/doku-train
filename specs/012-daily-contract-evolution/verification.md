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

## 2026-10-31 v5/v6 身份兼容本地验收

Source: 本次本地执行；不代表已部署或生产重试成功。

- `uv run pytest recommend/train/tests/unit/test_daily_contract_evolution.py recommend/train/tests/unit/test_dataset_identity.py -q`：17 passed。
- `env -u UV_PROJECT_ENVIRONMENT PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 uv run --project recommend_tf --locked python -m pytest recommend_tf/train/tests/test_dataset_identity.py -q -p no:cacheprovider --disable-warnings`：2 passed，100 warnings。
- 最终 `make check`：退出码 0；两套依赖锁检查、Ruff 格式/lint、Mypy（70 文件）通过；
  树/Torch 320 passed、3 skipped（173.26 秒），TF 170 passed、154 warnings（166.83 秒）。
  首次执行发现新增测试闭包未绑定循环变量的 B023，修复后重新完整执行通过。
- `git diff --check`：通过；AGENTS、interfaces 与本 Spec 共 6 个 Markdown 文件链接检查无缺失。
- 两套 daily_contracts.py 的 `diff -u` 无差异；没有跨框架 import。

合成 fixture 覆盖真实 daily prepare / TF freeze 入口，v5/v6 混合且新增未选特征无默认值，
读取所选列成功；删除源 Manifest 后冻结恢复成功；逐日 dataset_id 保留。
身份负例覆盖不同家族、原始 canonical ID 混入版本化家族、schema 不匹配、标签配置不同；
原有合同语义冲突、物理缺列、默认值和冻结证据校验继续通过全量测试。
现有未提交 TF 工作保留；未执行提交、部署、ACK、云训练或重做样本。
