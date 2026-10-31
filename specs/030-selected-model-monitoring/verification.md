# Spec 030 验证记录

状态：2026-10-31 本地实现与验证完成；未部署、未改变生产日更触发、Endpoint、Grafana 或告警。

## 已运行验证

| 范围 | 命令与结果 |
| --- | --- |
| TF 全量 | 仓库根目录 `recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests -q --disable-warnings`：**168 passed**，144.98 秒；包含离线实验、early_stop、日更、导出和发布回归。 |
| 监控全量 | doku-monitoring `PYTHONPATH=src .venv/bin/python -m unittest discover --start-directory tests -q`：**781 tests，OK，50 skipped**，658.235 秒。 |
| 监控最终相关回归 | `PYTHONPATH=src .venv/bin/python -m unittest tests.test_prod_config tests.test_oncall_rosters tests.test_reco_train_dashboard tests.test_model_metrics tests.test_model_metrics_reconcile -q`：**48 tests，OK**，5.520 秒；包含全量回归后补充的边界用例。 |
| Terraform | doku-iac/modules/model_monitoring：`terraform fmt`、`terraform validate`、`terraform test`：配置合法，**1 mock test passed**；未执行 AWS plan/apply。 |
| 合同 | 根 `.venv/bin/python` 使用 jsonschema Draft 2020-12 校验两个事件 fixture；train / monitoring 的 schema 与 fixture 逐字节一致。 |
| 代码规范 | 本次训练/平台新增与修改文件 Ruff check/format 通过；三个仓库 `git diff --check` 通过；Spec 030 本地文档链接有效。 |
| 独立运行包 | `python scripts/package_model_monitoring.py --output ...` 连续两次输出 SHA256 相同；从 zip 单独 import `pkg.model_metrics_runtime` 成功，不需要加载作者层 registry。 |

最终运行包 SHA256：
`f8ae75e7f8c7b819737b8bf385065e763fb764f7909cf25e1d9a2a0aa3a5cc16`。
包仅位于本机临时目录，未上传。

Terraform 在受限沙箱内无法启动 provider；之后在允许启动 provider 的本地执行环境完成
validate/mock test。AWS provider 6.71.0 对 `aws_region.name` 给出弃用提示，不影响本次验证结果。
监控全量日志包含已有 CLI 负例的 no rules 文件提示；最终 unittest 退出码为 0，并非测试失败。

## 关键证据

- [选中 epoch 测试](../../recommend_tf/train/tests/test_selected_metrics.py)：严格读取 selection
  对应的同一 epoch；保留 null/原因；日期变化不改变 recipe，未知训练选项会改变 recipe。
- [监控发布与回填测试](../../recommend_tf/train/tests/test_monitoring_reports.py)：校验封存文件摘要，
  重复 outbox 幂等，冲突拒绝，历史配置不变；历史完成时间必须来自拥有该 run URI 的原训练任务。
  封存前的小文件证据与封存后报告完全相同，不扫描 prepared 全量数据。
- [日更测试](../../recommend_tf/train/tests/test_daily_publication.py)：监控 completion 回调故障时
  训练仍完成并封存原始 selection/epoch；不增加 test 阶段。
- monitoring `tests/test_model_metrics.py`：日志失败可重试、重复去重、同名配方冲突、乱序发布和
  回滚、失败保留 online；旧 run 晚到不能覆盖最新 null 或混用任务；目录丢失可由已投递事件重建；
  第二模型自动 Row、MSE 无 0..1 上限、旧树面板保留、日期控件持续包含新日更、错误发布目标拒绝。
- monitoring `tests/test_model_metrics_reconcile.py`：ECS 写 outbox 前退出仍从封存标准事件恢复；
  未完成 run 后续重新扫描；源事件摘要不符拒绝。
- Terraform mock：未启用的配置不能启动 queue/schedule 触发；执行并发固定为一，避免补偿游标
  与看板同步互相覆盖。生产角色、KMS、桶通知与实际 Grafana 权限仍需部署环境验证。

## 尚未执行的云端验收

1. 填充并复核生产环境配置、target profile、Secret、KMS/bucket 通知及角色 policy；生成真实 binding。
2. 部署独立共享 worker、训练镜像与平台适配；第一轮保持自动触发/告警关闭。
3. 显式回填现有 28+1 成功 run，核对所选 epoch 四任务值及旧文件摘要。
4. 对真实 CloudWatch/Grafana 验证 jsonParse/null、数据日期、跨窗口补报、目录自动新增、并发编辑重试。
5. 回放告警噪音并完成当前状态检查，再启用共享补偿与告警。

这些项目不能用本地模拟测试替代。本次没有重新训练、重新导出或发布生产模型，也没有提交 PR。
