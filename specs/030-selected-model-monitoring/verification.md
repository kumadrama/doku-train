# Spec 030 验证记录

状态：2026-10-31 代码已合并，监控和日更接线已部署生产；历史数据与实际调用验收通过。下一次真实日更随上游 ready 自然触发，本次没有重新训练。

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

## 实施初期的待验收项（历史记录）

1. 填充并复核生产环境配置、target profile、Secret、KMS/bucket 通知及角色 policy；生成真实 binding。
2. 部署独立共享 worker、训练镜像与平台适配；第一轮保持自动触发/告警关闭。
3. 显式回填现有 28+1 成功 run，核对所选 epoch 四任务值及旧文件摘要。
4. 对真实 CloudWatch/Grafana 验证 jsonParse/null、数据日期、跨窗口补报、目录自动新增、并发编辑重试。
5. 回放告警噪音并完成当前状态检查，再启用共享补偿与告警。

这些项目不能用本地模拟测试替代。本次没有重新训练、重新导出或发布生产模型，也没有提交 PR。

## 2026-10-31 生产部署与验收

本次明确授权：按顺序合并并部署生产；不重新训练，不主动更新 Endpoint 模型。
生产账号 `703844615824`，区域 `us-east-2`。

- 依次合并 doku-iac #650、doku-monitoring #301、doku-train #28；各 PR CI 通过。
- 从独立干净 worktree 构建，排除原工作区 Cangen 等未提交实验；TF 回归 **147 passed**。
- 监控查询实测发现 ISO 日期字符串范围比较返回空结果，经 doku-monitoring #302 改为
  `YYYYMMDD` 数值比较；实际 Logs Insights 与 Grafana datasource 查询均返回四项指标。
- 原生 amd64 CodeBuild `doku-train:bb65eeea-05ea-4c3e-99ad-87820c33e344` 成功：
  验证最终 ECR 镜像入口、导出入口，选模、回填、日更、离线实验 **41 passed**。
- 完整下载并校验已有日更 run 的 **291** 个封存文件；配置摘要
  `f1f879d416c790552e547450e0c934a665d4fe81b290204fe4f0068d43909e25`，文件未改写。
- 原选模记录选择 **epoch 6**，数据截止 `2026-10-29`，validation 156566 行；没有重新选 epoch。

| 任务 | Test 指标 | 原始值 |
| --- | --- | --- |
| completion | AUC | 0.8238272465707811 |
| fast_swipe | AUC | 0.7808322961199293 |
| immersive_click | AUC | 0.7715376417670834 |
| watch_seconds | MSE | 1026.044862521159 |

CloudWatch 查询的数值序列化保留四位小数，看板显示三位；S3 原始事件保留完整精度。
选模事件 ID `32d26aa762a9a5bdf0fa55075f9b336d0502bfb6a67232ddf96f7b2cd6797685`，
重复回填写入同一个不可变对象并验证内容，不重复写指标。

当前线上版本从 Endpoint 状态、真实合成请求和发布日志交叉核验：
`410f512b49800352e265deb24f3563b8ff7aa521e1af59833be59d6d6301fcc3`。
历史发布事件 revision 6 来自恢复演练的最后记录及 Endpoint LastModifiedTime，
没有用回填时刻伪造发布时间，也没有修改历史发布日志。

Grafana dashboard UID `reco-train-test-auc`：自动新增 `tf_rerank_ln_lr001` 的四任务面板和
发布状态，原五个树模型 panel JSON 逐字段不变。四个 Grafana datasource 查询返回 HTTP 200、
Complete 状态和实际数据帧；MSE 轴无 0..1 上限。

监控采用独立 Lambda/SQS/EventBridge、KMS 加密日志组 `/doku/prod/model-evaluation`；
Grafana 专用 Editor service account 每次同步创建 600 秒 token，并在 finally 删除。
未创建静态凭据。生产环境绑定由 Terraform output 生成，保存在 doku-iac 的
`bindings/prod/model-monitoring.json`。

最终生产接线经 doku-iac #651 合并部署：

- ECS task definition `doku-prod-tf-rerank-daily:5`；先注册新 revision 再切换 target。
  本次旧 revision 4 因原状态未设置 skip_destroy，已变为 INACTIVE；定义仍可查询。
  新 revision 设置 skip_destroy=true，后续替换会保留注册状态。回退本次镜像需重新注册旧定义。
- 原数据 ready 规则与日更规则保持 ENABLED；监控 outbox/reconcile 两条规则 ENABLED，
  SQS event-source mapping Enabled，Lambda Active/Successful，每五分钟补偿一次。
- 稳定名称 `tf_rerank_ln_lr001` 来自每日配置；28 天 train + 1 天 validation 不变。
  新配置与历史配置 recipe 摘要相同，避免把 transport/model_name 变化误判为新配方。
- 配置摘要 `c230fc809646e0525679f2254151a7a5c11f68aa4653914ff3d5be250b4dfbb5`；
  binding 摘要 `47ba11eef751bb48476195e6883d6a0620f5dbc1c7954f9aaee54d145e94df6a`；
  S3 下载后的 SHA256 与 EventBridge target 中的冻结摘要一致。
- pipeline Docker V2 digest：`sha256:eea6d24f7343a24afaa1d3d019f045ca4bb663af5118d8149b4d8b991c047805`。
- train Docker V2 digest：`sha256:ef974b2170d4e90c2be330a4d93e2dd0d2839064b7fe7426bf4c97705421b640`。
- EventBridge pattern 对真实 outbox 坐标匹配；向已启用 SQS 重投原事件，队列恢复为空，
  CloudWatch 对应 event_id 仍仅一条。当前 catalog 含四任务、epoch 6、真实训练 job 和线上版本。
- `DescribeEndpoint` 仍为 InService，EndpointConfig 与部署前相同；
  `ListTrainingJobs(NameContains=doku-daily-)` 仍只有原任务，没有触发重训。

完整跨语言验证补齐了生成的告警名称和中文标题（doku-monitoring #303）。
`make test PYTHON=... PROMTOOL=/opt/homebrew/bin/promtool` 的 Python 部分 **665 tests OK**
（1 个既有 expected failure）、**35** 个 public pattern variants 通过、合同 fixture 通过。
Go 阶段因本机默认版本为 1.27.1，按仓库要求显式设置 `GOTOOLCHAIN=go1.26.6` 后继续运行：
`make -C notifier check`、`make -C investigator check`、`make -C release-observer check` 全部通过；
`make -C dsh-agent/alert test` **25 tests OK**。

验收命令包括 `terraform plan/apply`（两个独立状态根）、
`aws lambda invoke`、`aws logs start-query/get-query-results`、Grafana `/api/ds/query`、
`aws events test-event-pattern`、SQS SendMessage/GetQueueAttributes、SageMaker Describe/Invoke。
图表查询 API 实测完整数据帧，未以人工目测代替验证。

边界：本次没有重新训练、重新导出或更新模型 Endpoint，未改业务推荐服务。
新代码下的下一次真实日更需等待上游自然 ready；已验证最终容器与接线，不把它声称为已完成的新生产训练。
监控告警资源已创建，但通知 actions 仍关闭，尚未完成历史通知噪音回放；本次只启用采集、看板和补偿。
