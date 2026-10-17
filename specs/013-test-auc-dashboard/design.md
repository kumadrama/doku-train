# 设计

Source: 用户批准的仓库分工及本仓 run.py / scheduled.py / deploy/prod/task-definition.yml；monitoring 的 pkg/grafana.py 目前只支持 PromQL。

## 数据与接入

- train 在 daily_run 的 execute_test 成功后、gate 前记录 run-scoped `test-auc-events.json`，通过可注入 observer 交给 scheduled 的 stdout JSON-lines 出口。普通本地调用不强制发送日志。
- 只读取已声明 func 为 auc/regression_auc 的指标，保留 metric 名及 kind，允许同任务多个显式指标，禁止按 key 猜口径。框架不硬编码四任务。
- 每条事件包含 format=doku-test-auc/v1、model_name、task、metric、kind、run_id、completed_at_ms、day_start_ms（Asia/Shanghai）、auc（可 null）、observation。
- observation 为定宽完成毫秒 + run_id + 包含 auc 数值/null 的 JSON 对象 的排序字符串。排序先完成时间，同毫秒按 run_id 稳定仲裁。重放原事件不能覆盖更晚完成的 run。运行标识只用于内部去重，不展示。
- 原事件 create-only 写入，重发必须复用原时间与内容；成功和未发布训练归档均保存事件。日志投递失败显式记录错误状态但不更改已完成训练/发布逻辑。日志系统的送达不由 stdout 成功保证，不声称 exactly-once。

## 看板

- monitoring 提供独立的本地 CloudWatch artifact 生成入口；沿用现有 Grafana JSON 形状，不扩展/冒用 PromQL publisher。不会自动出现在现有 monitor-publish 发布清单。
- 显式传入 region、CloudWatch datasource UID、log group 名称/ARN 及一个或多个训练 YAML。每个配置的 publication 映射选择四个 AUC 指标（若无 publication，要求四任务各有唯一 AUC 指标）；歧义或非 AUC 指标拒绝生成。
- 每个 model_name 生成独立 Row，四图 2×2；增加模型时将其 YAML 加入生成命令并重建 artifact，不做运行时模型自动发现。
- Logs Insights 按 day_start_ms 分组并 sortsLast(observation)，再 jsonParse 得到数值/null；毫秒时间列由 Grafana convertFieldType 转为绝对时间；按事件业务时间聚合而非日志到达时间。单模型单任务查询，禁止 avg/max AUC。
- 图中仅时间和 AUC；null 不连线，超过一天的缺口断开；默认 30 天。日志查询受日志保留期与查询范围约束，不回填既往未记录事件。

## 决策

[INFERRED] 使用已有 awslogs 可避免本次新增采集服务/权限；仍须在未来部署前验证 Grafana CloudWatch datasource 和读日志权限。不会修改 iac/pipeline，也不调用 AWS/Grafana API。

协议权威为本 Spec；monitoring 使用合成 fixture 验证，无跨仓实现 import。

参考：[AWS stats](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax-Stats.html)、[AWS JSON/时间函数](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax-operations-functions.html)、[Grafana LogsQuery](https://grafana.github.io/grafana-foundation-sdk/go/Reference/cloudwatch/object-LogsQuery/)。
