# 验证

执行日期：2026-10-17。全部为本地合成数据；没有调用 AWS/Grafana、部署或启动线上任务。

## doku-train

```bash
UV_CACHE_DIR=/tmp/doku-train-uv-cache UV_OFFLINE=1 make check
```

结果：锁文件校验、ruff format/lint、mypy 通过；**168 passed, 3 skipped**，74.71 秒。
包括真实最小 CPU smoke：成功发布/发布门禁拒绝都保存观察文件，重复触发不再输出；
训练/test 失败没有事件，日志 observer 故障不改变发布决策。
UV_CACHE_DIR 使用临时目录是为了避开当前沙箱不可写的默认 uv cache。

## doku-monitoring

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_reco_train_dashboard tests.test_loader -v
```

结果：**22 tests OK**。六个新测试覆盖显式配置别名、四任务八图双模型、缺失/最后 null、
乱序重放、上海跨日、输入校验和实际 CLI 生成。其余测试验证既有 loader 激活行为。

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover --start-directory tests --verbose --failfast
```

初次完整入口结果：95 tests，1 failure，4 skipped；**没有全绿**。失败是既有
`test_domain_definitions_live_in_domain_packages`：fyp_rank_samples 缺少 dashboards.py / metrics.py。
`git ls-tree -r --name-only HEAD src/monitors/fyp_rank_samples` 确认 HEAD 也只有
__init__.py / alerts.py；本次没有修改此目录或对应规则测试。
最初未加 failfast 的探索运行在发现该失败后终止；最终代码的上述 failfast 结果为准。
另一次包含 authoring_boundaries 的定向回归为 43 tests，只有同一个既有失败。

### 后续：补齐 FYP 领域模块并完整重跑

2026-10-17 用户追加授权后，monitoring 补齐 fyp_rank_samples 的 metrics.py / dashboards.py
及显式导入。两者只说明 CloudWatch-only 边界，不增加指标或看板。结构及 loader 测试
37 项通过，原缺模块失败消除。

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover --start-directory tests --verbose
```

本次完整运行结束：**775 tests，4 failures、50 skipped，641.419 秒**。
剩余失败为现有 FYP mention 通知与电话例外约束不一致（1 项），以及 item-feature 三张
既有图的固定 5m/15m/5m 与 query_window 测试不一致（3 个子案例）。未修改对应告警、
窗口或放宽测试；**仍不声明完整回归全绿**。详情见 monitoring 的 training-test-auc.md。

## 跨仓及产物

```bash
cmp recommend/train/tests/fixtures/test-auc-v1.json \
  /Users/yujiatang/doku/doku-monitoring/tests/fixtures/test-auc-v1.json
git diff --check
git -C /Users/yujiatang/doku/doku-monitoring diff --check
```

协议 fixture 字节一致；两个仓库 diff 检查通过。生产者按固定时间生成 fixture 的逐字段
等价事件；消费者验证 sortsLast + jsonParse 的本地合同语义。双方不 import 对方实现。
实际运行本地 CLI，以合成 model_a/model_b 生成 dashboard.json 并读回确认 2 rows / 8 panels；
使用假 datasource/log group/ARN，无网络调用。示例产物只留临时目录，不提交 Git。

## 验证限制

本地排序 oracle 不是 AWS Logs Insights 执行器。真实 CloudWatch 查询、Grafana 渲染、
日志送达及 datasource 权限未验证，留待后续明确授权的部署验收；不将静态 JSON 测试
声明为云端可用。新增模型需将配置加入生成命令，不支持运行时自动发现。

### 最新：配置与测试约束对齐后完整通过

2026-10-17 用户继续授权修复四项失败。monitoring 将四条 FYP 外部告警改为 phone；
item-feature 三张图使用 QUERY_WINDOW，告警固定窗口不变。未放宽任何测试断言。

`PYTHONPATH=src .venv/bin/python -m unittest discover --start-directory tests --verbose`
结果：**775 tests，654.145 秒，OK (skipped=50)，0 failures / 0 errors**。
50 项跳过均为未开启 DOKU_MONITORING_TEST_PROMTOOL 的可选测试；默认完整入口中实际执行的
promtool 合同回放通过。通知/窗口/名单定向测试 36 项通过，git diff --check 通过。
此结果取代前述历史失败状态；没有部署或触碰生产。
