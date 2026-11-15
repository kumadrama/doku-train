# 本地验证

用三份冻结 recipe 及对应的原始任务定义、binding 运行 prepare_daily。逐字节比较输入与输出配置，并对同一 ready 计算接入前后的 operation_identity。三个模型均完全相同。

| 模型 | recipe SHA256 |
| --- | --- |
| tf_rerank_ln_lr001 | 1361aae10d639a5da1582c42e44005638d18e089342e589f528760b383e304ad |
| tf_rerank_ln_lr002_3feat | 107feb7b7d97022f60565f4dff37b04933f6364c39a4165381e0e60aaf9a3ac6 |
| idea_mapped_rm32_drop10 | 532181ebcee082ae79477c272b3a1df38685f46553ef1e675ac8221e8cf30cad |

可执行回归覆盖绑定生成、配置字节和逻辑身份、已密封事件重试、旧产物无投递。监控仓测试对同一任务的两个 FYP 模型分别查询，保留日期和模型状态；Idea 补偿扫描跳过入组前的历史观察。IaC mock plan 覆盖现有任务族无重复失败报警，以及新任务族报警和通知主题。

```bash
make check
PYTHONPATH=. recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests/test_release_preparation.py recommend_tf/train/tests/test_monitoring_reports.py
```

生产验收需在版本启用后的自然日更比对密封事件、目录及 Grafana 返回值。此记录未计入任何生产训练或回填结果。

本次执行 make check（使用已安装的锁定依赖）通过。公共训练测试 329 通过、3 跳过，TF 测试 228 通过，推理合同 8 通过。跨仓目标回归 39 项通过，Terraform 使用 mock provider 对监控根、共享失败模块及原流水线执行 plan / mock apply，合计 6 项通过。
