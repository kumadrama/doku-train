# 验证

可执行检查：

```sh
python idea-mapped-daily/build/package_source.py --pin
python -m pytest -q recommend_tf/inference/tests \
  recommend_tf/train/tests/test_release_preparation.py \
  recommend_tf/train/tests/test_daily_runtime_policy.py
```

本地结果：上述 32 项检查通过；仓库 inference-check 对应的 unittest discover 为 26 项通过。daily recovery/publication 的 TensorFlow 训练回归在当前 Python 3.12 环境缺少 TensorFlow，未执行；完整训练回归仍需仓库锁定的 Python 3.10 / TensorFlow 2.11 环境。

使用生产已发布的 2026-11-14、2026-11-15 两份真实模型完整包校验 canonical inventory，依次生成 Discovery profiles，并用未修改的 AI `cmd/releases --workload discovery` 加载成功。实际新模型身份分别为 `e3dbff6c83f096b9…`、`ea4d23f6e1622690…`，两份配置均保留原 bootstrap；这是本地合同回放，未调用线上预测或修改 AppConfig。

配套 IaC `envs/prod/tf_endpoint/discovery_appconfig.tf` 通过 Terraform validate；mock plan 两项通过，覆盖仅给原 FYP worker 发布两个 Discovery profiles 的权限及错误 profile 身份拒绝。未执行 apply。

`test_discovery_appconfig.py` 覆盖连续三天更新、定义先于路由、旧列表混读、bootstrap 保留、合同差异拒绝、发布重试和无操作；通过原 daily advance 路径验证只更新原 endpoint，合成探测成功之前不写配置。

`test_idea_publication.py`、`test_idea_appconfig.py` 保证已有发布的 CAS 与重试语义；`test_official_pipeline.py` 及 daily recovery/publication 保证未配置 hook 的流程。

上线检查：模型 defs/list 发布完成后，各副本已有 `ai_engine.model_warmed`；实际 Discovery RPC alias 返回新的完整 release/version；同请求多批次身份一致，样本只写 Discovery 前缀。记录端点更新窗口的降级和 RT；实验保持 10%。当前仅代码与本地回放验证，不以测试替代线上采纳验收。
