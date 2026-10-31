# Finder CPU migration (Spec 024)

Independent Python 3.10 / TensorFlow 2.11 package. The old TF implementation is not imported.

From repository root:

```sh
uv sync --project recommend_tf
PYTHONPATH=. recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests -q
PYTHONPATH=. recommend_tf/.venv/bin/python -m recommend_tf.train.run --help
```

`train_rerank.train` accepts an explicit Doku dataset configuration and normalized Finder `model`
configuration. `run` is the lower-level prepared-input entry. No production source is defaulted.
The runnable synthetic example is `train/tests/test_migration.py`.

Input preparation stores basic fields and ragged histories; transforms, lookup and labels remain
in the graph. Use separate scalar/embed groups. Labels have no implicit validity mask: configure
default values and `loss_weight`. `loss_by_weight` preserves Finder's nonzero-count denominator.

Adam does not accept a Torch-style `weight_decay`. Finder AdamW requires an explicit coefficient;
its coefficient is not automatically multiplied by learning rate and sparse decay touches only
indexed rows. Dynamic embedding services are replaced with fixed train-only mappings.

`source-inventory.json` records exact extracted source functions. Local adapter functions are
identified in Spec 024; whole-file identity is not claimed for adapted model functions.

Current implementation and remaining acceptance work are recorded in Spec 024 tasks/verification.
Do not treat passing synthetic tests as production or throughput validation.


For a published Manifest, normalize the experiment with
`models.rerank.data_utils.migrate_experiment(experiment, raw_dtypes, optimizer_settings, shuffle_buffer_size)`.
The Finder optimizer settings must be explicit; no implicit Torch decay conversion is provided.
Pass the resulting JSON to `python -m recommend_tf.train.run --config ... --output ...`.
`--local-root` supports offline synthetic input. Prepared checkpoint evaluation uses
`--prepared ... --evaluate-checkpoint ... --split test --output ...` with the frozen resolved config.

The separate TF test suite is part of repository `make check` through `make tf-check`.
See Spec024 verification for host/platform limits; native Linux input/training acceptance passed on SageMaker; see Spec024 cloud-verification.md.


Manifest CLI now runs train → validation selection → selected-checkpoint test. Per-epoch JSON and TensorBoard are stored in training/epoch-metrics and training/tensorboard. cloud_train publishes completed epoch artifacts to an explicit run-scoped S3 URI; prepared input/cache are excluded. Dockerfile.train runs native acceptance with DOKU_MODE=verify before the first formal job.

## 每日训练与 SageMaker 发布（Spec 027）

日更入口 `python -m recommend_tf.train.daily_train_model` 使用 `online_config` 解析训练/验证窗口，与离线实验共用数据准备及训练循环，不执行 test。`export_model` 独立导出选定 checkpoint 与配套状态，提供四任务、无标签 SavedModel 签名。

`python -m recommend_tf.platforms.sagemaker.daily_pipeline` 供上游成功触发的一次性 ECS 调用，支持训练跟踪、导出、不可变 S3 发布、固定 Endpoint 一次性切换与回滚。必须提供明确环境绑定、受信任 ready 证据和独立 serving 仓库镜像。本仓没有 serving 实现，未部署真实 Endpoint。

命令、历史实验导出与跨仓合同见 [运行说明](../specs/027-tf-daily-serving/operations.md)，本地验证与尚待外部联调的边界见 [验证记录](../specs/027-tf-daily-serving/verification.md)。

## Spec 028：最小 Endpoint 交付

新增 `platforms/sagemaker/audit_run.py` 审核历史 run，`executor.py` 装配 ECS 输入与身份，`build_context.py` 生成干净镜像上下文。最小推理在 [inference/README.md](inference/README.md)，不导入训练内部实现。发布器复用 Spec 027，使用 container_environment 传入显式推理预算。

[操作说明](../specs/028-tf-endpoint-rollout/operations.md)、[任务](../specs/028-tf-endpoint-rollout/tasks.md)、[验证](../specs/028-tf-endpoint-rollout/verification.md)。本地通过不代表实际历史模型/Endpoint 已上线，云端环境装配和验收仍需执行。

## Cangen：drama 多目标双塔训练（Spec 029）

`model_type: cangen` 使用独立双侧 DNN、逐任务线性投影与逐行点积；保留真实曝光，
不构造 batch 内/随机负例。仅训练、评估、checkpoint，不生成向量文件或召回索引。
具体合同见 [需求](../specs/029-tf-cangen-training/requirements.md) 与
[设计](../specs/029-tf-cangen-training/design.md)。

模型配置见 [schema](train/models/cangen/config.schema.yml)，运行时由
`models/cangen/model_params.py` 校验。实验入口接受 YAML 或 JSON：

```bash
PYTHONPATH=. uv run --project recommend_tf --locked python -m recommend_tf.train.train_cangen \
  --config /absolute/path/cangen.yaml --output /absolute/path/new-run
```

配置沿用 Manifest 输入、histories、dtypes、item_filter、资源预算、日期及 early_stop；
不得省略必要实验配置或使用生产默认。`feature_roles` 显式标注原始字段为 user/item/label，
变换表达式传播依赖并拒绝跨塔输入；它是声明合同，不替代真实数据时间可用性审计。
共享 drama 表通过 embed_name 声明；两塔层和任务投影通过各自配置声明，不硬编码维度。

时长任务的 label 显式配置 `log1p(clip(watch_seconds, 0, 300))`，loss_type=mseloss、
predict_fn=softplus。`metrics.log_mse` 使用默认预测；秒数 MAE/RMSE 显式设置
`prediction: expm1` 和秒数标签表达式。`task_weight` 是任务系数，`loss_weight` 是样本权重，
二者不混用。模型正则、学习率调度与早停沿用公共实现；不默认启用可选正则。

合成四任务配置与入口证据见 `train/tests/cangen_fixture.py` 和 `test_cangen.py`；
这些是独立测试 fixture，不能直接当作生产字段配置。首次实验仍需声明真实字段、输入日期、
优化器及预算。恢复/独立评估继续使用统一 `train.run --prepared ... --resume ...` 或
`--evaluate-checkpoint ...`，并使用原冻结 resolved-config.json。

### Spec 030：日更选中模型监控

日更配置顶层必填稳定 `model_name`（例如 `tf_rerank_ln_lr001`），离线配置仍可缺省。
同名代表同一配方；另一个实验配方使用另一个名字。日期、Manifest 和 TrainingJobName 每日变化。
`training/selected-model-metrics.json` 只读取已有 selection 的 epoch，不重选、不重新评估。

发布 binding 可增加 `model_name` 和 `monitoring`：后者含 `outbox_uri`、`kms_key`、
`catalog_uri`，以及仅在指标别名有歧义时使用的 `metric_mapping`。主任务仍由 `outputs` 声明。
这些物理地址来自 IaC binding。监控失败不改变训练/Endpoint 发布结果；公共事件随成功 run
封存，monitoring 仓库负责补偿、目录和 Grafana 同步。

历史回填先执行 dry-run：

```bash
python -m recommend_tf.platforms.monitoring.backfill_metrics \
  --binding /absolute/path/binding.json --runs /absolute/path/runs.json
```

`runs.json` 是显式 `{run_uri, training_job_name?, historical_identity?}` 数组；历史身份须包含
原 `experiment-config.json` 的 `config_sha256` 及 model_name。可加 `--cutoff-from/--cutoff-to`
限制列表中的数据日期；确认报告后加 `--write` 写监控 outbox。此命令不训练、导出或部署模型，
也不改旧 run。Spec 030 代码当前仅本地验证，尚未替换生产训练镜像或监控看板。
