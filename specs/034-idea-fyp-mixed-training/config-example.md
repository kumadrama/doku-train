# 多源配置片段（示意，不是生产实验配置）

在现有 TF envelope 上扩展；日期、执行预算、输入特征、网络和标签定义沿用显式配置。

```yaml
data_sources:
  - name: fyp
    source_id: 0
    format: doku-train-dataset-manifest/v1
    manifest_uri: 's3://<bucket>/<fyp-daily-prefix>/date={date}/_SUCCESS'
  - name: idea
    source_id: 1
    format: doku-train-dataset-manifest/v1
    manifest_uri: 's3://<bucket>/ml/idea/train/daily/environment=<env>/date={date}/_SUCCESS'
item_filter: '(datasvr_source == 0 and stable_hash(u_account_id) % 50 == 1) or datasvr_source == 1'
shuffle_buffer_size: 65536
seed: 42
model:
  tasks:
    idea_read_more:
      loss_weight: '1.0 * (datasvr_source == 1)'
      loss_reduction: sum
    idea_watch_now:
      loss_weight: '1.0 * ((datasvr_source == 1) and (placement == "WATCH_NOW"))'
      loss_reduction: sum
    idea_vote:
      loss_weight: '1.0 * ((datasvr_source == 1) and (placement == "REGULAR"))'
      loss_reduction: sum
  eval_tasks:
    idea_read_more:
      metrics:
        auc: {func: auc, filter: 'datasvr_source == 1'}
    idea_watch_now:
      metrics:
        auc: {func: auc, filter: 'datasvr_source == 1 and placement == "WATCH_NOW"'}
    idea_vote:
      metrics:
        auc: {func: auc, filter: 'datasvr_source == 1 and placement == "REGULAR"'}
```

示例省略必需的模型头、label、dtypes、默认值等，不可原样运行。`u_account_id` 是 FYP 投影的账号字段，作为采样输入显式声明，不自动加入模型特征；Idea 无该列时显式配置默认值 0。placement 缺失时采用显式空字符串默认值，不推断为 REGULAR。标量特征默认值为 0，序列默认值遵守其形状合同（如空列表）。

Source: doku-offline/fyp-rank-samples/tests/unit/test_dataset.py 验证 u_account_id 投影；2026-11-05 用户确认按用户稳定哈希后取模。实际输入合同必须包含该采样列，否则补齐上游投影，不能以默认 0 静默替代 FYP 账号。

原示意 shuffle 对象收敛为现有 `shuffle_buffer_size` / `seed` 键；多源自动启用交错，每 epoch 使用 seed+epoch，无重复配置。

运行时 dtypes 的字符串使用 `str`（多源也接受 `string` 并规范化）；源合同仍使用 string。所有缺列（含跨源标签占位）都需在 feature_default_values 显式配置。多天可将 manifest_uri 替换为 manifest_uris 精确 URI 列表，二者互斥。

### 2026-11-05 单入口按日期解析

每个来源只需 manifest_uri。精确 Manifest 地址行为不变；需要按日期自动解析时使用上述含 `{date}` 的入口模板，日期来自 dataset_config.split 的 train/validation/test 范围或显式 dates。也支持直接指向 date={date}/run_id=固定版本/manifest.json 的模板。无需每日手填列表。
读取 _SUCCESS 后校验 manifest_sha256、日期与不可变运行路径，逐来源冻结解析结果、指针原文和 Manifest/合同。恢复读取冻结记录，不再读取可变指针。缺训练日期静默跳过来源；缺 validation/test 日期只跳过对应日期。没有默认生产桶或自动发现 latest。旧 manifest_uris 仍可用，但不作为日常推荐配置。
