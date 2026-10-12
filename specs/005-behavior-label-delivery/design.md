# Spec 005：交付设计

Source: doku-offline fyp-rank-samples/contracts/feature-spec-v5.yaml 与本仓 PublishedDataset。

S3 标签 behavior_flags → 样本 v1 原样保留 → build-dataset v5 展开五列 →
contracts/features.json、contracts/labels.json、Parquet 和 Manifest → PublishedDataset → 配置任务。

v5 保留 v4 的 101 个特征、20 个审计字段和 13 个标签/观察列，再追加五个标签，总计 139 列。
训练 reader 依据 Manifest 和合同读取字段，已有实现支持新版本，无需增加特定行为分支。
mask 使用 not is_null(label)；不使用 video_finish_seen 或 immersive_click_valid，避免丢掉负例。
旧实验的全局 item_filter 为 video_finish_seen > 0，仍会作用于所有任务；复用该配置即继续
使用旧人口。若新实验要求包含未回收 finish 的行为样本，应显式调整 item_filter，并分别
声明播放任务自己的 mask。本次仅在合成 smoke 中设 item_filter=True，不改旧实验人口。
上游 label_definition_version 未变；仅列投影 feature_schema_version 升 v5。

新实验配置需固定已实际发布的 Manifest URI、dataset version、feature_schema_version、
label_definition_version 和 manifest_sha256，不能把旧 v4 Manifest 仅改名为 v5。
为每个标签添加二分类任务，神经任务使用 logloss/sigmoid，树任务使用 binary:logistic。
loss_weight 与 selection 权重由实验显式声明，本次不替用户调整旧实验。

例如神经模型任务（其余配置沿现有模型）：

```yaml
tasks:
  like:
    dims: [32]
    acts: ['relu']
    norms: ['']
    label: 'like'
    mask: 'not is_null(like)'
    num_outputs: 1
    loss_type: 'logloss'
    loss_weight: '1.0'
    predict_fn: 'sigmoid'
    metrics:
      auc: {func: 'auc'}
      logloss: {func: 'logloss'}
      valid_rows: {func: 'valid_rows'}
      positive_rate: {func: 'positive_rate'}
```

其余四项替换任务名、label 和 mask；数值权重仅为示例。不得把五个 label 加入 features。
未发布新数据集前，不创建带虚构 URI/摘要的可执行生产配置。

## 生产切换准备

当前 configs/model-comparison-20261001 的训练/验证/测试合计覆盖 2026-10-01 至 2026-10-08 UTC。
若按该范围接入新标签，先在授权环境发布/使用包含 v5 spec 的构建，在 doku-offline/fyp-rank-samples
执行以下一次性构建（会写生产 S3，尚未执行）：

```bash
AWS_PROFILE=doku-prod-developer DOKU_ENV=prod uv run fyp-rank-samples build-dataset \
  --environment prod --from 2026-10-01 --to 2026-10-08 \
  --feature-spec contracts/feature-spec-v5.yaml \
  --dataset-version behavior-v5-20261012-01
```

使用镜像中已安装的 CLI 时省略 --feature-spec，默认 v5；镜像内没有源码 contracts 相对路径。
输入为各小时现有样本 _SUCCESS 所指向的 run；若需严格复现旧实验人口，应在构建前比较输入版本
和行数，不能假定当前指针仍与原 v4 构建时一致。若缺小时则构建失败，不跳过缺失分区。

产出前缀为 `ml/fyp-rank-train/fyp-rank-features-v5/behavior-v5-20261012-01/`。
构建成功后核验每项非空数和正例数、列 schema、行数及摘要，然后在新的实验配置中使用返回的
Manifest URI/version/content_sha256。禁止手填或猜测摘要，也不覆盖旧实验配置。
新生产数据构建与模型训练是两个独立动作；本次未启动真实数据训练或发布模型。
