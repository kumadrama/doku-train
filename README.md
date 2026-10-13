# doku-train

Doku 推荐精排的单机 CPU 训练仓库。运行时读取公共和模型 YAML schema，验证独立实验配置。

- `models/rerank`：动态任务 DNN；`fm_layers` 配置 Finder 式点积交互后成为 DeepFM 实验。
- `models/rerank_tree`：按 task 构造 CPU hist Booster，支持分类和时长回归。
- `comm`：Manifest、特征、时间切分、指标、产物和搜参编排，不引用具体模型实现。

默认值只作用于原始 features；转换表达式可声明 log 缩放、数值分桶和 embedding。不自动添加缺失标记。
共享词表仅训练集拟合。没有固定四目标、27/1/28、refit、旧格式兼容或指标质量门禁。

## 本地使用

```sh
uv sync --locked
uv run python -m recommend.train.run validate --config /absolute/path/to/config.yaml
uv run python -m recommend.train.run search --config /absolute/path/to/search.yaml --output /tmp/doku-search-01 --dry-run
uv run python -m recommend.train.run train --config /absolute/path/to/config.yaml
uv run python -m recommend.train.run test --run-dir /absolute/path/to/completed-run
uv run python -m recommend.train.run report --run-dir /absolute/path/to/dnn-run --run-dir /absolute/path/to/tree-run --output /tmp/final-report.md
make check
```

`--local-root` 可指定精确发布前缀的本地镜像，仍校验源Manifest和分片摘要。不会发现latest，
不会从当前业务存储补特征，不写PROD。输入身份和范围在配置显式声明，输出仅本地create-only目录。

实验 YAML、结果报告、权重与运行产物不随源码发布。测试独立生成合成数据和配置，
不依赖本地实验文件或生产访问权限。字段合同见 [配置说明](specs/004-three-model-comparison/config-contract.md)。

搜参每个trial独立进程，仅加载本模型框架，避免macOS多份OpenMP共进程冲突。DNN和DeepFM以
配置文件名独立报告，不能用共同的model_name合并身份。搜索结束只出验证报告，由用户决定
是否对其中一个模型精搜；最终测试由显式test命令执行，不参与选模。

成功产物包含checkpoint、metrics、feature state、lineage和完成状态；目录另含展开配置、
可恢复神经训练的resume.pt及本地编码缓存。仅有文件、没有complete状态的目录不是完整产物。
`train --resume /path/resume.pt` 在新的run_id目录恢复神经训练，要求输入、配置和源码身份一致。
树暂不支持中途轮次恢复，会明确拒绝该请求。Checkpoint只作训练格式，不是在线推理合同。

更多设计及验证见 [Spec004](specs/004-three-model-comparison/design.md) 和
[验证记录](specs/004-three-model-comparison/verification.md)。

### 行为标签训练数据 v5

点赞、评论、分享意图、关注和追更订阅通过 doku-offline `fyp-rank-features/v5`
的 `like/comment/share/follow/subscribe` 列交付，接入方式见
[Spec 005](specs/005-behavior-label-delivery/design.md)。使用新 Manifest 的真实身份与摘要，
各任务以 `not is_null(label)` 为 mask；不能把上游缺失填成负例，也不能只保留正例。
五列属于标签，不能加入模型输入 `features`。已有 v4 配置继续用于原实验。

## 每日投影数据输入

按日输入与冻结契约见 [Spec 006](specs/006-daily-projected-input/design.md)。将 dataset 中
`manifest_uri`、`manifest_sha256`、`version` 替换为：

```yaml
daily_prefix: s3://<data-lake>/ml/fyp-rank-train/daily/environment=prod/
projection_sha256: <由发布 manifest 提供的规范投影 SHA256>
```

其他 manifest/feature/label 版本、AWS profile、adapter 和 allowed_prefixes 仍显式配置。
`feature_schema_version` 与所有选中日期一致，日期沿用 split。只有已发布 `_SUCCESS` 的日期可用。
训练会在 run/input/ 固定 manifest 与契约，之后使用不可变分片，不复制训练数据。
`resolved-config.json` 可用于重现；`--resume` 使用原 checkpoint 对应的冻结输入。
首次切换前需由离线任务发布所需历史日期；之前的整段 manifest 继续有效。
