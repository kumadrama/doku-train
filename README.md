# doku-train

Doku 推荐精排的单机 CPU 训练仓库。运行时读取公共和模型 YAML schema，验证独立实验配置。

- `models/rerank`：动态任务 DNN；`fm_layers` 配置 Finder 式点积交互后成为 DeepFM 实验。
- `models/rerank_tree`：按 task 构造 CPU hist Booster，支持分类和时长回归。
- `comm`：Manifest、特征、时间切分、指标、产物和搜参编排，不引用具体模型实现。

默认值只作用于原始 features；转换表达式可声明 log 缩放、数值分桶和 embedding。不自动添加缺失标记。
共享词表仅训练集拟合。没有固定四目标、27/1/28、refit 或旧格式兼容；训练不设质量门禁，发布门禁由 YAML 声明。

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
不会从当前业务存储补特征，不写PROD。输入身份和范围在配置显式声明，训练输出为本地 create-only 目录；显式 daily/publish 命令可写入指定 S3 前缀。

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

## Dataset 输入

```yaml
dataset:
  manifest_uri: 's3://<data-lake>/<frozen-input>/manifest.json'
```

日期在 split 配置。版本、摘要与合同由读取层从 Manifest 校验，不在 YAML 重复声明。
使用 AWS_PROFILE 或运行环境角色提供访问身份，例如
`AWS_PROFILE=<profile> uv run python -m recommend.train.run train --config <config.yaml>`。
日更目录须先产出冻结 Manifest；训练不解析可变 daily 指针。
每轮保存 run/input/ 下的 Manifest、合同和摘要，resume/test 使用这份冻结输入。
独立新训练会重新冻结源地址。详见 [Spec 007](specs/007-manifest-only-dataset/design.md)。

## 树模型每日发布

Source: [Spec 009 需求](specs/009-daily-tree-model-publication/requirements.md)、
[设计](specs/009-daily-tree-model-publication/design.md)、
[验证](specs/009-daily-tree-model-publication/verification.md)。

已完成 PROD S3 日更发布和清理 dry-run 接线（Spec 009）；本仓新增可选 Redis 分发
（Spec 010），尚未部署该功能。两个服务后台加载模型、请求使用同一已就绪内存版本。
S3 published、Redis committed 与消费者 ready 分别记录；加载、RPC 和 pin 由 mono 接入。

配置迁移：原 `model_name: rerank_tree` 改为 `model_type: rerank_tree`，新增用户自定
`model_name`；不保留旧字段语义兼容。以下是追加到完整模型配置的字段片段，非完整配置：

```yaml
model_type: rerank_tree
model_name: recommendation_tree
daily:
  training_window_days: 30
# 每个 task 显式声明 output_unit，并在 metrics 下声明发布指标。
# 分类 func: auc，回归 func: regression_auc；任务名及 metric 键由配置定义。
publication:
  completion: {metric: auc, minimum: 0.60}
  watch_seconds: {metric: regression_auc, minimum: 0.60}
```

`publication` 必须覆盖全部配置任务；每个 task 的 `output_unit`（如 probability/seconds）
与 `predict_fn` 描述输出；示例任务不能当作框架固定任务。daily 将模板 split 解析成明确的
train_dates/validation_dates/test_dates，三段各至少一天，缺日允许跳过。

```sh
# 由 iac 在上游投影成功后执行一次。根路径、日期和 AWS 身份必须显式提供。
uv run python -m recommend.train.run daily --config /absolute/path/model.yaml \
  --daily-root 's3://<data-bucket>/<daily-prefix>/environment=<env>' \
  --frozen-root 's3://<output-bucket>/<frozen-prefix>' --cutoff YYYY-MM-DD \
  --artifact-root 's3://<output-bucket>/<model-prefix>' --export-root /absolute/path/exports

# 独立评估与导出；test 可重跑，checkpoint 不变。
uv run python -m recommend.train.run test --run-dir /absolute/path/run
uv run python -m recommend.train.run export --run-dir /absolute/path/run --output /absolute/path/exports
uv run python -m recommend.train.run publish --package /absolute/path/exported-version \
  --artifact-root 's3://<output-bucket>/<model-prefix>'

# 默认 dry-run；--apply 仍受 consumer_protocol_ready、current、pin 和过渡期保护。
uv run python -m recommend.train.run cleanup --model-name recommendation_tree \
  --artifact-root 's3://<output-bucket>/<model-prefix>'
```

本地替身使用 `file:///absolute/path` 根目录，走同一发布状态机。`--force` 可显式重跑相同
输入；同名 active_run 不自动超时释放。进程崩溃后，确认停止才能用 `recover-run --owner
<id> --confirmed-stopped` 释放；不要在原进程仍运行时恢复。已发布的同一版本不重写。
`rollback --version <id>` 先校验保留文件，恢复版本引用；不允许回滚 deleting/deleted。

`pin --version <id> --owner <instance-boot-id>` 登记，排空后加 `--release` 释放。
消费者全部实现该协议后，由运维显式执行 `consumer-protocol-ready` 开启自动删除能力；
这些命令均需 `--artifact-root` 和 `--model-name`，不能只凭失联判定 pin 失效。
清理带 `--run-root` 与 `--export-root` 时，仅删除远端已 tombstone 的本地对应目录；未知
目录保留。上游数据与共享冻结 Manifest 不由模型清理程序删除，避免破坏其他 run 的引用。

上传中断恢复：独立 `publish` 可加 `--run-dir /absolute/path/original-run`，补齐原先登记但
尚未上传的训练归档；不提供目录时只可复用远端已存在且摘要一致的归档。manifest 最后写入，
发布清单提交前校验完整对象集。清理按 run_id 保护共享训练目录，并与本地 test/export
读锁协调；不要通过复制或复用已退休的 run_id 绕过这些保护。

生产调度由 iac 衔接：PROD 日投影成功后执行 `python -m recommend.train.scheduled daily --cutoff YYYY-MM-DD`，
每天北京时间 20:00 执行 `python -m recommend.train.scheduled cleanup`。DTE 不部署对应调度。
容器使用环境显式绑定 `DOKU_TRAIN_{ARTIFACT_ROOT,MODEL_NAME,S3_KMS_KEY_ARN}`；daily 另需
`DOKU_TRAIN_{DAILY_ROOT,FROZEN_ROOT,CONFIG_URI,CONFIG_SHA256}`。配置先校验 SHA256 和模型身份，
不使用可变 latest 配置。实际实验 YAML 保留本地，部署前按 iac 声明的摘要上传；不打进源码镜像。
镜像及 task definition 入口见 `build/` 和 `deploy/prod/`；`./build/build.sh` 仅打包源码和依赖锁，
`./build/docker-build.sh <image>` 构建 CPU 镜像。跨仓接线与上线顺序见 Spec 009 设计。


## 可选 Redis 分发

Source: [Spec 010 设计](specs/010-redis-model-distribution/design.md)、
[验证](specs/010-redis-model-distribution/verification.md)。

`run.py` 统一编排训练、测试、导出、发布；`publish_model.py` 独立装配 S3 发布与 Redis 分发，
不触发训练或导出。`python -m recommend.train.publish_model --package ... --artifact-root ...`
与 `run publish` 使用同一用例。

由部署环境显式注入以下绑定，不能把凭据写进 YAML 或提交到仓库：

| 环境变量前缀 `DOKU_TRAIN_REDIS_` | 值 |
| --- | --- |
| SECRET_ID | 平台 Redis Secret 名称，运行时读取，和 URL 二选一；PROD 采用此项 |
| URL | TLS Redis URL（rediss，凭据运行时注入）；仅本地测试允许 loopback redis |
| ENVIRONMENT | 安全的环境标识，参与键前缀 |
| CLUSTER | `true` 或 `false`，无默认值 |
| TIMEOUT_SECONDS | 正数，连接/命令超时 |
| MAX_FILE_BYTES | 单个对象预算，包括读取上限 |
| MAX_PACKAGE_BYTES | 单版本 serving 文件、descriptor、ready 总字节预算 |
| MAX_MODEL_BYTES | 同模型所有未删除版本的保留字节预算，含未完成上传 |

未配置时继续 S3-only；部分配置报错。首次同步将该模型的 registry 升为 v2，旧发布/清理
二进制不支持 v2，须先停止旧任务并升级；v2 不允许因缺少 Redis 绑定而静默降级。

以下管理命令都要求 `--artifact-root <root>`、`--model-name <name>` 和上述绑定：

- `redis-sync --version <exact-version> --stage-only`：验证既有 S3 包并上传，不切换目录。
- `redis-sync --version <exact-version>`：同步 S3 current，完整校验后原子切换目录。
- `redis-resume --operation-id <id> --confirmed-stopped`：旧执行者及其未决请求已停止后恢复。
  pending 期间 Redis 丢失/回退时加 `--rebuild-redis`：以新 epoch 重建原操作完整目标目录，
  保持原发布/回滚/排空意图；shadow 不变成上线，删除先提交已撤销旧引用的目录。
- `redis-cancel --operation-id <id> --confirmed-stopped`：仅取消尚未提交目录的 sync/stage；
  已提交、未知结果、回滚或清理必须 resume，已登记半包保留供恢复/合规清理。
- `redis-rebuild --confirmed-stopped`：确认旧写入者停止后，从 S3 重建权威引用并更换 epoch。
- `rollback --version <exact-version>`：用新 generation 同时协调 S3 与 Redis 回滚。
- `cleanup`：默认 dry-run。`--apply` 还要求 consumer-protocol-ready 和
  distribution-protocol-ready 两个门禁；只有完成 mono 双服务引用验收后才显式开启。

重复输入去重仍检查成功版本的 Redis 分发；被门禁拒绝的 run 不同步。若存在未完成操作，
日更明确报失败并给出 operation_id，运维确认终止后恢复，不能自动抢占。迟到旧输入不覆盖
后续发布或人工回滚。Redis 失败不撤销 S3 发布，不伪称完整分发成功。

清理保护两侧 current 和持久 pins；版本超过 30 天、替换保护期结束后，先从两侧日期目录
撤销并进入 draining。两侧撤销确认后再等至少 24 小时且 pins 清空，先删 Redis，再删 S3。
不设 TTL，不扫描全库删除；Redis 状态未知或有未完成操作时停止清理。每次优先完成
已到期删除，避免日更排空候选让删除永久等待；逐项删除及转入 S3 删除前复核目录。

iac 后续装配专用容量、TLS/ACL、noeviction、网络与上述任务绑定；mono 实现一次解析版本、
RPC 携带精确版本/摘要、两服务独立 pin 与排空。先 stage-only，再验证完整请求，最后接通
日更同步和双清理门禁。DTE 不部署。本轮没有修改 PROD 或开启任何门禁。

本地真实 Redis 验证使用 `DOKU_TEST_REDIS_URL=redis://127.0.0.1:<port>/0 make check`；
CI 自带隔离 Redis 服务。未提供该变量时仅跳过两项真实 Redis 测试，其余故障注入测试照常运行。
