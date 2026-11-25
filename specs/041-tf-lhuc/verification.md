# Spec 041 验证

日期：2026-11-20。环境：本机 macOS ARM64，Python 3.10 / TensorFlow 2.11.0 CPU，
intra/inter-op 线程均为 1。

仅本地合成数据；不代表模型效果或生产部署验收。

## 来源复用

通过 Python `ast.parse` / `ast.dump` 比较本仓和 Finder 本地源码的 `dnn_layer`、
`target_layer`，两个函数 AST 均完全一致（忽略源码行号/格式）。迁移修改仅接通
`rerank_core_model` 的条件输入及开关。来源函数摘要按完整函数源码（含末尾换行）
写入 `recommend_tf/source-inventory.json`，与原有来源登记的计算方法一致。

## 专项验证

```sh
env PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  recommend_tf/.venv/bin/python -m pytest \
  recommend_tf/train/tests/test_lhuc.py \
  recommend_tf/train/tests/test_migration.py \
  recommend_tf/train/tests/test_inference_contract.py \
  -q -p no:cacheprovider --disable-warnings
```

结果：**55 passed, 171 warnings in 46.65s**。

| 要求 | 可执行证据 |
| --- | --- |
| 配置类型、重复组名、未知输入拒绝 | `test_lhuc.py::test_invalid_config`（9 组） |
| 公式、拼接顺序、多层、FM、多输出任务及梯度 | `test_core_matches_finder_formula_and_condition_gradient`（4 组）：NumPy 公式与条件输入有限差分 |
| share_embed 与任务 stop_grad | `test_gradient_boundaries_preserve_main_path_and_trainable_gates`（4 组）：主干/条件输入/各层参数梯度 |
| 门控 kernel 正则、EVAL 变量复用 | `test_regularization_and_eval_variable_reuse` |
| 无隐藏层任务的 logits 前门控 | `test_empty_hidden_task_still_gates_logits` |
| 关闭兼容 | `test_disabled_matches_original_graph_and_one_update`：与原 core 调用比较变量名、loss、预测、梯度及一步更新，逐项完全一致 |
| CPU 七任务训练、LN/L2、恢复、损坏分片拒绝 | `test_migration.py::test_prepared_integrity_shuffle_and_seven_task_train[lhuc]` |
| 成功训练产物与导出 parity、空历史/OOV | `test_inference_contract.py::test_raw_request_matches_training_tensors_and_savedmodel[True]`：条件专用 context 从主干移除后仍出现在签名；export adapter 内对 EVAL/PREDICT/SavedModel 比较 |

## 仓库统一检查

`make check` 初次因沙箱无法读取既有 uv 缓存退出；获得工具执行权限后重新运行。
结果如下：

- 主项目及 TF 锁文件检查通过。
- Ruff：414 个文件格式检查通过，lint 通过。
- mypy：70 个源码文件通过。
- 主训练栈：**329 passed, 3 skipped in 180.72s**。
- TF 全量：**250 passed, 243 warnings in 254.02s**。
- inference 初次 29 项中 1 项失败：Idea 本地清单摘要仍指向变更前源码。
  此为既有源码包一致性保护，修复方式为 README 指定的本地 `--pin`，未绕过测试。

```sh
python3 idea-mapped-daily/build/package_source.py --pin
.venv/bin/python -m unittest discover -s recommend_tf/inference/tests -p 'test_*.py' -v
```

结果：源码包 73 个文件、101700 字节，SHA256
`b23e1126e959401ad3d3460095a14745d106bf9d461a65a411b6a469bed1ad9b`；
inference **29 项全部通过**。清单只更新 `ARTIFACT_SHA` 与 `ARTIFACT_KEY`，
没有云端上传或部署。训练源码未再变化，因此没有重复执行已通过的训练套件；
`make check` 整体初次返回失败，修复后其失败的 inference 阶段已单独重跑通过。

补充：`git diff --check` 通过；163 个相关文档链接与 source-inventory JSON 检查通过。
原有 Spec 009 两份未提交修改保持原样，不属于本次 LHUC 交付。

## 国家专用配方（2026-11-20 后续要求）

Source: 用户要求“把国家这个特征加入lhuc 其他特征不加”。

生成本地 `runs/lhuc-country-20261120/config.yaml`，参数校验通过；来源摘要和新配置摘要
在同目录 `source.json`。执行断言确认：LHUC 展开后字段恰为 `[ctx_ip_country]`，
国家仅存在于一个输入组，拆分前后主干字段序列完全一致，其他非模型配置保持不变。
实验配方已确认被 Git 忽略，未修改原基线快照或提交训练任务。

```sh
env PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests/test_lhuc.py \
  -q -p no:cacheprovider --disable-warnings
.venv/bin/ruff format --check recommend_tf/train/tests/test_lhuc.py
.venv/bin/ruff check recommend_tf/train/tests/test_lhuc.py
git diff --check
```

结果：**21 passed, 145 warnings in 4.80s**，格式/lint/diff 检查通过。
新增 `test_country_only_gates_ignore_platform_and_country_still_learns_from_main_path`：
固定合成国家及平台 embedding，平台不同、国家相同的门控值完全相同；
国家变化时三处门控都变化，所有门控 kernel 输入宽度只等于国家 embedding 维度。
门控到国家 embedding 的梯度被阻断，国家 embedding 从主干收到非零任务损失梯度。
本轮只改本地配方、测试与文档，模型运行源码/源码包摘要保持不变。

## 指定 other50 基线的单次云训练启动

Source: 用户明确授权基于 `doku-exp-fyp-us-other50-20261120-0a923a14` 改为全国家过滤、
国家 LHUC 并启动一次训练。此处是后续授权的云实验记录，不是先前本地验证的部署声明。

- 2026-11-20 15:59:58（Asia/Shanghai）创建
  `doku-exp-fyp-all-lhuc-20261120-440fa353`。
- 实际任务及 S3 recipe 与本地基线核验一致；配置只调整 `item_filter` 为
  `video_finish_seen > 0` 及国家 LHUC 接线。新的 JobName、输出路径、Manifest 副本路径
  和源码校验入口为此次独立运行的元数据，不改变原冻结分片。
- 保留原模型标识、30 epoch、batch 2048、学习率 0.002、单机 ml.m5.4xlarge、100 GB
  磁盘和 86400 秒上限；冻结 train 2026-10-19～2026-11-17，validation 2026-11-18。
  LHUC 输入仅 `country/ctx_ip_country`；share_embed=true，target_layer_use_lhuc=true。
- 沿用原镜像 digest `5e9ef04832e66350684c086061ec4c5dd9c748e51df05dd2c28269e9ff20da94`。
  CloudWatch 已记录 `lhuc_runtime_verified`，原源码摘要、依赖锁摘要、三文件替换前后摘要
  全部验证成功；实际覆盖后训练源码摘要为
  `2fab4aa537427a07690fa7245ef743f84e0825b86844ec1777a4d326c45dff35`。
- 最新核验为 `InProgress / Training`，无 FailureReason；日志出现
  `{"event":"prepare_split","split":"train"}`。这是成功启动与开始数据准备的证据，
  不代表 epoch 已结束或完整训练成功。
- 首次 CreateTrainingJob 请求因 ContainerArguments 超过 256 字符被 AWS 拒绝，未创建任务；
  缩短等价摘要校验入口至 214 字符后，同一唯一任务名提交成功，没有启动第二个任务。

执行与结果：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-all-country-lhuc-20261120/prepare.py
env PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  recommend_tf/.venv/bin/python runs/fyp-all-country-lhuc-20261120/verify.py
env PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests/test_lhuc.py \
  -q -p no:cacheprovider --disable-warnings
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-all-country-lhuc-20261120/submit.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-all-country-lhuc-20261120/status.py
```

本地过滤用例保留 US/空/CA/GB/缺失国家、排除无效曝光；覆盖包正常应用以及篡改覆盖文件、
基线源码时拒绝的测试通过。LHUC **21 passed, 145 warnings in 4.24s**。
`submit.py` 成功返回 TrainingJobArn，`status.py` 返回上述状态与源码校验日志。
完整配置、不可变输入副本、请求、回执及日志保存在本地
`runs/fyp-all-country-lhuc-20261120/`；未修改日更、Endpoint 或发布线上模型。

## Context + country 对照实验启动（2026-11-20）

Source: 用户要求“再开一个训练任务，把context和country都加入lhuc”。

- 新任务：`doku-exp-fyp-context-country-lhuc-20261120-c4e222a3`。
- 基线：`doku-exp-fyp-all-lhuc-20261120-440fa353`。
- 唯一训练行为差异：`model.lhuc_inputs` 从 `[country]` 改为 `[context, country]`；
  条件字段按序为 ctx_platform、u_deep_link_drama_id、u_adjust_network、ctx_ip_country。
- 全国家过滤、冻结输入、主干、依赖镜像与源码覆盖包、30 epoch、batch 2048、LR 0.002、
  ml.m5.4xlarge、100 GB、stop-gradient=true、任务塔开启均保持不变。
- 新任务使用独立名称和产物路径；原任务未停止或修改。

执行：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-context-country-lhuc-20261120/prepare.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-context-country-lhuc-20261120/submit.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-context-country-lhuc-20261120/status.py
git diff --check
```

结果：参数校验、条件字段列表、还原唯一变更后与基线相等、冻结输入及源码覆盖包字节
一致性断言通过。提交前远端基线 recipe 核对通过，CreateTrainingJob 返回成功回执。
真实状态为 **InProgress / Training**，FailureReason 为空；日志含
`lhuc_runtime_verified` 与 `prepare_split/train`，实际源码摘要与前一次国家 LHUC 实验相同。
此为成功启动和数据准备证据，尚不声明 epoch 完成。原始配置/请求/日志保留在本地
`runs/fyp-context-country-lhuc-20261120/`。

## 七模型 TensorBoard 对比（2026-11-20）

Source: 用户要求比较两组 LHUC、e16、mlp256、e16+mlp256、other50 与美国 base 的逐 epoch
指标，并强调训练与验证日期必须对齐。

- 美国 base 选取日更 `doku-daily-9472383918b6e0f358a26101eae7e103752a78be`，实际过滤为
  US 或未知国家；不使用验证日期 11-17 的旧实验。
- 7 个实际提交配置 SHA256 核验通过，输出实验配置与 resolved config 日期核对一致。
  全部为 UTC train 2026-10-19～2026-11-17，validation 2026-11-18；冻结 Manifest
  分片列表、特征版本、标签版本和合同摘要相同。
- 下载原始已关闭的逐 epoch TensorBoard 事件与指标 JSON，保留全部标签。完成任务的
  元数据和指标文件通过 COMMITTED 摘要验证；其余标记为运行中部分结果。
- 初次同步可用轮数：美国 base 30、other50 24、country-LHUC 11、context-country-LHUC 10、
  e16 11、mlp256 10、e16-mlp256 9。每个 run 56 条曲线，共 5880 个标量点。
- EventAccumulator 与 TensorBoard HTTP API 的全部曲线 epoch 序号及数值均与源 JSON
  一致（rtol=1e-6、atol=1e-7）。美国子集 auc_us/mse_us 定义一致，各任务每轮 52777 行。
  普通验证指标分别使用 52777/126271/199530 行，保留原统计口径并显式说明差异。
- 对比目录：`runs/fyp-seven-model-comparison-20261120/`；服务地址
  `http://127.0.0.1:6012/`。本地刷新进程每 120 秒只读同步这些任务，全部终态后退出；
  TensorBoard 每 15 秒重载文件。本次没有启动训练或修改云资源。

执行证据：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-seven-model-comparison-20261120/sync.py
.venv/bin/python runs/fyp-seven-model-comparison-20261120/verify.py
```

均通过。详情见本地 `alignment-and-status.json`、`tensorboard-verification.json` 与
`refresh-status.json`；README 记录曲线说明和运行到任务名的映射。

### 增加日期对齐的全量日更 base

Source: 用户随后要求找到相同日期的全量样本日更 base，并加入已有 TensorBoard。

已找到 `doku-daily-31f5bbcb6d4ef18cfd36b4549a63c881fe632b2a`，状态 Completed，
30 个 epoch 的指标和事件文件全部存在并通过 COMMITTED 摘要验证。
新增 run 为 `08_All_base_daily`，继续使用 `http://127.0.0.1:6012/`。
8 组的 UTC 训练/验证日期、冻结分片及合同仍完全一致。新增 base 的过滤为
`video_finish_seen > 0`，验证样本 199530 行，与其余全量实验相同；无 auc_us/mse_us，
保留缺失而不补造。其 49 条曲线、1470 个点已逐项核对 HTTP/事件/源 JSON 相同。

再次执行上述 sync.py 与 verify.py 均通过。既有刷新进程曾遇 AWS SSO SSL 错误，
切换到与成功手动同步一致的 TF Python 环境后，已完成一次全部 8 任务同步，状态 running。
新增本地文件锁避免重叠同步写入；原训练任务和产物未修改。

### 全量日更 base 美国子集补评估（2026-11-20）

Source: 用户要求测量 `08_All_base_daily` 的 auc_us。

- 原任务的 30 个 checkpoint 和冻结配置、词表、合同通过 COMMITTED 摘要验证；
  复用训练后的词表和 validation 2026-11-18 原分片，仅进行推理，无权重更新。
- 复用其他 7 组的 US 指标定义，过滤 `ctx_ip_country == 'US'`；每轮各任务均为
  52777 行。训练日期继续为 2026-10-19～11-17，与既有对比一致。
- 本地 Python 3.10.20 / TF 2.11.0 与原任务 Python 3.10.22 / TF 2.11.1 存在环境差异，
  两侧源码与依赖身份分别记录。第 1 轮全量 199530 行的全部原验证指标复现通过
  （rtol=1e-5、atol=1e-6），预筛 US 与全量预测后筛 US 的指标也通过相同容差校验。
- 全部 30 轮已完成，新增六项 auc_us 与 watch_seconds/mse_us，合计 210 点；
  以独立本地事件文件加入现有 run。原 S3 产物与原生事件未修改。
- EventAccumulator 和 TensorBoard HTTP 的 8 组全部曲线逐点核验通过；
  `08_All_base_daily` 现在共 56 条曲线、1680 个标量点，美国子集无缺失。

实际执行且成功：

```sh
recommend_tf/.venv/bin/python runs/fyp-all-base-us-eval-20261120/fetch.py
env PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 recommend_tf/.venv/bin/python runs/fyp-all-base-us-eval-20261120/evaluate.py
.venv/bin/python runs/fyp-all-base-us-eval-20261120/add_tensorboard.py
.venv/bin/python runs/fyp-seven-model-comparison-20261120/verify.py
git diff --check
```

本地证据：`runs/fyp-all-base-us-eval-20261120/` 下 `download-verification.json`、
`full-validation-parity.json`、`epoch-metrics/`、`result.json`、`tensorboard-receipt.json`；
对比目录的 `tensorboard-verification.json` 记录逐点验证结果。

### 国家 LHUC 容量三实验（2026-11-20）

Source: 用户要求基于 03，分别复用 06 的 MLP、05 的 embedding 和组合并提交训练。

| 变体 | SageMaker 任务 | 提交后状态 |
| --- | --- | --- |
| MLP [256, 128] | doku-exp-fyp-lhuc-mlp256-20261120-1282ca20 | InProgress / Training |
| drama embedding 16 | doku-exp-fyp-lhuc-e16-20261120-1db86bf5 | InProgress / Starting |
| 两者组合 | doku-exp-fyp-lhuc-e16-mlp256-20261120-92af459b | InProgress / Starting |

三个配置均通过 parameters 校验；还原指定 MLP/embedding 配置后与 03 基线完全相同，
重定位后的冻结输入及源码覆盖包（除 config.json 和 input/record.json 地址）字节一致。
03/05/06 的远端原始提交配置与 SHA256 验证通过，03 的全部 recipe 文件远端逐字节核对通过。
三个新任务使用同样的镜像、源码覆盖包、资源和训练日期，各自固定名称、独立产物前缀。
CreateTrainingJob 均返回成功回执，describe 的配置、资源、输入、镜像、角色与预期一致；
显式纳入 SageMaker 返回的默认 CompressionType / RecordWrapperType / MetricsTimeSeries 字段。
首个任务的严格校验曾因这些默认字段暂停，修正后按固定名称复查继续，没有重复创建任务。
最新三任务 FailureReason 均为空；MLP 实验日志包含 lhuc_runtime_verified 和 prepare_split/train，
另两个尚在 Starting。本记录验证成功提交，不声明 epoch 已完成。

执行并通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-country-lhuc-capacity-20261120/prepare.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-country-lhuc-capacity-20261120/submit.py
recommend_tf/.venv/bin/python runs/fyp-country-lhuc-capacity-20261120/mlp256/status.py
recommend_tf/.venv/bin/python runs/fyp-country-lhuc-capacity-20261120/e16/status.py
recommend_tf/.venv/bin/python runs/fyp-country-lhuc-capacity-20261120/e16-mlp256/status.py
git diff --check
```

配置、request、verification、submission、status 与日志在上述本地目录各变体子目录中。

### 容量三实验完成及 TensorBoard 对比（2026-11-22）

Source: 用户要求查看三任务是否完成，并将逐 epoch 指标加入原 TensorBoard。

SageMaker describe 确认三任务均 Completed，FailureReason 为空，均完整 30 epoch。
北京时间 2026-11-21：mlp256 于 04:53 完成，e16 于 04:40 完成，组合于 05:09 完成。
原对比目录新增 09_All_country-LHUC_mlp256、10_All_country-LHUC_e16、
11_All_country-LHUC_e16-mlp256，沿用 6012 端口。11 个模型全部同步到 30 轮。

训练日期 2026-10-19～11-17、验证日期 11-18、冻结 Manifest 分片及特征/标签合同完全一致。
完成产物的元数据、指标、事件均通过 COMMITTED 摘要验证。每组 56 条曲线、1680 点，
共 18480 点；源 JSON、EventAccumulator、TensorBoard HTTP 的 epoch 与数值均一致，
rtol=1e-6、atol=1e-7。所有 run 的 auc_us/mse_us 定义一致，每任务每轮 52777 行。

实际执行并通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-seven-model-comparison-20261120/sync.py
.venv/bin/python runs/fyp-seven-model-comparison-20261120/verify.py
git diff --check
```

本地证据：上述目录的 alignment-and-status.json、tensorboard-verification.json 与 raw/。
此次仅只读同步云端产物、扩展本地对比并恢复本地服务，没有提交新训练。

### 三个独立实验提交（2026-11-22）

Source: 用户确认三个改动分别基于 03，且 MLP 使用 [512,256] 两层。

- `doku-exp-fyp-lhuc-network-country-20261122-fe609c95`
- `doku-exp-fyp-lhuc-mlp512-20261122-54f3caf0`
- `doku-exp-fyp-lhuc-task64-20261122-5b4b2ac5`

三个 CreateTrainingJob 均成功，describe 返回 InProgress / Starting，FailureReason 为空；
资源、镜像、启动参数、环境、输入输出、角色和停止条件均与预期一致。尚不声明训练完成。

prepare.py 的 schema、差异还原、冻结输入与源码覆盖包逐字节断言通过。
network-country 序列化配置的 embedding lookup 顺序、表映射及次数与 03 完全一致；
主干字段拼接顺序不变，LHUC 字段严格为 u_adjust_network、ctx_ip_country。
任务64修改七个隐藏层，二分类输出1维和时长输出61维保持不变。
远端 03 原 recipe 全文件及原参考配置核对通过。三项互不叠加，日期及训练设置保持原值。

实际执行并通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-country-lhuc-independent-20261122/prepare.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-country-lhuc-independent-20261122/submit.py
git diff --check
```

各变体配置、验证、请求、成功提交回执、describe 状态保存在上述本地目录，未提交运行产物。

### 三个独立实验进度及 TensorBoard（2026-11-22）

Source: 用户要求查询最新三个任务是否完成，并加入原 TensorBoard。

Describe 返回全部 InProgress，FailureReason 为空；已上传完整 epoch 指标分别为
network-country 9、mlp512 6、task64 8，尚不声明训练完成。
新增 run 12_All_network-country-LHUC、13_All_country-LHUC_mlp512、
14_All_country-LHUC_task64，沿用原6012端口。14组的冻结日期、分片及合同完全一致；
各组已有56条曲线，US子集口径一致，每任务每轮52777行。新增1288个标量点与
源JSON、事件和TensorBoard HTTP逐项核验通过（rtol=1e-6、atol=1e-7）。
原11组仍各30轮，08的本地US补评估事件保留。恢复本地120秒只读同步进程。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-seven-model-comparison-20261120/sync.py
.venv/bin/python runs/fyp-seven-model-comparison-20261120/verify.py
git diff --check
```

证据保存在对比目录的 alignment-and-status.json、tensorboard-verification.json、
raw/ 和 refresh-status.json。此次未启动新训练或修改云资源。

### 两组容量融合实验提交（2026-11-22）

Source: 用户要求提交基于03的 mlp512/task64/e16 与 mlp256/task64/e16。

- `doku-exp-fyp-lhuc-mlp512-task64-e16-20261122-f95102d3`
- `doku-exp-fyp-lhuc-mlp256-task64-e16-20261122-b65ca421`

两个 CreateTrainingJob 成功，describe 均 InProgress / Starting、FailureReason 为空。
配置schema、三项差异还原后与03相等、冻结输入和源码覆盖包字节一致均通过；
两组除MLP宽度及run-scoped Manifest地址外完全相同。
保持国家LHUC输入及其8维embedding，embedding16仅作用于drama。
远端03完整recipe和参考配置核对通过；新任务describe的环境、资源、镜像、启动参数、
输入输出、角色、停止条件和VPC均与预期一致。此为提交成功证据，尚不声明epoch完成。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-country-lhuc-combined-20261122/prepare.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-country-lhuc-combined-20261122/submit.py
git diff --check
```

请求、配置、verification、submission、status保存在上述本地目录的两个变体子目录中。

### 两组融合实验完成及 TensorBoard（2026-11-23）

Source: 用户要求确认最近两个融合任务完成情况并加入原TensorBoard。

两个任务均Completed，完整30个epoch，成功封印及指标/事件摘要核验通过。
原目录新增15_All_country-LHUC_mlp512-task64-e16、
16_All_country-LHUC_mlp256-task64-e16。此前12～14也全部完成，16组均30轮。
全部train 2026-10-19～11-17、validation 11-18、冻结分片及合同一致；
每组56条曲线/1680点，总26880点，EventAccumulator与TensorBoard HTTP逐点匹配
源JSON（rtol=1e-6、atol=1e-7）；US指标定义相同，每任务每轮52777行。
08原有本地补评估US事件保留；服务仍为本地6012端口。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-seven-model-comparison-20261120/sync.py
.venv/bin/python runs/fyp-seven-model-comparison-20261120/verify.py
git diff --check
```

证据：对比目录alignment-and-status.json、tensorboard-verification.json及raw/。
此次只读同步云端结果，无新增训练或云端修改。

### 国家+渠道+deep link LHUC提交（2026-11-23）

Source: 用户指定LHUC输入ctx_ip_country、u_adjust_network、u_deep_link_drama_id。

任务 `doku-exp-fyp-lhuc-country-network-deeplink-20261123-2cf5d639` 创建成功，
describe为InProgress / Starting，FailureReason为空。返回环境、资源、镜像、启动参数、
输入输出、角色、停止条件和VPC与预期一致；尚不声明epoch完成。

prepare.py验证schema、准确LHUC字段、主干拼接顺序、embedding表及lookup顺序/次数、
还原分组后与03完整配置相等，均通过；冻结数据及源码覆盖包逐字节一致。
远端03完整recipe及参考配置核验通过。MLP[128,64]、七个task隐藏层[32]、
原embedding维度、stop-gradient与任务塔LHUC、30epoch及冻结日期均保持原值。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-country-network-deeplink-lhuc-20261123/prepare.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-country-network-deeplink-lhuc-20261123/submit.py
git diff --check
```

请求、配置、验证、成功提交回执和describe状态在上述本地目录的country-network-deeplink子目录中。

### 三条件 LHUC TensorBoard（2026-11-23）

Source: 用户要求将最新三条件LHUC逐epoch指标加入原TensorBoard。

新增17_All_country-network-deeplink-LHUC，目前InProgress / Training，FailureReason为空，
已同步18/30轮，56条曲线、1008点；尚无完成封印，不声明训练完成。
17组日期、冻结分片及合同一致，美国子集指标定义相同、每任务每轮52777行。
全部事件/HTTP逐点匹配源JSON（rtol=1e-6、atol=1e-7）；原16组均30轮，08补评估保留。
恢复原6012端口服务及本地120秒只读同步进程，任务全部终态后自动停止。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-seven-model-comparison-20261120/sync.py
.venv/bin/python runs/fyp-seven-model-comparison-20261120/verify.py
git diff --check
```

证据在对比目录alignment-and-status.json、tensorboard-verification.json、raw/及刷新日志。

### 全 context LHUC 六组容量实验提交（2026-11-23）

Source: 用户指定基于04提交六组独立容量调整。

| 变体 | SageMaker任务 |
| --- | --- |
| 256 | doku-exp-fyp-ctx-lhuc-mlp256-20261123-4963c9be |
| 512 | doku-exp-fyp-ctx-lhuc-mlp512-20261123-09350781 |
| e16 | doku-exp-fyp-ctx-lhuc-e16-20261123-2718c851 |
| task64 | doku-exp-fyp-ctx-lhuc-task64-20261123-2175181e |
| 256+e16+task64 | doku-exp-fyp-ctx-lhuc-mlp256-e16-task64-20261123-cd7f57fd |
| 512+e16+task64 | doku-exp-fyp-ctx-lhuc-mlp512-e16-task64-20261123-53e281a1 |

六次CreateTrainingJob均成功，提交后describe均InProgress / Starting、FailureReason为空。
资源、固定镜像、环境、启动参数、输入输出、角色、停止条件及VPC与预期逐项一致。
参数schema及指定差异还原后与04相等通过，冻结输入/覆盖包字节一致通过；
原04完整recipe及05/06参考配置远端核对通过。六组LHUC字段均严格为ctx_platform、
u_deep_link_drama_id、u_adjust_network、ctx_ip_country，验证日期均2026-11-18。
所有任务30epoch，batch2048；未声明epoch完成或训练成功结束。

实际执行并通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-context-country-lhuc-capacity-20261123/prepare.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/fyp-context-country-lhuc-capacity-20261123/submit.py
git diff --check
```

配置、请求、验证、提交回执及describe状态保留在上述本地目录各变体子目录中。

### inner_watch_lr003 + 全 context LHUC提交（2026-11-23）

Source: 用户要求找到inner_watch_lr003配置、加入04的LHUC、按04日期提交训练。

来源：已发布成功训练work/inner_watch_lr003/b1db1706ac28347cde3e01c0e1d1763db887a2760c92af3051f378539620f582/run。
experiment-config、resolved-config、input/record及两个合同逐项通过COMMITTED摘要校验。
原模型8任务（含inner_watch_seconds）、学习率0.003、任务损失/标签/指标/输出和选模设置保留。
增加[context,country] LHUC、stop-gradient及任务塔门控；主干字段次序保持一致，
还原LHUC及日期改动后与原配置相等；参数schema验证通过。
04冻结Manifest、合同与源码覆盖包逐字节一致，labels合同确认series_watch_duration_ms存在；
训练UTC2026-10-19～11-17，验证11-18，30epoch、batch2048、单台ml.m5.4xlarge/100GB。

新任务：`doku-exp-inner-watch-lr003-ctx-lhuc-20261123-8651e039`。
CreateTrainingJob成功，describe为InProgress / Starting，FailureReason为空；
环境、资源、镜像、启动参数、输入输出、角色、停止条件和VPC均与预期相同。
这是成功提交证据，尚不声明epoch完成。独立实验路径未触发日更发布或Endpoint变更。

实际执行通过：

```sh
recommend_tf/.venv/bin/python runs/inner-watch-lr003-context-lhuc-20261123/fetch.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-lr003-context-lhuc-20261123/prepare.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-lr003-context-lhuc-20261123/submit.py
git diff --check
```

本地来源及摘要在source/，新配置、请求、verification、submission和status在experiment/子目录。

### inner-watch LHUC 六组容量实验提交（2026-11-23）

Source: 用户引用8651e039，明确要求提交六组容量配置。

| 变体 | 任务名 |
| --- | --- |
| 256 | doku-exp-iw003-ctx-lhuc-mlp256-20261123-2c57fb3b |
| 512 | doku-exp-iw003-ctx-lhuc-mlp512-20261123-c7e1288b |
| e16 | doku-exp-iw003-ctx-lhuc-e16-20261123-3d681120 |
| task64 | doku-exp-iw003-ctx-lhuc-task64-20261123-6d6fb8f5 |
| 256+e16+task64 | doku-exp-iw003-ctx-lhuc-mlp256-e16-task64-20261123-6d76acc5 |
| 512+e16+task64 | doku-exp-iw003-ctx-lhuc-mlp512-e16-task64-20261123-60bb5168 |

六个CreateTrainingJob均成功，describe均InProgress / Starting，FailureReason为空。
各项环境、资源、镜像、启动参数、输入输出、角色、停止条件和VPC与请求一致。
基线8651e039状态及完整recipe远端核验通过；所有配置schema、容量改动还原后与基线相等、
冻结输入和源码覆盖包字节一致均通过。确认全部保留8任务、LR0.003、inner-watch最终输出101维、
完整context+country LHUC、UTC验证11/18；task64改动覆盖全部8个任务隐藏层。
本记录只验证成功提交，尚不声明epoch完成。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-lr003-context-lhuc-capacity-20261123/prepare.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-lr003-context-lhuc-capacity-20261123/submit.py
git diff --check
```

六组配置、request、verification、submission和status保存在上述本地目录各变体中。

### 独立14任务TensorBoard（2026-11-24）

Source: 用户标注04的六组容量实验、inner-watch六组容量实验与8651e039，并要求加入04基线，
新建独立TensorBoard。

新目录runs/context-lhuc-inner-watch-comparison-20261124，端口6013。14组全部Completed、
完整30epoch；UTC训练10/19～11/17、验证11/18（2026年）、冻结分片和合同一致。
成功产物摘要校验通过；原生事件和TensorBoard HTTP全部逐点匹配源JSON（rtol=1e-6、atol=1e-7）。
共同七任务标签/指标定义及199530行验证样本一致；inner-watch专有任务有效样本28487行。
inner-watch七组无原生auc_us/mse_us，保留缺失并说明；未进行额外checkpoint补评估。
训练侧展示逐任务loss及objective，原训练无train AUC；两组总objective不作为同口径评分。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/context-lhuc-inner-watch-comparison-20261124/sync.py
.venv/bin/python runs/context-lhuc-inner-watch-comparison-20261124/verify.py
git diff --check
```

本地证据：alignment-and-status.json、tensorboard-verification.json、
metric-population-verification.json及README任务映射。所有任务已终态，无需后台持续同步。

### Inner-watch 七组美国子集指标补评估（2026-11-24）

Source: 用户询问08～14为何缺失auc_us，并要求补齐。

原inner_watch_lr003及六组容量实验的配置均未声明auc_us/mse_us，因此原训练未记录。
现已只读下载并校验七组各30个checkpoint、原词表、配置和冻结输入元数据，
使用相同2026-11-18验证数据，按ctx_ip_country == US补评估52777条样本。
六个分类任务auc_us及watch_seconds/mse_us定义与01完全相同。
每组第1轮全量原生指标与训练记录通过rtol=1e-5、atol=1e-6核验，
并验证全量预测过滤US与US子集直接推理一致；每个checkpoint identity和每轮US行数均核验。
原训练TF2.11.1/Python3.10.22，本地评估TF2.11.0/Python3.10.20；原始与评估源码/环境摘要分开记录。
未重新训练，未修改云端权重或原始事件；EVAL图无train_op。

七组共210轮补评估完成，端口6013每组新增7条曲线、210个点，共1470点。
14组全量24990个事件及HTTP返回数据点与对应原始/补评估JSON逐点匹配，
rtol=1e-6、atol=1e-7。runs_without_us_subset_curves为空。
额外inner_watch_seconds是回归任务，不定义AUC。本记录更新上节“未补评估”的历史状态。

实际执行通过（下载曾因网络中断恢复，全部最终摘要校验通过）：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-us-eval-20261124/run.py 13_InnerWatch_lr003_context-country-LHUC_mlp256-e16-task64 14_InnerWatch_lr003_context-country-LHUC_mlp512-e16-task64
.venv/bin/python runs/inner-watch-us-eval-20261124/attach_all.py
.venv/bin/python runs/context-lhuc-inner-watch-comparison-20261124/verify.py
git diff --check
```

run.py完成时检查全部七组result.json；08～12已在先前分批执行中完成。
本地证据位于runs/inner-watch-us-eval-20261124：complete.json、各run的result.json、
download-verification.json、full-validation-parity.json、30份epoch-metrics和tensorboard-receipt.json。
最终HTTP核验见runs/context-lhuc-inner-watch-comparison-20261124/tensorboard-verification.json。

### 14号账号特征本地配置（2026-11-24）

Source: 用户要求给14增加account_id并核实上游字段。

已核实原任务60bb5168的COMMITTED配置摘要、全部31天冻结合同，以及2026-11-18
本地验证Parquet的SHA256。字段u_account_id为int64，源user_features.account_id。
原14未在dataset.features、dtypes或模型输入中启用它。过滤video_finish_seen > 0后，
199530条验证样本包含37791个不同账号，空值和非正值均为0；未输出任何原始账号。
该实测覆盖验证分片，训练日期已核验合同，未声明逐日账号值覆盖率。

新增独立16维account embedding接入DNN，LHUC不变。模型参数与冻结Dataset合同验证通过；
撤销账号字段相关差异后配置与14完全相等。只准备本地配置，未提交云训练。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-account-id-20261124/prepare.py
git diff --check
```

配置与证据：runs/inner-watch-account-id-20261124/{config.yaml,config.json,verification.json}。

### 14号账号特征实验云提交（2026-11-24）

Source: 用户在本地账号特征配置准备完成后明确要求“提交训练任务”。

已提交doku-exp-iw003-ctx-lhuc-account16-20261124-48c03be4；提交后describe显示
InProgress / Starting，FailureReason为空。仅新增u_account_id独立16维embedding进入DNN；
保持14的MLP[512,256]、drama embedding16、八任务隐藏层[64]、LHUC context+country、
学习率0.003、30epoch、batch2048、单机ml.m5.4xlarge。
训练UTC2026-10-19～11-17、验证11-18。云端基线60bb5168及完整recipe逐文件核验通过；
新recipe仅配置与自身manifest位置变化，冻结分片、合同与源码overlay保持字节一致。
上传后逐文件回读通过；任务实际环境、镜像、资源、输入输出、角色、VPC等与请求一致。
此记录仅声明提交成功，不声明训练已完成。独立实验路径未触发模型发布或日更配置变更。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-account-id-20261124/prepare_submission.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-account-id-20261124/submit.py
git diff --check
```

证据：runs/inner-watch-account-id-20261124中的request.json、submitted-config.yaml、
submission-verification.json、submission.json、status.json及cloud-verification.json。

### 账号特征inner-watch权重0.2实验云提交（2026-11-24）

Source: 用户要求基于48c03be4，仅改inner-watch loss权重为0.2并提交训练任务。

任务doku-exp-iw003-ctx-lhuc-acct16-w02-20261124-596ba302已成功提交；
describe为InProgress / Starting，FailureReason为空。
仅将inner_watch_seconds.loss_weight的系数0.3改为0.2，有效标签mask保持不变。
还原该字段和独立manifest地址后，配置与48c03be4完全相等；冻结输入和源码overlay字节一致。
保留MLP[512,256]、任务塔[64]、drama/account embedding16、LHUC context+country、
LR0.003、30epoch、batch2048、ml.m5.4xlarge；训练UTC10/19～11/17、验证11/18（2026年）。
基线云端完整recipe、新上传recipe均逐文件回读核验通过；提交后请求与实际任务配置匹配。
此处仅声明提交成功，未声明训练完成；原48c03be4任务保持运行，不涉及线上发布。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-account-id-w02-20261124/prepare.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-account-id-w02-20261124/submit.py
git diff --check
```

证据位于runs/inner-watch-account-id-w02-20261124：config.yaml、request.json、
submission-verification.json、submission.json、status.json、cloud-verification.json。

### 权重0.2无账号特征实验云提交（2026-11-24）

Source: 用户引用596ba302，明确要求提交不加入u_account_id的实验。

已提交doku-exp-iw003-ctx-lhuc-noacct-w02-20261124-e97e7194；提交后状态
InProgress / Starting，FailureReason为空。移除account输入组、账号embedding及对应
数据投影/default/dtype配置，保留inner-watch loss权重0.2与原mask。
恢复账号相关配置和manifest位置后与596ba302完全一致；独立对照14号原始配置，
仅inner-watch loss权重和run位置不同。冻结输入、合同与源码overlay字节一致。
MLP[512,256]、任务塔[64]、drama embedding16、context+country LHUC、学习率0.003、
30epoch、batch2048、单机ml.m5.4xlarge不变；UTC训练2026-10-19～11-17，验证11-18。
基线云端recipe与新上传recipe逐文件回读通过，实际任务配置与请求匹配。
仅声明提交成功，不声明训练完成；未更新线上模型或修改原任务。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-no-account-w02-20261124/prepare.py
PYTHONPATH=. recommend_tf/.venv/bin/python runs/inner-watch-no-account-w02-20261124/submit.py
git diff --check
```

证据：runs/inner-watch-no-account-w02-20261124中的config.yaml、request.json、
submission-verification.json、submission.json、status.json及cloud-verification.json。

### 账号特征三组结果加入TensorBoard（2026-11-25）

Source: 用户要求最近两组596ba302/e97e7194加入TensorBoard，随后补充48c03be4。

三个任务全部Completed，各30epoch；在原6013面板加入15（账号16/权重0.2）、
16（无账号/权重0.2）、17（账号16/权重0.3）。原14组及08～14的US补评估事件保留。
所有17组冻结训练UTC2026-10-19～11-17、验证11-18，分片及合同一致。
新任务COMMITTED摘要核验通过，每组56条原生曲线/1680点，三组新增5040点；
全部17组30030点通过事件及HTTP与源JSON逐点核验（rtol=1e-6、atol=1e-7）。
新三组无原生auc_us/mse_us，本次未进行checkpoint补评估，界面及文档明确保留缺失。
TensorBoard在原端口6013重启；仅只读云端产物，无新云训练或发布操作。

实际执行通过：

```sh
PYTHONPATH=. recommend_tf/.venv/bin/python runs/context-lhuc-inner-watch-comparison-20261124/sync.py
.venv/bin/python runs/context-lhuc-inner-watch-comparison-20261124/verify.py
git diff --check
```

证据位于该目录：alignment-and-status.json、tensorboard-verification.json、
sync-20261125.log、verify-20261125.json及README.md。

## 2026-11-25：模型14独立FYP发布验收（进行中）

Source: 用户授权模型14上线并启动FYP日更，暂不开AB。

- 原训练任务 `doku-exp-iw003-ctx-lhuc-mlp512-e16-task64-20261123-60bb5168` Completed，
  原选中epoch4；223个封存文件共462,793,199字节，经COMMITTED逐项校验后导出。
- 新模型版本 `c4a6245445ace123856f25b7fb0a7106d865d7dd2427addf2c6db4dccc03eba0`。
  原始manifest SHA256 `456f7775516265be1db15fb75400a75aae120a2feef4a0a8c447b7fcf00e5943`。
- 初始 Endpoint 为 `doku-prod-fyp-inner-watch-lhuc-m512-e16-t64`，
  私有FYP网络中 serving_with_model_version 探针验证3行、5头、有限数值与版本。
  InvokeEndpoint request ID：`609ce830-25f9-43f1-85b5-4128f8ce47b4`。
- IAM simulate_principal_policy 验证FYP ai-engine运行角色可调用新Endpoint。
- 使用线上ai-engine提交2e9e211的cmd/releases验证旧列表/新定义及新列表/新定义均成功。
  原始导出country分组合同经单独核验；所有文件摘要真实保留。
- 持FYP共享publication lock发布定义v26/deployment820、列表v30/deployment821，
  两步均COMPLETE且数据面原文一致。两个运行实例均记录新别名generation-ready。
- 本地 `pytest recommend_tf/train/tests -q -p no:cacheprovider --disable-warnings`：263 passed。
  `ruff check`、`ruff format --check`、`git diff --check`通过。
- Terraform首阶段计划7 create、1 update，无删除；apply通过。
  日更阶段计划9 create、3 update，无删除；fmt/validate通过。
- 完整回执及冻结配置仅保存在本地 `runs/lhuc14-release-20261125/` 和独立S3发布前缀。

- 两个ai-engine实例的实际BatchEstimate均通过：匿名user_id=0，对两个有效内容候选
  分别调用新别名及inner_watch_lr003对照；共4次请求，每次2行5头，无error_info，
  分数有限且两个实例结果一致，新模型执行版本精确等于c4a624...3eba0。
  初始示例内容ID不存在及episode_id JSON类型错误均已更正，不作为验收成功证据。
- 统一 `make check`：格式/lint/类型通过，公共栈329 passed/3 skipped，TF栈263 passed。
  首次inference-check发现共享源码包pin需要同步；按仓库`package_source.py --pin`流程
  更新本地manifest后重跑`make inference-check`，37 tests OK。未注册或激活Idea新任务。
- content-rec当前线上a5ae5f7未登记新别名，已准备`caller-binding.patch`，仅追加ai_models
  与routes，保留默认选模及融合设置。此调用方变更未部署；开AB前需接线并验收业务选模。

- 日更阶段Terraform apply完成：9 added、3 changed、0 destroyed；
  ready Lambda代码摘要与本地59f8d9...8fb3一致，上游/恢复触发均接入，worker固定到
  `doku-prod-tf-rerank-daily:35`。
- 2026-11-24 cutoff的31天输入已按不可变manifest/合同冻结，
  通过事件入口提交首个日更请求（event ID 7ea736d4-7136-3a43-7b3d-d9e0549afaf9）。
- `final_verify.py`核验Endpoint InService、Lambda Active、正确任务修订及两条触发器；
  AB profile 3ers4y8数据面原文与操作前完全一致。
