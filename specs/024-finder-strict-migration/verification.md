# Spec 024 验证记录

本文按日期保留历史验收记录；最新状态见文末 PR 合入验收。

初始验收日期：2026-10-28。当时验证对象是独立 worktree，未使用旧 TF 实现、未访问生产样本。

## 已运行

| 命令/验证 | 结果 |
| --- | --- |
| `UV_CACHE_DIR=/tmp/spec024-uv UV_PROJECT_ENVIRONMENT=/Users/yujiatang/doku/doku-train/.venv UV_NO_SYNC=1 make check` | 根锁检查、格式、lint、mypy通过；原有318 passed / 3 skipped；当时TF10 passed |
| `PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests -q -p no:cacheprovider --disable-warnings --tb=short` | 最终增补PREDICT测试后11 passed，6.71秒；108项主要为TF2.11旧API/Protobuf弃用警告 |
| `ruff check . --no-cache`、`ruff format --check . --no-cache` | 通过；247个文件格式通过（包含独立包） |
| `git diff --check` | 通过 |
| Python AST比较源与目标transform_features、dnn_layer、target_layer、scalar_features | 四函数AST一致；其调用依赖中的本地适配另列source-inventory.json |
| 本地Linux amd64镜像，网络关闭，TF2.11.0，Finder AdamW零梯度dense一步 | 通过，变量1→0.99，学习率0.1未额外乘入衰减系数 |

原生完整TF测试环境：macOS arm64、Python3.10.20、tensorflow-macos2.11.0、NumPy1.23.5、Arrow17.0.0。独立锁文件同时包含Linux tensorflow-cpu2.11.1平台分支，Linux完整链路尚未验证。

## 覆盖

- Finder表达式广播、loss_weight=0、loss_by_weight两种分母。
- Finder AdamW dense全变量与IndexedSlices命中行衰减；padding不生成触碰OOV行的稀疏梯度。
- 历史重复权重合并、排序、截断、length与动态padding；空历史与weighted_mean梯度。
- 固定中间值：特征转换、scalar、dot attention、FM；共享OOV表与不需要标签的PREDICT。
- 分片完整性/摘要、未完成分片拒绝、有界shuffle exact-once、尾batch。
- 七任务训练/验证；启用Finder dropout的epoch恢复；模型参数与optimizer slots由Saver恢复。
- 原生合成Manifest/合同→基础分片→显式配置映射→训练→独立test，原始产物只读。
- 精确AUC排序聚合、同分处理、单类别null原因；选模写出。
- 禁止隐式Adam weight_decay及AdamW默认衰减参数。

## 限制与未完成验收

- Linux amd64模拟容器在import pyarrow时发生segfault（未进入测试）；禁用Arrow SIMD后仍出现。没有将其算作训练代码测试通过。
- 未执行真实e1、40 epoch容量测试或云任务；没有提速结论。
- 实际e1的Adam+0.0001不得静默转换；migrate_experiment要求调用方显式传入Finder optimizer settings。
- 固定本地表不是私有动态embedding服务的逐参数等价替代；只声明所选本地变量后端语义。
- 当前计时为准备、词表、图构建/恢复、训练含输入、验证含输入；更细input/forward/update拆分尚未做。
- Doku固定Manifest/合同实现从本仓复用并独立命名空间化；没有导入Torch模型、旧TF包或Spec022特征准备。


## 后续云验收已完成

用户授权后，原生Linux SageMaker已完成全部11项合成测试，任务Completed；此前Linux待验收状态已解除。详情见[云验收记录](cloud-verification.md)。真实数据容量与性能仍未验收。


## extended40-lr004 配置适配（2026-10-29）

用户选定方案②，基于原提交包生成独立 Finder TF 配置；详见[配置差异与 review](experiment-review.md)。实验配置保存在本地 runs/finder-e1-extended40-lr004-review；未提交云任务。完整配置合成验证结果以该目录 verification.json 为准。


## 首次40 epoch已启动

新增逐epoch指标、TensorBoard、S3持久化与完整入口；本地make tf-check和同镜像原生Linux均14 passed。用户授权后的首个正式任务已提交，详情见[正式训练记录](formal-training.md)。不把启动成功当作40轮完成。


## 2026-10-29 固定真实切片输入耗时诊断

用户要求不影响运行中的40 epoch任务。仅S3只读下载冻结Manifest的10/01、10/12、10/23分片，各取32768原始行，共93582过滤后样本/46 batch；复用正式训练完整词表，batch2048、shuffle8192、seed42、线程4/1。训练源码摘要与正在运行镜像一致。

本地macOS arm64 / TF2.11.0，独立图权重，warm-up后重复3次；模型数据准备和网络下载不计入。执行本地 runs/finder-input-profile-20261029/profile_slice.py：读取/shuffle中位6.493s、数组/feed0.957s、sess.run0.956s、总8.433s。

诊断发现prepared含未用于模型的impression_time，Arrow转Python时间重复尝试查找未安装的pytz。仅在隔离脚本profile_projected.py中排除该列解码：读取4.345s、数组0.975s、sess.run0.953s、总6.282s，时间缩短25.5%。check_parity.py验证全部46 batch/93582样本模型输入数组完全一致。未修改正式实现、依赖、任务或云端数据。

数据与详细证据保留本地 runs/finder-input-profile-20261029/{report,projected-report,reader-profile,import-probe,parity,summary}.json。该结果定位本机输入瓶颈，不直接外推AWS每轮提速比例；原生Linux同规格测量仍未执行。git diff --check通过。


## 2026-10-29 本地落实输入列投影及剩余开销诊断

用户确认去掉未使用的时间列，保持运行中任务不变。本地train_rerank在事件时间切分后仅保存声明输入；prepared.rows支持columns，词表拟合/train/validation/test统一投影模型输入和历史length，不硬编码禁止显式需要的字段。

同一已缓存真实切片93582行/46 batch，完整词表、seed42、batch2048、shuffle8192、线程4/1，重复3次中位：读取/shuffle4.132s、数组/feed1.000s、sess.run0.966s、总6.099s；相较原8.433s缩短27.7%。环境仍为本机macOS ARM，不外推AWS绝对耗时。

独立细分读取测量（含计时开销）中位总4.027s：to_pylist2.709s、RSS检查0.881s、随机选择/pop0.074s、Arrow读取/解码0.059s、打开文件0.018s、校验和0.013s，其余Python迭代及计时开销约0.250s。列式数据转换成逐行Python对象是剩余主要开销，不是shuffle抽样本身。

验证：make tf-check 15 passed；新增新分片不含impression_time断言后，pytest -k 'manifest_entry or projection' 2 passed；全46 batch所有模型输入数组逐项相等。ruff check/format --check、git diff --check通过。证据：本地runs/finder-input-profile-20261029/implemented-summary.json、reader-detail.json、implemented-parity.json，脚本profile_after.py/profile_reader_detail.py/check_implemented_parity.py。未修改运行中镜像、未提交新训练任务。


## 2026-10-29 列式行引用及RSS检查降频

用户授权继续本地优化，保留训练语义和shuffle顺序，不影响正在运行任务。输入改为Arrow→NumPy列缓冲+ColumnRow引用；null/不支持类型逐列回退保持语义。同步RSS检查从逐行改为每1024输入行和每个输出batch，后台50ms采样不变。未修改随机调用序列、pop位置、batch边界和尾部排空方式。

验证命令make tf-check：17 passed（含跨块、空历史、null、字符串、多个seed/缓冲大小的完整顺序、RSS调用次数与异常）。ruff check/format --check及git diff --check通过。

同切片93582行/46 batch、完整词表、batch2048、seed42、shuffle8192、4线程，旧新版本串行各3次，固定Python/NumPy/TF初始化种子及验证初始变量摘要。中位数：读取/shuffle3.993→0.628s；数组/feed0.987→1.137s；sess.run0.938→0.952s；总5.919→2.697s，缩短54.4%。本地采样峰值RSS962.1→1065.4MiB，列块引用有内存代价。该数据不外推AWS绝对收益。

check_columnar_parity.py验证全46 batch全部输入dtype与字节逐项一致。严格逐batch loss rtol1e-6对照未通过；固定初始权重后跨版本最大loss差0.000472188，旧版自身重复最大差0.000472188，新版自身0.000472903，未超既有重复波动。单线程附加对照仍有0.000471950差异，不能简单归因于线程数量，也不宣称位级轨迹等价。首轮未固定完整初始化种子的结果保存在uncontrolled-init-*，不用于最终性能/数值结论。

证据在本地runs/finder-input-profile-20261029/columnar-{baseline-report,candidate-report,parity,summary}.json；复现脚本profile_columnar.py/check_columnar_parity.py。当前云任务、镜像和产物未修改。


优化版40 epoch任务已按用户授权独立提交；原生Linux同镜像17项测试通过。配置及依赖与首轮一致，详情见[正式训练记录](formal-training.md)。云端性能验证待新任务产生epoch结果，不能以本地54%提速代替。

## 训练前准备/词表拟合对照（2026-10-29，本地）

范围：本次优化前的当前 Finder TF 代码为 baseline（已包含上一轮输入读取/RSS优化），不是已弃用的旧 TF。复用同一冻结 Manifest 的 2026-10-01、10-12、10-23 各前32768原始行，共98304行，过滤后93582行，batch2048共46批。使用本地已校验缓存，不含S3下载；没有启动云训练、构建/推送镜像或修改在跑任务。macOS ARM / TensorFlow2.11，本地数据与报告在 gitignored `runs/finder-pretrain-profile-20261029/`。

先用 cProfile 定位热点，再关闭 cProfile，以块级计时各测3次。下表为各项中位数，子项中位数之和不必等于总计中位数：

| 阶段（秒） | baseline | 优化后 |
| --- | ---: | ---: |
| 原始缓存读取、校验与列解码 | 6.254 | 3.018 |
| 输入块组装 | 0.275 | 0.014 |
| 默认值与历史整理 | 2.073 | 1.937 |
| 过滤图构建与执行 | 0.096 | 0.075 |
| Parquet写出 | 0.288 | 0.292 |
| prepared校验和 | 0.016 | 0.016 |
| 筛选、Arrow构造及其他准备开销 | 0.743 | 0.723 |
| **一次性准备总计** | **9.754** | **6.074** |
| 拟合：prepared读取与分批 | 0.526 | 0.502 |
| 拟合：张量组装 | 0.739 | 0.098 |
| 拟合：构图/图执行/词表集合收集排序等 | 0.566 | 0.555 |
| **词表拟合总计** | **1.826** | **1.158** |
| **两阶段合计** | **11.593** | **7.242** |

准备缩短37.7%，拟合缩短36.6%，合计缩短37.5%。变化不是每轮shuffle优化；训练shuffle和tensor_batch未改。主要收益来自审计时间的列式校验/免Python解码，以及拟合直接按列组装张量。过滤图占比很小，本轮没有实现过滤图缓存；Parquet写出未加速。cProfile中审计时间解码触发大量时区模块查找，这部分收益可能依赖本地依赖环境；不能按此比例外推全量Linux/SageMaker的34分钟准备耗时。

一致性：`parity.py` 检查93582行全部基础字段、历史及length；三次拟合输出的完整词表JSON精确相等；无随机/seed42/seed43三种设置各46批张量shape、dtype及bytes全部一致。`fit_parity.py` 单独验证列式拟合路径46批张量bytes与原路径全部一致。没有宣称重新训练的浮点参数逐bit一致，本次没有改变模型图、优化器或表达式。

执行证据（均已完成）：

```text
PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 recommend_tf/.venv/bin/python runs/finder-pretrain-profile-20261029/benchmark.py baseline plain
PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 recommend_tf/.venv/bin/python runs/finder-pretrain-profile-20261029/benchmark.py candidate plain
PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 recommend_tf/.venv/bin/python runs/finder-pretrain-profile-20261029/parity.py
PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=2 recommend_tf/.venv/bin/python runs/finder-pretrain-profile-20261029/fit_parity.py
UV_CACHE_DIR=/tmp/spec024-uv make tf-check
# 22 passed, 110 warnings in 34.54s
```

报告含 `baseline-plain.json`、`candidate-plain.json`、`parity.json`；保留baseline代码快照、profiler数据和脚本。脚本输出目录create-only，重复运行需选择新目录。新增测试覆盖列/行基础处理一致、空历史跨分片批合并、尾批、过滤前错误、带时区timestamp的s/ms/us/ns边界和null。修改文件ruff check/format --check及git diff --check通过。云端全量性能尚未验证。

### 本轮云端提交验证

后续用户已授权独立40 epoch任务：同镜像原生Linux amd64测试22 passed（21.06s），CodeBuild成功，源码/锁摘要校验通过后提交 `doku-exp-finder024-prep40-20261029-040027`。配置、资源、网络及最大时限与第二次任务逐项一致，任务名及S3输出路径独立。详见 [正式训练记录](formal-training.md)。这确认镜像验收与提交成功，不代表全量训练或云端性能验收完成。

## Finder LN搜参扩展本地验收（2026-10-29）

用户要求六组单因素实验。Finder公共LayerNorm affine路径已接入，LN顺序为Dense→LN→ReLU→Dropout，共享层与任务塔支持；空norm保持原路径。配置限定每个隐藏层一个norm，拒绝其他norm类型及不匹配长度。

`PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 recommend_tf/.venv/bin/python runs/finder-six-experiments-20261029/check_finder_ln.py`：二维[4,8]和三维[2,3,8]输入，对照原Finder类，前向及输入/gamma/beta梯度精确相等。单测另有非默认gamma/beta、零方差行、TRAIN/EVAL变量复用、无norm/LN两路径七任务完整训练及checkpoint恢复。零方差非零输入下Finder的x*inv+(beta-mean*inv)存在float32消减误差，独立居中公式测试该行绝对容差2e-4，其余行1e-5；保留Finder计算顺序，不以改变公式消除差异。

`UV_CACHE_DIR=/tmp/spec024-uv make tf-check`：25 passed, 113 warnings in 46.83s。

`verify_design.py`：LR001/LR002仅learning_rate变化，WD0/WD3E5仅weight_decay变化，LN仅共享/任务隐藏层norms变化。YAML使用行内数组，配置保存本地runs，未提交源码。修改文件ruff check和git diff --check通过。

Adam语义及额外同资源baseline仍待答复；没有启动本轮任何云训练或镜像构建。后续即使本地通过，也需按最终源码/配置做原生Linux同镜像验收后再提交。

后续用户确认纯Adam及额外同资源baseline。七任务同镜像原生Linux验收25 passed（31.68s）、七配置差异校验通过后已提交；源码摘要 `96a5ec1b5f15edc2ef076e441d277f158cb67e4c4c7df6a88d67437a4b2e6fb0`。每份请求核对16 vCPU实例、独立输出及同镜像摘要，详见formal-training。完整结果待训练结束。

## 2026-10-29 Adam Dense/Embedding L2 本地验证

命令：`UV_CACHE_DIR=/tmp/spec024-uv make tf-check`，38 passed，122 warnings，51.08s。环境为本地 macOS TF 2.11；日志 `runs/finder-regularization-audit-20261029/local-tests.log`。`ruff check recommend_tf/train`、`ruff format --check recommend_tf/train`（38 files）、`git diff --check` 均通过。

新增验证：共享表 candidate/history 的重复 ID 按出现次数累计；padding 不贡献梯度；全空历史为零；真实 OOV 有正则；Dense 仅配置隐藏 kernel，梯度为 2λW；训练总 loss 分解相加；Adam 多步结果与显式 occurrence L2 对照一致；默认关闭与显式关闭 loss/梯度逐元素一致；非法系数/空匹配/混用 AdamW 拒绝；七任务 LN+L2 两轮训练及 checkpoint 恢复一致；独立正则 TensorBoard 标量与 epoch JSON 持久化。

在跑七组本地配置逐一调用 parameters，全部仍可解析且新增正则默认关闭。未修改这些实验配置、镜像或 AWS 任务。

初轮默认/显式关闭对照因测试未固定 Python/NumPy 初始化随机状态失败，补齐后通过；中间一次恢复测试因验收运行期间本地源码发生编辑触发 source identity mismatch，冻结代码后完整重跑通过。未放宽恢复一致性校验。

限制：以上为本地合成验收，不代表原生 Linux 或真实切片性能验收；本轮没有启动云端构建或训练。未据此声明训练质量或耗时收益。

## 2026-10-29 Adam L2 五组云端提交前验收

用户授权原生 Linux 验收后提交 D1/D3/E1/E3/D2+E2。CodeBuild `doku-train:075e6a61-3e17-4b29-b7d0-ae0437fdb304` 成功；Linux x86_64、Python 3.10.22、TF 2.11.1、Arrow 17.0.0，38 passed、22 warnings、48.53s。五组配置相对冻结纯 Adam baseline 的差异断言全部通过。源码摘要 `84dc5606d48924e75070aaf00230678741fe14a8f69c18319b12e6389d79369d`，锁摘要 `b2763acbc6cb48f18bd9326b605e9bf7634a6d3a4ec05355023c8f8c983d0bd5`。

证据：`runs/finder-adam-l2-five-20261029/native-verification.json`、`linux-build.log`、各组 `request.json` 与 `submission.json`。此次验收为合成语义与运行环境验证；真实数据训练结果、耗时和质量需等待新任务完成，不据此声明收益。

## 2026-10-29 opt40/prep40 指标差异审计：输出层初始化可复现性缺口

两次云任务的 experiment-config、冻结 input/manifest、input/record、最终 training/feature-state 内容摘要完全相同。resolved-config 仅 prepared_identity 不同；该值包含源码摘要，不能当作数据内容变化证据。归档运行包只有 published_dataset、prepared、training_pipeline（拟合扫描）、train_rerank 四个生产文件不同，rerank_model 与优化器源码相同。

从两份运行归档取出相同 rerank_model，使用合成固定输入在四个独立进程中复现：仅 tf.set_random_seed(42) 时，31个参数中七个 task 最终 dense/kernel 初值不同；同时固定 Python random、NumPy 和 TF seed 后，31个参数全部相同。隐藏层显式 VarianceScaling 与 Embedding 初值在原测试中一致；问题具体位于未显式指定 initializer 的最终 tf.layers.dense 路径，不能泛称所有 Dense 初始化均随机失控。

可执行证据：`python3 runs/finder-three40-comparison-20261029/audit/run_probe.py`（本地 TF 2.11），`initializer-results.json`、`findings.json` 与 `frozen_rerank_model.py`。此为初始化缺口复现，不是恢复两次云任务的第0步权重；两次真实初始权重未保存，不能据此量化全部指标差异的来源。全量 prepared 分片未随产物保存；先前93582行切片的输入/词表/shuffle张量一致性验证仍有效，但不能冒充全量验证。

结论：已有对照未严格控制最终输出层初始化，存在混杂变量；尚不能把曲线差异归因于预处理优化。生产修复及固定初始 checkpoint 的对照另行处理。本次只审计，未修改模型、训练实现、运行镜像或云任务。

## opt40/prep40 独立进程与原生Linux排查

过程、证据和限制见 [一致性排查](consistency-investigation.md)。不再以自身重复波动作为训练一致性通过标准。

## 两处可复现性缺口正式修复（2026-10-29）

用户明确授权修复。新增 test_reproducibility.py 覆盖：跨所有优化器的梯度屏障可达性（dense/sparse衰减及Adam写操作）、同一/独立sparse优化器的可手算梯度与命中行衰减、不同PYTHONHASHSEED的独立子进程完整初始化与2 epoch训练逐字节一致、恢复epoch2与连续训练参数/slots及指标精确一致，以及不同seed导致不同初始化。

本地命令 `UV_CACHE_DIR=/tmp/spec024-uv make tf-check`：41 passed，122 warnings，63.36s。`ruff check recommend_tf/train`、`ruff format --check recommend_tf/train`（40 files）、`git diff --check`通过。日志在 `runs/finder-repro-fix-20261029/local-tests.log`。测试期间未编辑运行时代码，未绕过源码身份校验。

Finder衰减实现、系数、稀疏命中行、loss、shuffle、配置和依赖锁未改；本次只修复本地适配层执行顺序和随机状态。已有云任务保持原镜像，不混用新源码恢复旧checkpoint。全量数据扫描和新40轮任务不作为本次修复已完成事项。

正式修复原生Linux验收：CodeBuild `doku-train:87264099-c3cf-4bc2-ac79-fdd63361af48` 状态SUCCEEDED；Linux x86_64、Python3.10.22、TF2.11.1、Arrow17.0.0，41 passed、22 warnings、65.26s。源码摘要 `97add71fc4c39152437f2a4118ce26dba8955b485d296712df42d40bcf947a62` 与上传冻结源包匹配。验收输出保存在 `runs/finder-repro-fix-20261029/native-verification.json` 与 linux.log；本次不上传模型/测试checkpoint、不发布镜像、不提交云训练。

## 2026-10-29 opt/prep 全量15轮重跑已启动

Source: 用户授权两版各重跑15个epoch，不再对比历史base。

验收命令：`/Users/yujiatang/doku/doku-train/.venv/bin/python runs/finder-parity15-20261029/cloud.py build-status`。CodeBuild `doku-train:0f68b9a6-c540-451e-9f8e-fded16d10c12` SUCCEEDED；opt历史快照加两处修复：20 passed（49.50秒），prep历史快照加相同修复：25 passed（48.32秒），均为Linux x86_64、TF2.11.1。测试数量差异来自两版历史测试集；两版都新增了相同的复现与更新顺序回归。

`cloud.py submit` 已成功创建 `doku-exp-f024-opt15-20261029-121657` 和 `doku-exp-f024-prep15-20261029-121657`。两者沿用8 vCPU、4线程，配置摘要相同 `abf5865b6d85ef092ad9dfe718a213c4cafe1999b7045622639f5ae482ffbd8b`。详细请求、不可变镜像摘要、源码摘要与补丁保存在本地 `runs/finder-parity15-20261029/`。尚未完成15轮训练，不声明全量一致。

完成后执行该目录 `fetch_results.py` 与 `compare_results.py`，比较15轮指标和checkpoint张量；结果写入comparison.json。持久定时检查创建被自动审批拒绝（缺少用户明确的定时监控授权），未创建自动化；后续需用户授权监控或手动触发查询。

## 16 vCPU 线程基准脚本验证与首次重试

本地对已验收切片执行 `runs/finder-threads16-20261029/worker.py` 的initial及2批smoke，通过。首次云基准 `doku-exp-f024-threads16-20261029-125556` 在训练前摘要断言失败，未产出线程计时：新脚本对所有有tolist的NumPy标量进行了转换，原验收仅转换ndarray，两种JSON序列化产生不同摘要。本地同一份93582行数据复现两种摘要分别为 `8b6b4d0fc5cc6bb519c804d2bab2f171685b284fb444f9f9cbeb5e1ebb8af5d4` 和 `fb2a42434cd9d8125934fc6ca02ced76b82250e5024db5259f6933f272bf1f8a`。修复为原序列化及原字段投影口径，保留原摘要断言，失败记录归档到attempt1。用户授权范围内重新提交独立基准，不更改15轮任务。

## 16 vCPU线程基准最终结果

任务 `doku-exp-f024-threads16-20261029-130033` Completed，实际cpu_count=16，ml.m5.4xlarge，固定inter_op=1，生产代码使用prep15不可变镜像（含种子/AdamW顺序修复）。命令：`/Users/yujiatang/doku/doku-train/.venv/bin/python runs/finder-threads16-20261029/cloud.py status`。同切片93582行逻辑摘要与历史证据相同；9次试验的初始checkpoint摘要相同，所有试验46批，每线程数3次。同线程3次最终参数摘要相同；不同线程最终参数不逐位相同，逐批task loss最大绝对差2.384185791015625e-7。未修改既有15轮任务。

时间为3次交错试验的中位数，单位秒。各阶段中位数之和与总时间中位数可能略有不同。

| intra_op | 完整46批wall | 稳态41批读取/shuffle | 稳态数组+feed | 稳态sess.run | 稳态合计 | 相对4线程缩短 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 12.4031 | 1.4626 | 3.2472 | 2.9772 | 7.6867 | 0% |
| 8 | 12.2572 | 1.4719 | 3.2309 | 2.8615 | 7.5725 | 1.49% |
| 16 | 11.9506 | 1.4648 | 3.2313 | 2.8146 | 7.5108 | 2.29% |

排除前5批仅用于观察首次执行后的稳态；完整46批结果也保留，分别缩短1.18%/3.65%。完整wall区间：4线程12.005–12.698秒，8线程12.051–12.302秒，16线程11.936–12.390秒，存在重叠，不将小幅中位数改善解释为保证收益。数据准备一次32.025秒，不纳入各轮输入处理；未进行词表拟合，使用相同冻结完整词表。峰值进程RSS约915–918MiB，不含整机所有进程。模型较小的本切片不能推导完整epoch/多轮端到端同等提速。

结论：稳态输入处理约61%，增加TF计算线程不降低这部分耗时；16线程sess.run约快5.46%，总耗时仅约快2.29%。下一轮优化优先关注数组组装与输入/计算衔接；保持当前15轮一致性实验线程不变。结果：本地`runs/finder-threads16-20261029/{report,analysis,trials-summary,status}.json`与cloud.log。汇总校验断言通过：9次、每组3次、cpu_count=16、初始摘要一致、组内最终摘要一致；`git diff --check`通过。


## PR 合入验收（2026-10-29）

Source: 用户确认 opt15/prep15 一致并授权 TF 迁移、两项耗时优化经 PR 合入远端 main，随后同步本地 main。

本次从两次已完成任务只读下载冻结输入元数据、Feature State、每轮 checkpoint 与指标。运行本地 `runs/f024-pr-parity-verification/fetch_results.py` 和 `compare_results.py`：15 轮的 train_loss / validation_loss / 全部 validation_metrics 完全一致，每轮 327 个 checkpoint 变量的名称、形状、dtype 和字节完全一致；Manifest、record、Feature State 一致，`all_exact=true`。此证据覆盖 opt/prep 两版本的完整15轮输入/训练结果，不宣称 TF 与 Torch 数学等价，也不把15轮外推为任意运行条件的保证。原始数据、checkpoint 和实验报告未纳入 Git。

合入分支直接保留已验证的当前 TF 运行时代码，包括随机状态/梯度屏障修复及默认关闭的配置驱动正则；不带入旧 `recommend_tf/src` 原型。交付的两项优化分别为：训练批输入的列投影/NumPy列视图和有界RSS检查；训练前列式整理/时间验证与直接列批词表拟合。后者相对前者的15轮严格一致性由上述证据覆盖，前者另有先前切片与合成回归证据。

刚运行的检查：

- `UV_NO_SYNC=1 UV_CACHE_DIR=/tmp/doku-tf-pr-uv make tf-check`：41 passed，122项旧TF/Protobuf弃用警告，45.68秒；独立锁检查通过。使用既有已同步的隔离环境，避免修改其他工作区共享环境；CI采用正常锁文件安装。
- `ruff check . --no-cache`、`ruff format --check . --no-cache`：通过，265文件。
- `uv run mypy recommend`：70个源文件通过。
- `./build/build.sh`、`bash -n build/build.sh build/docker-build.sh`、`git diff --check`：通过。
- `UV_NO_SYNC=1 UV_CACHE_DIR=/tmp/doku-tf-pr-uv make check`：完整通过；原有训练测试 318 passed / 3 skipped（176.40秒），TF 测试 41 passed（41.42秒），两份锁、格式、lint、mypy 均通过。3项跳过为未配置本地 Redis 服务的既有集成测试，远端 CI 提供 Redis。远端 PR CI 作为合并前最后检查。
