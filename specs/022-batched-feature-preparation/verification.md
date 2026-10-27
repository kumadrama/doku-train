# Spec 022：本地验证记录

Source: 2026-10-27 本会话实际执行；仅本地合成数据和模拟服务。未启动云训练或部署，未提交/发布模型。

## 一致性与集成

- `UV_NO_SYNC=1 UV_CACHE_DIR=/tmp/doku-train-uv-cache make check` 的 Python 主仓阶段：锁文件、格式（274 文件）、lint、mypy（69 源文件）通过；pytest **294 passed, 3 skipped，147.25 秒**。
- SageMaker 独立 test 阶段显式装配批计划后，`pytest recommend/train/tests/unit/test_sagemaker_platform.py -q`：**20 passed，16.23 秒**（全部本地模拟，不调用真实 AWS）。
- 最终编码分配前内存检查调整后，`pytest recommend/train/tests/unit/test_batch_feature_plan.py recommend/train/tests/smoke/test_batch_preparation.py -q`：**7 passed，8.65 秒**。
- `pytest recommend/train/tests/unit/test_batch_expressions.py -q`：**23 passed，0.11 秒**。

覆盖：新旧编码字段顺序、shape/dtype/字节内容、sorted 词表、fit 落盘状态与后续状态；训练两遍/验证测试两遍；多 reader worker；恢复状态不冒充已扫描证明；验证序列比训练长；过滤后空批/空集；带权历史重复/极端权重/排序；NULL、where 惰性、and/or 三值逻辑、对数分桶边界和超过 float64 范围的大整数；磁盘预算失败。新旧固定种子模型参数逐 Tensor 相等，训练历史/选模/验证/test 指标与逻辑 identity 相同。

计时覆盖更新为 batch_transform / batch_encode / column_decode，断言 train width_scan 不再出现。现有树分桶、resume、失败产物及 SageMaker 归档测试继续通过。源码身份校验保持，不宣称可以绕过旧 revision 的恢复校验。

## E1 配置覆盖（合成数据）

`PYTHONPATH=. .venv/bin/python runs/spec022-e1-synthetic/check.py` 通过。复用原 E1 冻结配置的全部特征转换、标签、mask、权重及历史规则，使用1,024条本地合成样本，替换为合成Manifest/日期/运行路径；未读取真实样本。train/validation/test过滤后726/121/121行，各141个输出字段逐字节一致，词表与最终状态一致。证据：`runs/spec022-e1-synthetic/results.json`（忽略目录）。

## 完整数据准备性能

基准源码：`recommend/train/tests/performance/benchmark_batch_preparation.py`，固定合成 fixture；不包含真实样本、云资源或模型训练。

方法：old/batch 各独立进程预热一次，随后交错顺序各运行三次取中位数；固定数据、批大小、线程设置。逐阶段计时并校验所有 split 数组 shape/dtype/SHA256 和拟合状态摘要相同；峰值 RSS 为进程总值。计时含校验、Parquet 解码、拟合、编码和 memmap 赋值，不用 fsync 强制物理落盘。

首轮独立基准：16,384 行、batch_size=2,048，额外64个合成对数桶列（来自同一计数列，非 E1 真实分布）。准备阶段旧版11.0718秒、新版2.4655秒，约4.49倍；进程峰值RSS中位数119,504,896与136,511,488字节。该轮不含reader/plan初始化，后续最终口径补入 setup，结果另列。

本地结果文件位于忽略目录 `runs/spec022-batch-benchmark-final/results.json`。不把微基准或合成完整准备收益外推为 E1 总训练节省；未验证28天容量或云端端到端性能。带权历史仍用旧循环，普通列计划仍包含Python标量内核，不能称为全原生向量化。

## 最终性能口径（包括初始化）

```sh
.venv/bin/python -m recommend.train.tests.performance.benchmark_batch_preparation \
  --output runs/spec022-batch-benchmark-with-setup --rows-per-day 2048
```

环境：macOS-26.5.1-arm64-arm-64bit，15个逻辑CPU，Python 3.12.13，NumPy 2.5.3。同上16,384条合成输入、2,048批大小、额外64个对数桶列，全部数组及拟合状态一致。

|项目|旧路径|批路径|
|---|---:|---:|
|setup 中位数/秒|0.0032|0.0041|
|fit 中位数/秒|2.1848|0.3032|
|train 中位数/秒|7.0191|0.8561|
|validation 中位数/秒|1.5722|0.6181|
|test 中位数/秒|1.5506|0.6232|
|总时间中位数/秒|12.3633|2.4434|
|进程峰值RSS中位数/MiB|112.78|130.36|

完整准备约 **5.06 倍**，时间减少约 **80.2%**。子阶段各自中位数之和不必等于总时间中位数。含初始化成本，排除产物SHA校验计时；输出SHA校验在两路径均执行且全部通过。该合成工作负载的重复表达式较多，不能外推E1、其他特征组合或28天容量。

## 提交前审查

修复批量 fit 后再次调用旧 fit 时残留训练扫描证明的问题：进入旧 fit 前清空 dataset 绑定与 widths，即使拟合失败也不能沿用旧证明。增加成功/失败两条回归测试；成功重拟合后训练准备重新执行宽度扫描。提交范围排除工作区中既有 TF、部署配置和其他 Spec 改动。

对数分桶边界测试分开覆盖正常批量内核与超大整数回退，避免单个溢出值使整批回退而掩盖优化路径覆盖。定向回归：`pytest recommend/train/tests/unit/test_batch_feature_plan.py -q` 为 8 passed；`pytest recommend/train/tests/unit/test_batch_expressions.py -q` 为 24 passed。

最终提交范围隔离验证：从 HEAD 导出仓库并只应用本次暂存差异，复用本地依赖环境，不包含未提交的 TF、Makefile 或部署改动。导出目录没有 Git 元数据，初次测试因无法读取 revision 失败；随后按现有镜像入口约定写入 HEAD 的 `build-revision.txt` 后重跑（该文件仅存在临时验证目录，不提交）：

```sh
UV_NO_SYNC=1 UV_CACHE_DIR=/tmp/doku-train-uv-cache make check
```

结果：锁文件、206 文件格式、lint、69 源文件类型检查全部通过；**297 passed, 3 skipped，147.46 秒**。7 份文档的 91 个本地链接存在，`git diff --cached --check` 通过。上述最终验证已包含本轮审查修复与正常/溢出分桶两组测试。
