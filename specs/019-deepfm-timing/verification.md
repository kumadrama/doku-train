# Spec 019：验证记录

Source: 2026-10-27 本地执行证据；全部数据为既有合成 fixture，无云训练/部署。

## 基线与范围

- `git fetch origin main` 成功；从 `origin/main` 的 c03ba8c107566de31c3bff7f16e2725fd08ba97d 创建 `codex/deepfm-timing`。
- 保留原有 Spec 008/018 verification 未提交修改；未纳入本次功能改写。
- 先创建本 Spec 需求、设计、任务与索引，再实施代码。
- 仅新增观测，没有缓存、扫描删减、批量编码、loss 合并、线程调整等性能优化。

## 执行结果

1. `.venv/bin/pytest recommend/train/tests/unit/test_timing.py recommend/train/tests/smoke/test_deepfm_timing.py -q`：初版 8 passed，7.04 秒。
2. `.venv/bin/pytest recommend/train/tests/smoke/test_declarative_train.py -q -k 'weight_decay or new_cpu_training'`：8 passed、5 deselected，27.84 秒。
3. `.venv/bin/pytest recommend/train/tests/unit/test_timing.py recommend/train/tests/unit/test_sagemaker_platform.py recommend/train/tests/smoke/test_deepfm_timing.py -q`：24 passed，18.78 秒。
4. `UV_NO_SYNC=1 UV_CACHE_DIR=/tmp/doku-train-uv-cache make check`：通过；uv lock、196 文件格式、lint、67 源文件 mypy 均通过；259 passed、3 skipped，130.11 秒。使用现有锁定虚拟环境，不重新安装依赖。
5. 全量检查后补充平台事件上传异常时的 finally 诊断保留，并新增对应测试；最终 `.venv/bin/ruff check .`、`.venv/bin/mypy recommend` 通过；`.venv/bin/pytest recommend/train/tests/unit/test_sagemaker_platform.py -q`：16 passed，15.35 秒。
6. `git diff --check` 及本次修改的文档本地 Markdown 链接检查通过（AGENTS、README、components 和 Spec 019 四份文档，共 7 份）。

## 证据覆盖

- 关闭计时：用禁止调用的时钟替身证明不读取计时器时钟、不生成目录/文件、不打印新增日志。
- 固定时钟：父子累计不重复相加、60 秒进度边界、epoch 累计及增量。
- 生成器：consumer 模拟耗时不计入 next() 的读取耗时；显式 collector 支持 worker 累计。
- 配置：禁止 print_logs=true 且 enabled=false；拒绝 DNN 开启与树模型 profiling 配置。
- DeepFM：开关前后 checkpoint 模型张量逐项完全相同；history/selection/validation 和逻辑 identity 相同。
- 恢复：未开启计时的中断训练，可以开启计时恢复；结果与完整训练一致。
- 输出：仅采集模式无 DOKU_TIMING stdout；子进程开启打印时转发 timing 日志；错误退出码仍检查。
- 独立 test 两次生成独立 invocation，指标与关闭模式一致；训练失败有 failed summary 且不声明成功。
- 真实合成树前置训练：仅父 DeepFM 有 tree_pretraining 总耗时，teacher 不创建 timing 目录，父记录无树内部子项。
- SageMaker：train/test/archive 三类计时文件进入归档清单并通过 checksum 校验；训练失败、事件上传失败仍保留 output/data/timing；失败不写成功归档清单。

## 限制

计时有运行开销，尚未在真实规模上测其比例；不声明吞吐改善或生产完成时长。
周期进度在处理边界触发，长时间 native 调用内没有额外后台心跳。
下载包含本地写入，Parquet batch 包含读取/解压/解码；数组赋值不表示物理落盘。
父阶段包含子项、并行 worker 可重叠；平台退出后的压缩上传不在容器计时范围。
普通异常保留记录，SIGKILL/断电不能保证最后 summary，已 flush 的 JSONL 供诊断。


## 提交前审查修复

Source: 2026-10-27 用户要求审查后提交并创建 PR，不部署。

- 修复并行分片读取的 ContextVar 传播：每个 worker 继承提交时的独立上下文，下载/校验计时保留正确父阶段；并行读取的开关对照 smoke 覆盖参数与指标一致。
- 补齐开启 profiling 的归档完整性校验：要求 train、配置要求的 test、archive 成功 summary 及匹配 JSONL；缺少某一阶段或 JSONL 会拒绝，计时文件继续纳入 checksum。
- `.venv/bin/pytest recommend/train/tests/unit/test_sagemaker_platform.py recommend/train/tests/smoke/test_deepfm_timing.py -q`：23 passed，20.87 秒；ruff、mypy 通过。
- 提交前再次 `git fetch origin main`：HEAD 与 origin/main 仍为 c03ba8c，左右差异 0/0。

- 审查修复后的最终 `UV_NO_SYNC=1 UV_CACHE_DIR=/tmp/doku-train-uv-cache make check` 全部通过：264 passed、3 skipped，137.99 秒；锁文件、196 文件格式、lint、67 源文件 mypy 均通过。`git diff --check` 和 7 份文档本地链接检查通过。
