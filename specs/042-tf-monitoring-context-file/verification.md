# 验证记录

Source: 2026-11-20 告警现场只读查询和本次本地实施。

现场 `logs filter-log-events` 确认 CreateTrainingJob ValidationException；`ecs describe-tasks` 确认失败 executor STOPPED/exit 1；`sagemaker describe-training-job` 返回 Requested resource not found。读取 journal 确认 active cutoff 2026-11-19/stage training；Endpoint InService，与 journal 的 cutoff 2026-11-18 online 配置一致。

本地与生产验证结果在实际执行后追加，不以实现完成代表生产恢复。

## 本地定向验证

在独立分支 `codex/tf-monitoring-context-file` 执行（使用主工作区已锁定的 Python 3.10 环境，PYTHONPATH 指向本分支）：

```sh
PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  /Users/yujiatang/doku/doku-train/recommend_tf/.venv/bin/python -m pytest \
  recommend_tf/train/tests/test_monitoring_context.py \
  recommend_tf/train/tests/test_daily_publication.py -q --disable-warnings
```

结果：46 passed，17.79s。覆盖完整大 JSON 往返、摘要篡改、协议冲突、512/513 长度边界、只读计划、原请求留档、冻结输入不变、已有作业拒绝、旧执行器仍运行/未知、权限错误、配置摘要损坏以及 journal CAS 竞争。

## 恢复操作

先构建并验证支持文件协议的训练镜像与编排镜像，取得不可变镜像 digest 和镜像内真实训练源码 SHA256。不得把工作区摘要误当作旧基底镜像打补丁后的摘要；生产现有训练摘要与当前仓库 HEAD 摘要不同。

```sh
AWS_PROFILE=EXPLICIT_PROFILE python -m recommend_tf.platforms.sagemaker.repair_monitoring_context \
  --binding /path/to/original-binding.json \
  --operation ORIGINAL_OPERATION \
  --training-image REGISTRY/IMAGE@sha256:VERIFIED_DIGEST \
  --source-sha256 VERIFIED_TRAIN_SOURCE_SHA256
```

输出原请求与候选请求。核对只包含监控传递字段、训练镜像及源码摘要变化后，同命令加 `--apply`。工具只修复 journal，不创建训练任务。随后通过原 deferred envelope 恢复：它会读取原绑定并计算原 operation，已修复 journal 提供实际训练请求。新日期使用新绑定/训练镜像并同时升级 worker；保留原 deferred envelope 的旧绑定，不改变原 operation。

当前未执行生产构建、部署、journal 写入或训练启动；上线验收仍需验证新请求被接受、上下文摘要正确、成功密封监控事件及既有发布流程完成。
