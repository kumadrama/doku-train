# Spec 024 原生 Linux SageMaker 验收

状态：通过。Source: 用户明确授权最小 SageMaker 合成验收；SageMaker DescribeTrainingJob、CloudWatch日志及下载的output.tar.gz/pytest.xml/verification.json交叉核对。

- 任务：[doku-exp-finder024-20261028-155439](https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/jobs/doku-exp-finder024-20261028-155439)。
- 区域 us-east-2，单实例 ml.m5.large，30GB卷，最长600秒，网络隔离开启，无生产数据输入。
- 最终状态 Completed；11 tests、0 failures、0 errors、0 skipped。
- Linux x86_64、Python3.10.22、tensorflow-cpu2.11.1、PyArrow17.0.0。
- pytest耗时17.34秒，验收入口总耗时20.76秒；SageMaker TrainingTime及BillableTime均73秒。该时间不包含CodeBuild计费，也不是金额账单。
- 镜像：`703844615824.dkr.ecr.us-east-2.amazonaws.com/doku-train/doku-reco-train@sha256:79f04258a1213a30279f832adfafd4aa788dae9cad2c3e22da31ed9d87a39999`。
- 源码摘要：`50dec46b4215f76ed3e85c543993270624d224b59afbf6929f99fc0f6bec9219`。
- 锁文件摘要：`b2763acbc6cb48f18bd9326b605e9bf7634a6d3a4ec05355023c8f8c983d0bd5`。

## 覆盖与证据

云端完整执行与本地相同的11项测试：Parquet与Manifest合同输入、基础分片、七任务训练/评估、启用dropout的epoch恢复、独立test评估、Finder loss分母、AdamW dense/稀疏衰减、padding梯度、weighted_mean、OOV共享表及无标签PREDICT。
源码和锁文件摘要在容器启动时检查，下载的JUnit报告再次核对计数。操作与报告保存在本地忽略目录 runs/spec024-cloud-acceptance-02，未提交运行产物或模型权重。

## 构建过程

复用已有doku-train CodeBuild项目、ECR仓库和SageMaker角色，不修改IAM/网络/调度。
首次CodeBuild因Docker Hub拉取python:3.10-slim遭HTTP429而失败，未启动SageMaker任务。
改用public.ecr.aws/docker/library/python:3.10-slim后构建成功；依赖由uv.lock导出并以pip --require-hashes安装。
成功构建ID：doku-train:69858735-a372-4b62-ba9d-43d525ec0ff9。
只提交一个SageMaker验收任务，任务已完成，不存在本次仍运行的训练实例。

## 结论边界

原生Linux完整合成链路已通过，取代此前“Linux完整链路待验收”的状态。
Mac本地amd64模拟器中的Arrow导入segfault仍是未定位的模拟路径问题；不据此断言根因，但它没有在本次原生Linux环境重现。
本次不验证真实e1输入、40epoch容量、模型效果或提速，也未发布线上模型。
下一阶段仍需明确实际实验Finder优化器参数，再另行授权真实数据切片实验。
