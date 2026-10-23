# 验证

Source: 2026-10-20 本次执行记录。

- `UV_CACHE_DIR=/private/tmp/doku-train-uv-cache make check`：exit 0，181 passed、3 skipped；格式、lint、mypy通过。日志 `/private/tmp/tree-check-isolated.log`。
- SageMaker专项：11 passed。神经/树测试在Mac同进程触发OpenMP卡住/崩溃，树验收使用独立子进程；不改变实验4线程。
- 生成47份配置全部schema通过；原始数据审计确认train/validation三组无序drama集合均不超过100，E04跳过，本轮46 Run。
- 验证集138001行，episode watched命中191、skipped31、finished0；如实报告覆盖稀疏，不预判收益。
- 本地amd64模拟构建uv进程exit139，改用现有CodeBuild原生构建；云端训练尚未启动，不声称通过。
- 所有真实配置/审计输出位于忽略目录 `runs/tree-sagemaker-20261020/`。

## 云端执行启动

- 原生 CodeBuild `cf2118a0-101f-4136-b90a-443144167fcc` SUCCEEDED；dnn/deepfm/xgboost 三模型容器 smoke 均输出 PASS machine=x86_64。
- E00 `doku-exp-tree-20261020-e00` 已进入训练并通过 freeze_input；未完成前不声称成功。
- 本地冻结 Manifest/合同原字节复制至授权实验 submissions 前缀，分片与content摘要不变；未重新解析每日指针。
- 查询得到 m5.2xlarge 训练实例配额30，本轮有界并发4。
- 分群真实数据验证：missing0、unseen1124、low6084、medium23283、high107510，总数138001；报告工具验证7任务输出。
- 原基线新增episode命中合并222条；top200新增覆盖1637条；不据此宣称指标收益。

- 独立分群工具all切片与原始保存validation指标逐项比较，7任务全部在1e-7相对/1e-8绝对容差内一致。分群报告烟测输出232条指标，基线delta均为0。

## 长任务交接

- E00完成fit_features耗时552.091秒，进入encode_train；尚未完成。其余45个Run已生成并验证提交包，尚未批量启动。
- `runs/tree-sagemaker-20261020/run_remaining.py` 等待基线成功后，4并发执行其余实验，收集产物、生成分群与汇总；日志在同目录batch-execution.log。
- 当前对话自动跟进ID `sagemaker`，每15分钟检查，未变化保持安静，完成/失败才通知；交付后删除。不要重复提交新任务，恢复须沿用prepared包和job名。

## 2026-10-21 连接中断恢复

- 云端查询确认前26个Run全部Completed；本地控制器两次因SSL EOF退出，云端训练未失败。原始错误保存在忽略目录batch-execution-ssl-failure*.log。
- 同一Python SDK实测STS身份正确、SageMaker Completed、S3 head_object HTTP200，确认当前连接恢复。
- 实验提交器新增最多3次传输异常恢复，沿用同名prepared包与不可变校验；训练失败、权限及配置错误不重试，不修改容器或训练语义。
- ruff format/check通过；注入异常验证瞬时恢复、最多4次调用及非网络错误直接抛出均PASS。恢复原编排，仍保持4并发、46个实验范围。

- 用户要求全部剩余并发：核查m5.2xlarge配额30、原占用4，设置parallel=20。提交到17个活动任务时系统代理127.0.0.1:17891失败；使用proxies={}实测AWS STS/SageMaker直连成功，随后通过NO_PROXY=*恢复原编排。代理故障日志单独保留，不停止云端任务。

## 2026-10-21 最终验收

- AWS list-training-jobs实查本批46个全部Completed；没有启动第二轮或test/发布。
- collect/diagnose/report原编排完成，complete.json为46；46个归档逐一verify_archive通过，46份分群JSON，group-validation.csv共2320行。
- 独立逐项比较每Run全量分群与保存validation指标，290项在rel_tol=1e-7、abs_tol=1e-8内通过。7个T00与E00对应Booster字节相同（report.py检查）。
- 参数推荐按声明的验证logloss/RMSE：completion、fast_swipe、immersive_click、play10、play30选T04；play50选T05；watch_seconds选T02。特征取舍留用户，无自动组合/发布。
- 产物和报告位于runs/tree-sagemaker-20261020/{artifacts,reports,report.md,group-validation.csv}，S3地址逐Run保存在cloud/state.json和artifacts/validation.csv。
- 本轮平台完整检查此前make check为181 passed/3 skipped，原生amd64三模型smoke通过；后续仅实验控制器调整，ruff check与网络恢复异常注入验证通过；最终git diff --check通过。

## 2026-10-21 第 2–6 项隔离交付验证

基于 origin/main `2344c0a` 整理平台树模型、阶段选择、指标、测试、PROD binding 与 Spec008/014；不迁入 experiments 工具。Spec009/013 保留 main 全部记录，避免以旧工作区描述覆盖已发布证据。根 AGENTS 仅补充本 Spec 的交付索引。

- `UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/Users/yujiatang/doku/doku-train/.venv UV_CACHE_DIR=/tmp/doku-train-uv-cache UV_OFFLINE=1 make check`：exit 0；69 包锁文件、151 文件格式、Ruff、mypy 57 源文件通过；181 passed、3 skipped，83.91 秒。
- `git diff --check` 通过；AGENTS 与 Spec008/014 Markdown 本地链接检查通过。
- 本轮未重新执行容器构建或云训练；前文云端记录为历史证据。
