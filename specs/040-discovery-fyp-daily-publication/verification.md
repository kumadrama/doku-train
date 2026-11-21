# 验证

可执行检查：

```sh
python idea-mapped-daily/build/package_source.py --pin
python -m pytest -q recommend_tf/inference/tests \
  recommend_tf/train/tests/test_release_preparation.py \
  recommend_tf/train/tests/test_daily_runtime_policy.py
```

本地结果：上述 32 项检查通过；仓库 inference-check 对应的 unittest discover 为 26 项通过。daily recovery/publication 的 TensorFlow 训练回归在当前 Python 3.12 环境缺少 TensorFlow，未执行；完整训练回归仍需仓库锁定的 Python 3.10 / TensorFlow 2.11 环境。

使用生产已发布的 2026-11-14、2026-11-15 两份真实模型完整包校验 canonical inventory，依次生成 Discovery profiles，并用未修改的 AI `cmd/releases --workload discovery` 加载成功。实际新模型身份分别为 `e3dbff6c83f096b9…`、`ea4d23f6e1622690…`，两份配置均保留原 bootstrap；这是本地合同回放，未调用线上预测或修改 AppConfig。

配套 IaC `envs/prod/tf_endpoint/discovery_appconfig.tf` 通过 Terraform validate；mock plan 两项通过，覆盖仅给原 FYP worker 发布两个 Discovery profiles 的权限及错误 profile 身份拒绝。未执行 apply。

## 2026-11-18 发布准备

用户授权 CI 通过后直接合入上线、维持 10%。PR #59 的完整 Train checks 和容器检查已通过；#931 已合入并通过平台 `dj_55f03c1d-bdea-47b6-9630-47ce9fa29e55` 应用，计划与执行均仅新增 Discovery 专用 IAM policy。

原生构建 `doku-train:5c04c557-df9b-4084-8618-7336df92ebc4` 成功，对 PR #61 审查源码 `69bc04f` 构建 pipeline runtime 并检查 publisher 导入及 SDK deployment CAS。不可变 digest `448e5ed9ee19621017a083b348d0f7450745bbb976961aa52cb610872abe0e1d`。激活前由 main 的标准服务构建核对推荐源码一致并推广为正式镜像。

用实际已应用 IaC 输出运行标准 binding 生成器，新 binding `239ae9ae06032249022557b92632262278d1b1f7d2adf7659ab21326ab3c9b85` 仅追加 `discovery_appconfig`。配方字节、训练/推理镜像、资源、Endpoint 均与正在运行版本一致。任务定义和 ready 生产者必须同步切到这一 binding：ready 的持久化 envelope 显式携带 binding URI，不能仅依赖新任务环境。

ready ZIP 从当前线上已校验 CodeSha256 的原包派生，仅替换 `recommend_tf/platforms/sagemaker/ready_release.json`；其余条目逐字节相等。新 ZIP SHA256 `3c3b5147410f6de7ee3124921828994693e12bfbca9ccc6e62da39c666ef125a`。通过原 dkd Lambda 目标激活，不改 handler、环境和规则。

线上初次发布发现 COMPLETE 后 Data 端短暂返回旧内容，已在 PR #61 增加保持同一发布 pending 的保护，34 项定向回归及完整仓库 CI 通过。人工初次引导沿用原 publisher 的持久化 journal，未重复创建部署；最终两个 AI 副本均热切到 `discovery-ea4d23f6e1622690-1caf106ee557`，预热耗时 61/84ms，无 reload 拒绝。实验 SHA 保持 `2155326f…`；12 条真实投递/180 个商品的新模型身份、冻结特征、KMS、Discovery 命名空间和 FYP parser 零接受均通过。该证据为当前日更版本采纳，不等于后续自然日更已完成。

ready 制品上传须声明 `ChecksumAlgorithm=SHA256` 和完整对象 `ChecksumSHA256`。首次 plan 因对象仅有默认 CRC32 校验头而被拒绝，未执行 Lambda apply。已为同一 SHA256/相同 ZIP 字节生成带 FULL_OBJECT SHA256 的新不可变版本，并用 HeadObject 回读校验；发布声明仅更正对象版本。

`test_discovery_appconfig.py` 覆盖连续三天更新、定义先于路由、旧列表混读、bootstrap 保留、合同差异拒绝、发布重试和无操作；通过原 daily advance 路径验证只更新原 endpoint，合成探测成功之前不写配置。

`test_idea_publication.py`、`test_idea_appconfig.py` 保证已有发布的 CAS 与重试语义；`test_official_pipeline.py` 及 daily recovery/publication 保证未配置 hook 的流程。

上线检查：模型 defs/list 发布完成后，各副本已有 `ai_engine.model_warmed`；实际 Discovery RPC alias 返回新的完整 release/version；同请求多批次身份一致，样本只写 Discovery 前缀。记录端点更新窗口的降级和 RT；实验保持 10%。当前仅代码与本地回放验证，不以测试替代线上采纳验收。

## 2026-11-19 FYP 日更接线

DTE `fyp-daily` 使用 18 条独立可播候选。AI Engine 同时加载 `84eb07ca` 和 `b710e4b2`，旧版、切换、回退各取 10 个 HTTP FYP 页面，以请求 ID 对齐调用日志和 AI Engine 真实打分版本。三轮服务器页面 RT P95 分别为 560、337、444 ms。每次 11 条候选完成打分，其余候选出现缺特征或剧集与剧目关联不一致。恢复 DTE 配置和泳道镜像后，回读确认测试候选、临时端点及专用调用策略已清理。

生产日更接线沿用现有 recipe、训练及预估制品和 EventBridge 任务。发布声明将主分支 `7ef89d5` 的新发布器镜像与仅追加 `fyp_appconfig` 的绑定钉死。Ready ZIP 仅更新同一发布声明。验收时读取实际 task definition、Ready Lambda CodeSha256 和持久化 envelope 参数，核对相同的配置及绑定 SHA，再按旧版和回退 alias 已就绪、全量副本已加载、真实打分版本和 RT 的顺序验证生产切换。

## 2026-11-20 三路实验自动切流

- 32 个推理和发布测试以及 7 个绑定和运行策略测试通过。测试覆盖三个别名、独立 Endpoint、AppConfig 发布顺序、配置读取等待、发布锁恢复以及副本加载超时后的回滚。Terraform validate 和四个 mock plan 测试通过。
- 使用生产正在服务的三份模型制品与 AppConfig 做只读合同回放。首轮使用实际模型版本，随后三轮仅在本地换版本元数据，验证连续发布的配置容量。四轮模型定义大小分别为 1,336,006、1,682,576、1,682,576、1,682,576 字节，均在 2 MiB 内。
- 上线后仍须通过真实 Endpoint 探测、所有副本的 `model_generation_ready` 事件和线上打分版本验收。测试结果覆盖配置和发布行为，生产实流量验收在 worker 激活后执行。

## FYP 固定 Endpoint

`python -m unittest discover -s recommend_tf/inference/tests` 覆盖共享主模型首次启用前的回退、实验模型先切走旧式固定地址再迁回、两套签名的推理探测、版本不符时恢复 EndpointConfig，以及固定调用模式连续日更复用原 AppConfig。

投产验收读取实际 Endpoint、AppConfig caller 和 ECS 目标，观察四个别名的业务打分结果、响应中的实际版本和样本身份。当前验证范围为代码及合成回归。
