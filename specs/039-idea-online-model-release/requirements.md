# Idea 日训复用 FYP 在线发布链路

Source: 2026-11-17 用户要求直接复用已经验证的 FYP 发布方式，快速恢复稳定；这条指示替代此前双模型共包/新增观察状态机的设计。现有 FYP 与 doku-ai-engine 不改。

当前问题是 Idea 日训原地替换同一 Endpoint 的 named SavedModel，而在线 AppConfig 仍引用旧模型版本。复用 FYP 11 月 7 日实际发布先例：独立版本 Endpoint、保留旧定义、先 models config 后 model list、AI 既有热加载。

- 仅增加 Idea 可选接线；复用现有模型封包、deploy_step 和合成探测，不另建推理或发布架构。
- 每个版本使用独立 Endpoint，禁止覆盖已存在的不同版本 Endpoint；旧 Endpoint 不由本次流程删除。
- 保留当前运行预算、特征 DAG、输入/输出合同、不可变 release 与快照模型身份。只替换新模型制品、版本和 Endpoint 引用。
- 配置更新使用原内容校验、hosted-version CAS、deployment-number CAS 和同一发布 journal，失败不得推进 high_water 或覆盖外部更改。
- 部署完成表示 Endpoint 合成探测与 AppConfig 发布完成，不等价于每个副本已采纳。AI 现有校验/旧代保留机制继续负责热加载，首次上线和自然日训实际效果由 RPC/日志核验。
- 首次由外部已授权发布把当前 5d9e 路由迁到独立 Endpoint，并移除旧 mutable selector；后续日训保留的上一版指向独立 Endpoint。
