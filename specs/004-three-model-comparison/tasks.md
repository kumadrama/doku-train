# Spec 004 实施任务

Source: 用户已确认整体设计并授权本机训练与搜参实现。保持全部既有未提交修改。

- [x] 同步 rerank 合并方案与独立实验身份。
- [x] 运行时 schema、扁平表达式、配置语义校验，移除旧配置路径。
- [x] 版本化 Manifest adapter、显式切分、校验读取与训练独占特征拟合（fixture已验证）。
- [x] 默认值/log变换、共享词表、历史截断与pooling、树标量。
- [x] 动态任务 DNN/FM 与分类/回归 XGBoost，统一训练/评估/产物接口。
- [x] 独立搜参配置、9组展开、逐task对比和报告、独立实验身份。
- [x] 删除旧固定语义/门禁并迁移验证与文档。
- [x] 检查恢复、失败产物、泄漏、数值、存储及三模型fixture smoke。
- [x] 本机各跑一次XGBoost与DNN合成数据训练，另跑显式测试评估与报告。
- [x] 获得绑定真实v4的成熟性证据，完成192小时日志与778,014行标签归档核对。
- [x] 完成本机真实v4单次XGBoost与DNN训练；验证结果和产物摘要已核对，不启动整轮搜索。

- [x] 按用户后续授权完成真实4+4+1搜索（8次神经训练+1次基线复用），逐task验证报告、产物归档与摘要核对；测试集未评估。

未打勾不代表完成。真实数据执行结果单独记录，不以fixture结果替代。

## Numeric 桶 embedding 后续实施

- [x] 解析派生 embedding 的显式 cast 类型并写入 Feature State，保持原始输入合同。
- [x] 覆盖桶零/PAD/OOV、训练词表与恢复、DNN/DeepFM 前向梯度；make check 42 passed。
- [x] 每轮记录各 task 验证指标，保持原选模语义。
- [x] 本机各训练一次 DNN/DeepFM，与旧结果及 XGBoost 按 task 对比；不读取测试集。结果见 numeric-bucket-results.md。

## 2026-10-11 正则化实验

- [x] 支持 Finder 保留率语义 layer_dropout，默认 1；共享层与 task 隐藏层可配置。
- [x] 单测及 dropout resume 一致性验证，make check 47 passed。
- [x] A/B/C 三组独立 YAML，顺序各训练完整 20 轮，只使用训练/验证。
- [x] 汇总完整曲线与冻结 DeepFM/XGBoost 对比；三组未胜出，保留原基线配置。
