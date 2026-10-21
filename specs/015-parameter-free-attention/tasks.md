# Spec 015：实施任务

Source: 2026-10-21 已确认需求和设计。

- [x] 阅读 Finder 常规 attention 路径及本仓 pooling/checkpoint 接口。
- [x] 增加 rerank 专属 schema 与模型语义校验。
- [x] 独立实现参数无关 masked softmax 加权和并接入 DNN/FM。
- [x] 数值、空历史、padding、OOV、梯度、schema 和旧 pooling 单元测试。
- [x] DNN/DeepFM CPU train/test 与 epoch resume 验证。
- [x] make check、文档链接与 diff 检查；记录实际结果。
