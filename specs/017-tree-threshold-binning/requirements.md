# 树分裂阈值分桶需求

Source: 2026-10-23 用户通过 grilling 确认 Q1–Q8，并要求按方案实施。

- NN 配置引用一份树模型 YAML；启用后先按其配置训练，再导出数值分裂阈值，随后开始 NN。
- 合并这份 YAML 全部任务已保存 Booster 的数值阈值，排序去重，不设置桶数上限。
- 无分裂的数值特征仍导出空边界，表示一个桶。
- feature_transformations 支持 `[*]_tree = tree([*])`；引用导出文件没有的特征立即报错。
- 用户用 YAML 保证统计特征不做预处理；代码不检查这一限制、不自动反算或改名。
- 全部合法桶预分配 embedding 索引，沿用 initializer/optimizer；不冻结未出现桶。
- schema、特征状态、resume/test 及 SageMaker 打包同步支持。当前没有启动新云训练的授权。
