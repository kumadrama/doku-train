# 接口合同

Source: Spec004及实现。旧格式、固定四目标、固定28天与refit生命周期已移除。

配置：公共/模型config.schema.yml运行时合成；搜参独立search.schema.yml。未知字段报错。
模型工厂接收展开配置和feature state，返回fit/evaluate/load/checkpoint_name接口。
模型不读取环境配置、不访问上游存储；checkpoint写入由训练用例提供的本地run目录。

数据：当前版本adapter读取doku-train-dataset-manifest/v1；校验逻辑SHA256、合同版本、
指定分片SHA256/行数/事件时间，按显式train/validation/test自然日选择。
源合同对象摘要写入lineage；源v1未提供合同摘要及完整成熟性关联，不能因此声称已证明成熟性。
真实v4实验区间已完成独立历史证据核查；数据集名称不作为训练白名单。
详情见真实数据核查（本地实验记录，不随源码发布）。

模型产物：checkpoint.pt或checkpoint.xgb.zip、metrics.json、fitted-feature-state/state.json、
lineage.json和status.json；另有resolved-config、resume及编码缓存。status=complete最后写出，
失败仅记录failure，不覆盖原run。测试评估显式产生test-metrics.json，不参与训练选模。

分类预测均为概率；树使用raw margin后应用一次sigmoid。神经soft label预测按scale还原秒数。
比较分类logloss/AUC、回归MSE；训练loss和可比较指标单列，不跨模型直接比较不同目标函数。
