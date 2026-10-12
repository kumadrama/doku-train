# 数据与状态模型

Source: Spec004、published_dataset.py及feature_plan.py。

Dataset由精确Manifest URI、逻辑摘要、dataset/feature/label版本、分片与合同身份描述。
Run保存源码revision、未提交源码摘要、依赖锁摘要、配置、切分、词表和checkpoint摘要。

Feature State记录原始类型、共享表映射、训练词表、OOV/padding约定及形状元数据。
默认值只作用于原始features；计数log1p生成_log，不拟合标准化，不添加_missing输入。
候选与历史共享drama表，题材共用genre表；仅训练样本拟合词表。

Metrics按task保存值、有效样本数、不可计算原因；单类AUC为null，不影响logloss或训练成功。
训练状态、选模状态和评估状态分别记录，没有AUC阈值或质量门禁。

源Manifest的as_of_ms不是历史完整性证明。不得把固定D-2或24小时等待当作本数据的成熟性依据。
