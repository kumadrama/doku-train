# Spec 004：功能交付验证

Source: 2026-10-12 用户要求同步功能代码，排除实验配置、模型权重与相关产物。

## 本次交付范围

包含声明式配置、Manifest adapter、特征转换与历史 pooling、DNN/Finder 式 FM、CPU XGBoost、
独立训练/验证/测试入口、网格搜索、报告、checkpoint 恢复与配套合同测试。
本次不提交真实实验 YAML、指标报告、数据、权重或缓存；这些文件继续保留在本地。

## 可执行验证

```sh
UV_CACHE_DIR=/private/tmp/doku-uv-cache make check
git diff --check
```

测试配置由 `recommend/train/tests/synthetic_config.py` 独立提供，数据由
`experiment_fixture.py` 在临时目录生成。测试不读取 configs/ 或 Spec 中的实验 YAML。
覆盖三模型 CPU 训练与显式测试评估、九 trial 搜索、默认值与分桶、训练词表/OOV、
历史截断与带权 pooling、摘要与日期校验、指标不可计算记录、训练不读测试分片、
启用/禁用 dropout 的 epoch 恢复，以及 nullable 行为标签正负例。

真实实验记录已留在本地，不作为源码检出的验证依赖；这里的 smoke 均为合成样本，
不代表线上效果或真实测试集评估。

## 2026-10-12 结果

- 主工作区 `make check`：50 passed in 55.12s；lock、format、lint、mypy 通过。
- 仅暂存源码的独立 Git 快照：`UV_NO_SYNC=1 make check`，50 passed in 56.20s；
  不包含实验配置、报告、权重或缓存，使用相同锁定依赖。训练 lineage 要求有效 Git revision，
  因此隔离验证副本也初始化为本地 Git 快照。
- `git diff --cached --check`、提交文件排除检查、已修改文档相对链接检查全部通过。
