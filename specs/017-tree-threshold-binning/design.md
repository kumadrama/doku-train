# 树分裂阈值分桶设计

Source: 用户确认需求及现有训练接口。当前原实现无 tree()；本 Spec 实现以下目标。

```yaml
tree_binning: {enabled: true, config_path: ./tree.yaml}
feature_transformations:
  - {expr: '[*]_tree = tree([*])', feature_list: [stat_a, stat_b]}
```

配置相对路径由所属 YAML 所在目录解析。关闭或省略 tree_binning 时原行为不变；tree() 必须启用前置训练。仅支持一个配置文件，具体模型必须为 rerank_tree。同一 Manifest/日期切分，前置树复用 NN 已冻结的输入，不重解析指针；统计值预处理由 YAML 保证，不做专门检查。树按自己的配置选模，测试不参与训练或阈值提取。

现有 train_rerank.py 作为 composition root；comm/tree_pretraining.py 接收注入的训练/导出函数，不依赖模型实现。模型专属的内存阈值提取在 models/rerank_tree/thresholds.py；存储读写由 tools/checkpoint/export_tree_buckets.py 负责，独立进程运行，避免模型层读存储或混合原生运行时。树运行在独立子进程中，资源预算/运行时不与 NN 混用，前置树指标不混入 NN TensorBoard。前置失败则父 Run 失败，不继续 NN。

合并 checkpoint 中全部任务、全部已保留树的数值 split_conditions；排除叶子值与类别节点。用 tree_inputs.numeric 的索引恢复特征名。tree-buckets.json 记录版本、float32_right 比较语义、全部数值特征边界、树配置及 lineage。无阈值时导出 []。

表达式首版只支持独立 RHS tree(field)，不得嵌套或传任意表达式。值转 float32 后按右侧插入位置分桶：等于边界进入右桶，越界进入首尾桶。先沿用 feature_default_values；未解决的 null 保留现有 OOV 语义，非有限值报错。每个表达式输出 int64。合法桶 0..K 全部登记，随后沿用现有词表索引 +2、padding=0、OOV=1，保持原 embedding 初始化与优化器。不新增桶专属 embedding 参数。

边界及来源冻结进 Feature State 并另存 tree-buckets.json。restore/test 使用 Feature State；resume 复制已有边界，不读取树 YAML、不重新训练树。合法桶索引随 state 持久化。SageMaker prepare 将树 YAML 解析内容写入校验过的提交包；容器不依赖本机 YAML 路径，归档包含分桶文件及带来源的特征状态。

初期不做复用缓存、叶子 ID、阈值剪枝、动态 embedding、多个教师 YAML 或推理导出。

## 配置接入示例

在现有 NN YAML 中增加以下声明，并将派生字段接入现有 embedding/input_features/dnn_inputs；下面仅展示新增部分，原有字段保留。

```yaml
tree_binning: {enabled: true, config_path: ./tree.yaml}
feature_transformations:
  - {expr: '[*]_tree = tree([*])', feature_list: [u_expose_count_7d_all]}
embed_name:
  u_expose_count_7d_all_tree: u_expose_count_7d_all_tree
input_features:
  tree_statistics:
    features: [u_expose_count_7d_all_tree]
    combiner: concat
# 将 tree_statistics 添加到现有 dnn_inputs；维度沿用 embed_dim 配置。
```

独立导出已完成模型（不重新训练）：

```sh
python -m recommend.train.tools.checkpoint.export_tree_buckets --run /path/to/tree-run --output /path/to/tree-buckets.json
```

输出为 create-only。首次 NN 训练自动完成前置树训练和导出；恢复训练/独立 test 不读取原树 YAML。后续需要发布/构建镜像或云实验时另行执行。
