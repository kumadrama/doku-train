# Spec 022：批量特征准备设计

Source: [已确认需求范围](requirements.md)；当前 recommend/train/comm/{expressions,feature_plan,training_pipeline}.py、comm/datasvr/published_dataset.py；本会话本地原型与 E1 计时证据。

状态：2026-10-27 用户确认并授权实施；本地实现已完成，验证状态见 [验证](verification.md)。未部署。

## 1. 当前状态与目标数据流

优化前：原始数据 → Python 行字典 → 每行 AST 求值 → 每行编码小数组 → 逐字段逐行 memmap 赋值。AST parse 已缓存。fit 按声明 list 类型统计长度，而 width_scan 按运行时 list 值统计，二者需对齐后才能省扫描。

当前本地实现的数据流：

```text
配置/合同 → 白名单 AST 校验 → 批量执行计划（每个 FeaturePlan 一次）
Arrow batch → 原有校验/过滤顺序 → 默认值/历史处理 → 批量转换
  ├─ fit：收集唯一值、序列长度 → sorted 词表
  └─ encode：固定词表映射、任务表达式 → 每字段批数组 → memmap 块赋值
```

[INFERRED] 采用 NumPy/Arrow CPU 后端最贴合当前 memmap 边界；不增加 TF、torch.compile 或自定义 C++ 依赖。执行计划仍由 Python 按表达式节点调度，但每节点处理整列，不为每条样本遍历 AST；不声称自动算子融合或机器码编译。

## 2. 模块接口与接入范围

- batch_expressions.py：新增 compile_batch_expression，返回只含允许操作的 BatchExpression；保留 evaluate 作为语义基准及局部兼容执行器。禁用直接 eval/exec 配置字符串。
- reader：新增 iter_batches，复用分片下载、摘要、schema、行数和时间边界校验。现有 rows 及其调用方行为保持；不能为批路径绕开数据验证。
- BatchFeaturePlan：构建批计划，提供 transform_batch、fit_dataset、encode_batch；继承 FeaturePlan 的历史算法。历史局部逐行处理与列运算协作，避免为了历史字段把全部数值列 to_pylist。
- BatchFeaturePlan.prepare_dataset：批数组形状校验、容量预算及按块赋值；保留空有效集报错、create-only 目录和错误传播。
- training_pipeline：通过明确的数据准备接口选择优化路径，不加入具体模型名称分支。首版仅接入获准的 PyTorch 神经网络准备路径，独立树调用方继续既有路径；接口选择在用例装配层显式完成，不能意外扩展共用函数的影响范围。

批容量首版沿用现有显式 config.batch_size（reader 当前也使用它），绝不修改训练 batch 值。与优化器步数无关，过滤后的有效行数单独计数；未来独立调整准备批容量需另行设计配置，不静默按机器探测调参。长列表及中间列仍计入现有内存预算，超预算如实失败，不改样本语义。

## 3. 训练宽度复用

抽取与旧 width_scan 等价的长度收集规则，在 fit 的已转换有效样本上收集完整分配宽度；显式历史容量仍按 max_hist_size 控制。不得顺手把旧可接受的运行时列表改为新类型错误。

使用运行内 FitScanSummary，绑定当前 plan、训练 split 及冻结输入身份，显式标记已覆盖字段。只有完整证明成立时跳过 train width_scan；其他调用保留旧路径。继续检查编码溢出/shape，不能用自动截断掩盖缺失统计。

注意既有状态：fit 的 lengths、state() 字典引用及落盘时序并不完全分离；validation 宽度当前在既有 lengths 上取最大值，不是仅使用 validation 最大值。实现前锁定旧路径落盘状态、模型看到的状态和逻辑 identity 对照。额外分配宽度放运行内元数据；不将新标记写入 Feature State，不顺便重构状态/身份合同。

## 4. 表达式执行计划

批值包含 values、validity 和语义类型；常量按批广播。初始化阶段解析依赖及已知函数，生成列读取、运算、条件子计划和输出槽位。依配置顺序执行转换，不重排可能报错或有依赖的节点。

- item_filter 读取默认值填充前的输入；被过滤行不执行后续特征校验。
- ID/weight 成对有效性和长度校验保留在默认值之前；列表元素 NULL 与多列历史对齐规则不变。
- where 按 active_rows 分别执行选中子计划，禁止提前计算两侧；mask=false 不执行标签/weight，自定义指标 filter=false 不执行该指标标签。
- 当前 and/or 会计算全部子表达式并采用三值逻辑；不可改成 Python 短路。
- 算术保留 NULL 传播及非有限结果拒绝。当前 cast 的 Python 标量语义与最终存储 dtype 区分，不能直接引入 int 溢出或 float32 中间舍入。
- 对数分桶：使用本地原型的批操作与接近 round 半整数边界时的标量校正候选。对超出安全整数计算范围的数据保留旧算术；边界保护不是跨平台一致性的证明，需在目标 CPU/Python/NumPy 环境对照。
- tree 表达式若未验证批实现则局部调用旧实现；保留原 float32 阈值转换及 bisect_right 边界。不得删除现有树分桶特征能力。
- 不支持的节点可编译为 ScalarFallback，按 active_rows 调用旧求值器；仅回退必要表达式，统计表达式类别、处理行数和时间，不输出真实特征值。随机/有副作用表达式不擅自做公共子表达式消除。

## 5. 历史与数组写入

top_weight 完整沿用旧执行顺序和 Python 求和行为，不使用原型中的 np.unique/跨批排序。tail 历史保持顺序、重复值及 ID/weight 同步截断；批量切片只有对照通过才替换。整理为当前编码要求的容量，固定 padding/OOV。

每字段创建 [有效批行数, 原字段shape...] 数组，词表按旧 sorted 唯一值生成；映射不能把未知 ID 当作新词。labels/masks/weights/metrics 全部保留原字段和 dtype。写入 arrays[key][cursor:cursor+n]，cursor 只增加有效行数。空批跳过，全局空集仍失败。尾批、过滤造成的空洞、跨分片批次不得改变输出顺序。

保持 memmap 容量和 cache_limit_bytes 检查，不新增 fsync、不承诺物理写盘时间。局部历史循环允许保留；不能为了宣称全批量化接受性能回退。

## 6. 计时与验证方案

保留 Spec 019 父阶段和默认关闭开关，新增批转换/批编码/块赋值/局部回退统计；明确其与旧行计时的口径差异。skip 的 train width_scan 应缺席或有显式跳过记录，不能伪造零耗时扫描事件。父子时长仍不可相加。

实施后必须提供可执行自动化证据：

1. 表达式差分测试：固定种子、NULL/异常、边界值及 active_rows，与 evaluate 对照；整数/类别完全相同，浮点输出逐位相同。
2. 数据准备差分测试：同一合成 Manifest 执行旧/新路径，比较顺序、有效行数、词表、长度、所有数组及状态；用计数 reader 断言 train 两遍。
3. 集成验证：训练/恢复/test、树分桶特征、profiling 开关、独立树旧路径不变；同环境固定种子对比模型参数与指标。
4. 基准：旧/新完整准备同输入、同批大小，预热后重复至少三次取中位数，统计所有表达式和历史路径，包含转换/分配/写入，不只计单个 NumPy 算子；独立进程测 RSS。
5. 正式实施完成后运行 make check，记录实际命令及结果。实际新增测试与命令结果见 verification.md。

云端端到端验证必须另获授权。此前30–34分钟扫描推算以及微基准21倍都不作为端到端SLA。若局部批实现不能兼顾语义和性能，保留旧步骤并报告覆盖率，而不是放松已确认的语义要求。

## 7. Finder 参考边界

Source: finder-models/recommend/train/models/listrank/rerank_model.py 的 transform_features/model_fn；finder-train/recommend/train/comm/common_utils.py 的 eval_expr 和 Tensor 函数映射。

已确认 Finder 通过 Python 构建 Tensor 运算，避免每条样本重复标量解释；本设计借鉴执行边界，不复制其私有服务代码。Finder attention/longhist2 的带权 embedding 聚合与本仓准备阶段 top_weight 去重排序不是同一操作；尚未找到相同预处理链，不能据此宣称 Finder 已解决此热点。

## 实施细化

- reader 首版通过 Arrow to_pydict 解码为 Python 标量列，再构造一维 object 数组；不创建全字段逐样本字典。这不是零复制 Arrow/native 全链路，类型保真优先。
- 普通表达式预编译为列执行闭包，使用 object ufunc 保留 Python 大整数、math 函数、cast 及NULL语义。每个运算仍可能调用标量内核，但不再每行解释 AST；不称作全原生向量化。
- 常见 `cast(clip(round(log2(maximum(field + 1, 1))), 0, 31), "int64")` 使用按 AST 结构识别的 NumPy 批内核（不硬编码字段名）；边界附近用旧表达式校正，超出 float64 转换范围回退。其他 log 输出保持原 math.log2，不以近似中间值替代原结果。
- 训练宽度复用证明由成功 fit_dataset 后绑定的当前 dataset 对象及独立 widths 完整结果承载；初始化/restore/旧 fit 调用（含失败）清空证明，换 dataset 不复用。额外宽度在原 train 编码时点应用，不提前污染落盘 state。
- batch_transform、history_rows、history_scalar_rows、batch_encode、column_decode 是新计时项；array_assign 的单位由调用改为批内有效行数。旧路径的 row_transform/encode_row/to_pylist 仍保留，不能跨口径直接比较。
