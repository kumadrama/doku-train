# 验证证据

2026-10-23 本地 CPU 验证：

- `make check`：退出码 0；锁文件检查、165 个文件格式检查、lint、59 个源文件 mypy 全通过；pytest **184 passed, 3 skipped**，89.26 秒。包含非 performance 测试及 smoke。跳过项不计作通过。
- `.venv/bin/pytest recommend/train/tests/unit/test_nn_utils.py -q`：4 passed；覆盖极端 logits/梯度、指标独立 mask/标签、空集、树独立 loss 和树导入不加载 torch。
- `git diff --check`：通过。

初次完整检查发现树进程 SIGSEGV。移除指标层的 eager torch import 及树 runner 的 torch 线程初始化后，全套检查通过。树经注入函数调用自身的稳定 margin logloss，NN 仍统一通过 comm/nn_utils.py:get_loss。

未进行云训练或发布；旧实验指标未重写。
