# 验证记录

Source: 2026-10-28 本地执行；未构建或发布 PyTorch 训练镜像，未提交云训练。

- `.venv/bin/pytest recommend/train/tests/unit/test_learning_rate.py recommend/train/tests/smoke/test_declarative_train.py recommend/train/tests/unit/test_sagemaker_platform.py -q`：49 passed（119.57 秒）。
- `.venv/bin/ruff check recommend --fix`：通过；`.venv/bin/ruff format --check recommend`：113 files already formatted。
- `.venv/bin/mypy recommend`：70 source files 通过。
- 使用已有 E1 lr=.004 配置加上 1/10/30 节点，经 validate_config 校验通过；epoch 1/10/11/20/30/40 输出 .004/.004/.00385/.0025/.001/.001。
- 恢复 smoke 覆盖固定、线性和阶梯学习率，以及 dropout、attention、分组 weight decay；恢复与连续执行 history 完全一致，选中模型参数逐 tensor 相等。
- EventAccumulator 验证 train/learning_rate 使用 epoch=2 而非 global step=123；旧 loss 曲线继续使用 global step。
- `make check`：lockcheck、全仓 format/lint、PyTorch mypy 通过；PyTorch 回归 313 passed、3 skipped（179.53秒）。TF 检查初次因 Docker credential helper 不在 PATH 失败，补齐 Docker bin 路径后单独重试 check-tf。

- TF 重试命令：`PATH=/Applications/Docker.app/Contents/Resources/bin:$PATH make check-tf`；依赖安装成功，验证阶段 QEMU signal 11 / exit 139，未完成 TF 验证。本次未修改 TF 代码，不将根 make check 声明为全部通过。
- `git diff --check` 通过。
