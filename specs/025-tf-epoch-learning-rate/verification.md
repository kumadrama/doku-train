# 验证记录

Source: 2026-10-30 本次终端验证与原生 Linux 构建日志。实验配置和提交记录仅存本地，不随源码提交。

## 本地检查

- `make check` 的根目录锁检查、格式、lint、mypy 均通过；主仓 pytest 为 **318 passed, 3 skipped**（179.10 秒）。
- `make check` 的 TF 锁检查通过；随后 uv 重复下载 TF wheel 耗时，主动停止该下载阶段，因此不声明原始 make check 整条命令成功。
- 改用已有 TF Python 3.10 环境执行完全相同的 TF pytest 命令：`PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 <tf-python> -m pytest recommend_tf/train/tests -q -p no:cacheprovider --disable-warnings`，**62 passed**（76.50 秒）。该环境所属项目 uv.lock 与本仓逐字节一致，摘要 `b2763acbc6cb48f18bd9326b605e9bf7634a6d3a4ec05355023c8f8c983d0bd5`。上述为完整检查各组成项的实际执行证据。
- 新增调度测试独立执行 **21 passed**（39.21 秒）；`ruff check recommend_tf/train`、`ruff format --check recommend_tf/train`、`git diff --check` 通过。

## 行为验收

覆盖两种分段方案每一轮 LR、非法配置、线性插值、独立 sparse schedule；固定 LR 与等价常数调度的指标和完整 checkpoint 张量逐项一致。在两次衰减边界前恢复均与连续训练完全一致，包括优化器 slots；改变调度的恢复请求被拒绝。实际优化器入参、epoch JSON 和 TensorBoard LR 事件均核验。

## 目标环境验证

源码提交 `7366b2b07b47a4dc222969ae4338cb8fcaa6795b` 构建 CPU Linux amd64 镜像，通过 `DOKU_MODE=verify` 在断网容器执行完整 TF 测试：**62 passed**（129.67 秒），退出码 0。Python 3.10.22、TF 2.11.1、PyArrow 17.0.0；source/lock 摘要与冻结输入一致。构建及推送成功后才允许提交训练。
