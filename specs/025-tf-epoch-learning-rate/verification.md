# 验证记录

- 本地 Python 3.10 / TF 2.11，执行 `PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 <tf-python> -m pytest recommend_tf/train/tests/test_learning_rate.py -q -p no:cacheprovider --disable-warnings`：21 passed（39.21 秒）。
- 覆盖 A/B 每一轮 LR、非法配置、线性插值、独立 sparse schedule；固定 LR 与等价常数调度的指标和完整 checkpoint 张量逐项一致；在两次衰减边界前恢复均与连续训练完全一致，包括优化器 slots。实际优化器入参和 TensorBoard LR 事件均核验。
- `ruff check recommend_tf/train`、`ruff format --check recommend_tf/train`、`git diff --check` 通过。
- `make check` 正在执行；Linux amd64 镜像全套合成验证和 A/B 请求回读待补充。
