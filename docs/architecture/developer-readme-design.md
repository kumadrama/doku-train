# 开发者 README 信息架构

## 状态

Implemented。根 README、可执行模块 CLI 与合同测试已按本设计落地。

Source:

- 用户选择算法开发者作为 README 的主要受众；
- 用户确认首要成功路径是配置 S3 数据并启动一次真实训练；
- 用户选择 S3-first 操作手册型 README；
- [系统概览](system-overview.md)；
- [接口合同](interfaces.md)。

## 目标

根 `README.md` 应让首次接触仓库的算法开发者在不阅读实现源码的情况下完成以下动作：

1. 判断 `doku-train` 是否负责自己的需求；
2. 准备 Python 3.12、`uv` 和可访问 S3 的调用身份；
3. 理解 Dataset Manifest 与训练配置的最小要求；
4. 先执行 `validate`，再启动真实的 `train`；
5. 识别四类训练输出、软硬门禁和失败阶段；
6. 找到架构、接口、Spec 和本地验证文档。

## 信息顺序

README 固定采用以下顺序：

1. 一句话定位与当前能力；
2. 首版负责/不负责的边界；
3. 前置条件与安装；
4. S3 Dataset Manifest 最小示例及字段说明；
5. rerank 训练配置最小示例；
6. `validate → train → backtest` 命令；
7. 四类输出及如何读取 `metrics.json`；
8. 关键错误分类和排查入口；
9. Finder 对应目录；
10. 本地开发命令与深层文档索引。

架构细节只给出最小心智模型，正文不复制 architecture 或 Spec 的长篇内容。所有未实现能力必须
明确标注，尤其是 ONNX、线上 serving、GPU、多机训练、调度、发布和 S3/IAM 创建。

## 命令与示例约束

- README 使用中文说明和可复制的 shell/YAML/JSON 示例；
- bucket、prefix、run id 和本地路径全部使用明显占位值，不出现内部地址或凭据；
- 真实训练路径先执行 `validate`，验证成功后才展示 `train`；
- 示例不设置 AWS access key，身份由调用环境的标准凭据链提供；
- 命令统一使用 `uv run python -m recommend.train.run ...`；
- 为保证上述命令真实可执行，`run.py` 增加最小 `if __name__ == "__main__": app()` 入口，不创建
  第二套 CLI；
- 增加 CLI help 的子进程测试，防止 README 的启动命令再次失效。

## 验证

实施完成需验证：

1. README 中所有仓库内链接存在；
2. `uv run python -m recommend.train.run --help` 返回 0；
3. `make check` 返回 0；
4. README 未包含凭据、真实 bucket、ONNX 实现或未经验证的生产 SLA；
5. README 对 NFR001 的描述保持为“生产 28 日容量 benchmark 待执行”。
