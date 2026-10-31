# TF ready 发布单元

使用根目录 scripts/package_tf_ready.py 确定性打包。包保留 ready_event.handler，兼容现有 Lambda handler。
由 dkd lambda 目标发布；物理函数/桶/前缀只来自 IaC binding。
上传不可变 ZIP 后，用真实 key/object_version 创建 deploy/prod/lambda.yml（格式与 monitoring notifier 相同）；不提交占位制品坐标。
首次接管前先部署发布平台与受信 binding，再执行计划。仅代码发布，不修改环境或上游规则。

打包参数必须指定同一 release 的 `--release release.json --config daily-config.json --output ready.zip`。配方原始字节与摘要进入 ZIP；发布后的窗口不再读取 IaC 的旧训练天数。先完成兼容 pipeline 与启动参数切换，再发布本单元。
