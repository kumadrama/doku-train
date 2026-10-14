# 模型分发 Redis

本目录是 dkd store 身份声明。实例/TLS/noeviction/网络归 doku-iac；账号及 Secret 由
`dkd deploy setup db/doku-redis-model --env prod` 创建。不部署 DTE。

PROD 发布和清理任务运行时读取 `doku-prod/doku-redis-model/readwrite`，不把密码放入
Terraform、任务环境或模型配置。默认 readwrite 档位不包含 EVAL；setup 后必须执行 iac 中
`configure-tree-redis-access.py` 的计划及应用，将账号限定在模型键前缀并授予 GET/SET/DEL/EVAL。
readonly 档位供 mono 后续加载。setup --reconcile-existing 会重置 ACL，执行后须重新收敛。
