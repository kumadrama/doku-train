# 验证

Source: 2026-11-26在四个隔离worktree执行的本地检查。所有AWS交互测试使用fake/stub或Terraform mock；没有启动云训练、上传制品、apply基础设施、更新线上Endpoint、切换AB或接管生产模型。

## 实现与证据

| 需求 | 验证入口 | 结果 |
| --- | --- | --- |
| 新配置仅影响新请求；冻结runtime/数据/源run | `recommend_tf/train/tests/test_model_management.py` | 旧配置保持1轮，新请求15轮；升级worker后旧请求仍使用原revision；首发COMMITTED inventory冻结 |
| 重复/并发/取消/重建/恢复 | 同上 | 同日唯一槽位、cancel与launch互斥、同模型先后请求串行、响应丢失复用token、失败重建解除旧gate、优雅暂停恢复、失败终态仍可显式重建 |
| AppConfig CAS | 同上 + 既有inference/tests | hosted version响应丢失后同计划恢复；一次deployment；过期计划拒绝 |
| 固定Endpoint服务合同 | 同上真实CPU smoke + 既有daily publication测试 | 实际训练、封存、下载、导出及身份signature通过；合同变化在Endpoint更新前拒绝 |
| 存量接管 | 同上 | 旧入口暂停后等待明确重试窗口；abort恢复旧入口并重置等待；接管后原state及Endpoint配置未变；激活后不能abort |
| 统一CLI与skill | doku-rec-deploy `tests/test_model_manage.py` | 6项通过；账号或远端环境不符时禁止变更，配置只生成plan，批量pending不推进下一模型 |
| 多配方监控与历史重放 | doku-monitoring `test_managed_model_metrics` + `test_model_metrics` | 14项通过；新旧配方并存，旧事件不回退目录；可信接管允许worker family转换；伪造组/摘要/配方被拒绝 |
| 资源与权限 | doku-iac `modules/model_management_group/tests` | Terraform validate通过、1个mock plan通过、2个资源登记测试通过；默认暂停、独立Endpoint前缀、无内嵌模型配方 |

## 命令及结果

训练仓原有统一入口已执行：`UV_NO_SYNC=1 UV_CACHE_DIR=/private/tmp/model-management-uv-cache UV_OFFLINE=1 make check`。复用本机已锁定的Python环境，避免改动依赖。

- lock检查通过；ruff lint/format通过（最终445个Python文件）；mypy通过（70个source files）。
- 原训练栈：`pytest recommend/train/tests -q -m "not performance"`，329 passed、3 skipped。
- TF完整栈：`make tf-check`，283 passed；后续恢复边界补测使用下列管理测试，最终22 passed。
- 推理完整栈：`python -m unittest discover -s recommend_tf/inference/tests -p 'test_*.py' -q`，最终37 tests、OK。
- 管理最终测试：`PYTHONPATH=. TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 <TF-python> -m pytest recommend_tf/train/tests/test_model_management.py -q --disable-warnings --tb=short`，22 passed。
- 初次make check发现Idea共享worker源摘要需要更新；用`python3 idea-mapped-daily/build/package_source.py --pin`更新本地清单后，完整TF/推理检查通过。源bundle摘要：`fcb8c56f1359781a97f4b2fdaa3d833c1dedd9062ee03476920b7abbba6b3424`。这只是待发布清单，不代表已上传/部署。
- 四份JSON Schema通过Draft202012Validator.check_schema；文档相对链接与`git diff --check`通过。

控制器使用锁定boto3/botocore 1.43.120及其依赖，在隔离目录两次调用`scripts/package_model_management.py`。两个ZIP相同，SHA-256：`03855a8832546cd4a11e21551be731aac932ac5b2e075c441c5dd46ca9bf94e5`。解压后以`python -S`仅从制品导入API，通过SDK的AppConfig LatestVersionNumber CAS字段校验，且未加载TensorFlow/NumPy/PyArrow。制品只保留本地，不随源码提交。

管理仓：`python3 -m unittest discover -s tests -v`，6 tests、OK；新增skill和更新的部署skill通过skill-creator quick_validate，CLI ruff通过。

监控仓执行`make test PYTHON=<Python3.12> PROMTOOL=/opt/homebrew/bin/promtool`：766项Python测试通过（含1个已有expected failure），35组pattern回放通过，跨语言fixture生成无diff。随后Go阶段因本机默认Go版本不符中断；切换仓库要求的Go 1.26.6后，按Makefile顺序补齐notifier/investigator/release-observer/deploy-notify全部`make check`与dsh-agent/alert `make test`（33 tests、OK）。notifier缺失的CloudWatch v1.70.0依赖因官方代理/GitHub连接失败，改用公开Go镜像下载同一锁定版本，go.sum未改且`go mod verify`通过。未改这些无关Go模块源码。collector最后增补的历史重放用例独立复跑，14项通过。

IaC：`terraform -chdir=modules/model_management_group validate`、`terraform ... test`均通过（mock AWS，1 passed）；`python3 -m unittest discover -s modules/model_management_group/tests -v`，2 tests、OK。模块未在任何环境实例化。

## 尚未执行的生产验收

- 共享控制器、worker和资源登记器发布；collector组级接线；DTE真实AWS训练/推理闭环。
- 审查实际IAM、网络、旧target及重试窗口；PROD部署计划与授权。
- FYP/Idea各一例接管及下一次真实日更观察，再批量接管。

本地测试验证代码行为，不能证明AWS配额、实际网络、角色授权或现有生产状态；这些属于上述后续部署验收。
