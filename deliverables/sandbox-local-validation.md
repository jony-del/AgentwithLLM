# 本地 sandbox 构建与 Polaris 验收记录

验收完成：2026-09-15，Windows / WSL2 Podman，`linux/amd64`。

最终镜像已激活到 `agent.local.toml`：

```text
sha256:a14e3cae36e5e5370e88b402b8c42a113a7c52a684fe7dfe99227f48a5e263e3
```

`runtime="podman"`、`auto_pull=false`、`auto_start_machine=true`，机器名仍为
`podman-machine-default`。最终健康状态为 `ok`，所有检查通过；机器保持运行。

## 修改

- 修正 Python 3.12 / Bookworm 基础镜像摘要；从原有根目录 `uv.lock` 导出完整 `all,dev` 约束并保留平台条件，主环境使用 MCP 2.1.1。根锁文件没有升级。
- 保留 Node v24.14.1、PowerShell 7.5.7、Pyright 1.1.405 和 Debian 快照 20260809T000000Z。修正 PowerShell 官方 UTF-16、星号文件名前缀校验清单的读取；Node、PowerShell 下载均通过 SHA-256 校验。
- Git/Fetch/Time 2026.7.10 参考服务器仍使用 SDK 1.x 服务端接口，故在独立的 `/opt/polaris/mcp` 环境运行，使用单独锁定的 MCP 1.28.1；Polaris 主环境继续使用 2.1.1。两套环境均通过 `pip check`。
- 增加 `tools/build_sandbox.py`：导出锁定依赖、显式构建 `linux/amd64` / `TARGETARCH=amd64`、通过 `--iidfile` 读取真实 ID、运行准备探针和 OCI 集成测试，全部成功后原子更新本地配置。失败保留原配置，激活前检查并发编辑。
- 完整本地 ID 仅允许显式 Podman 且禁止自动拉取；检查实际完整 ID，每次探针及工具运行使用 `--pull=never`，缺失时要求重建。发行锁继续只接受远端仓库摘要。
- 健康检查使用实际解析后的项目/本地/环境/CLI sandbox 设置；在副本中关闭自动启动与自动拉取，保持 JSON 报告格式。
- 补充 OCI stdin 和 Windows Podman 客户端 APPDATA/连接环境，修复 LSP、MCP、Node 工作流协议与连接问题。容器网络隔离保持启用。
- 修复缺失 Python 钩子脚本的退出码 2 被误当作明确拒绝：启动失败遵循原 `fail_mode`。当前示例脚本不存在，按既有 `open` 模式记录失败；`closed` 模式和真实钩子的明确拒绝仍阻断，内置提示词校验保持启用。
- 保留构建上下文白名单，进一步明确排除嵌套虚拟环境和 `.env` 文件；保留此前 Podman 按需启动修改。

## 验证

| 项目 | 结果 |
| --- | --- |
| 最终镜像内主环境、MCP 服务环境 `pip check` | 通过 |
| 最终 OCI 集成测试 | 7 passed；停机项单独执行 |
| Bash、PowerShell、Python/pytest、Node | 通过 |
| Git/Fetch/Time MCP 握手、真实时间工具调用 | 通过 |
| Pyright 定义查询及宿主/来宾 URI 转换 | 通过 |
| 中文路径、E 盘挂载、工作区读写 | 通过 |
| 非 root、只读根文件系统、capabilities 清零、禁止提权、网络隔离 | 通过 |
| fake provider 驱动真实 sandbox Bash 工具 | 通过 |
| Polaris CLI 启动及中文离线 echo 任务 | 通过 |
| `polaris health --profile dev`，唤醒前后 | 全部 `ok` |
| 停机后健康检查 | 正确报错，未自动唤醒 |
| 停机后禁止宿主机回退 | 1 passed |
| 无其他运行容器时停止、由 Polaris 自动唤醒 | 通过；结束保持运行 |
| 相关源码回归 | 分批 226 passed、170 passed / 1 skipped、76 passed；测试集合有重叠 |
| Ruff | 通过 |
| Mypy | 174 个源码文件无错误 |
| `git diff --check` | 通过 |

本轮只验收本机 `linux/amd64`，所有代理运行使用 fake provider，没有调用付费模型，未发布 GHCR。
根目录 `uv.lock`、发行版镜像锁和安装器清单未改动。

## 复用

在项目根目录、已激活开发虚拟环境时运行：

```powershell
python tools/build_sandbox.py
polaris run "tool: echo 本地离线验收" --provider fake --sandbox --no-memory
polaris health --provider fake --profile dev --no-memory --json
```

只构建与验收、不激活：`python tools/build_sandbox.py --no-activate`。

## 原始记录与备份

- 最终构建记录：`tmp/sandbox-build.log`。
- 最终构建 ID、结果与激活前备份：`tmp/sandbox-build-pc88ppz9/`。
- 本轮开始时的本地配置备份：`tmp/sandbox-build-8e8brhiz/agent.local.toml.before`。
- 最终 CLI、健康检查、停机和唤醒的逐项 stdout/stderr：`tmp/sandbox-acceptance/`。
- 最终验收摘要：`tmp/sandbox-acceptance/summary.json`。
- 最终开发配置健康报告：[sandbox-local-health.json](sandbox-local-health.json)。
- 使用说明：[sandbox-wsl2.md](../docs/sandbox-wsl2.md)。

恢复配置时可将所需的 `agent.local.toml.before` 复制回项目根目录的 `agent.local.toml`；未删除原有镜像。
