# 安装与升级

## 默认安装内容

完整运行档包含：uv 管理的 Python 3.12、项目的 `[all]` Python extras、Git、ripgrep、
Node.js 24 LTS、npm/npx、一个可用的 Podman/Docker/nerdctl 运行时，以及
`docker.io/library/debian:stable-slim` 沙箱镜像。安装器还会注册最小权限的当前用户 Scheduler
服务；不支持或不可用的用户服务管理器会使安装明确失败，不会留下孤儿后台进程。

Node 使用 nodejs.org 官方二进制和 `SHASUMS256.txt` 校验，安装在用户目录；不会覆盖系统
Python。Git、rg 和 Podman 使用 WinGet、Homebrew/官方 macOS 包、apt 或 dnf 安装。脚本只在
需要系统变更时请求 UAC/sudo，不会读取或写入 API 密钥。

正式支持范围：

- Windows 10 build 19043+ / Windows 11 x64，使用 WSL2 Podman Machine；
- macOS Intel 与 Apple Silicon；
- Ubuntu 22.04+、Debian 12+、Fedora/RHEL 9+ 的 x86_64/arm64。

Windows 需要可用的 WinGet（Microsoft App Installer）。企业代理环境可使用标准的
`HTTPS_PROXY`、证书和 uv 索引环境变量。
Windows 的 agent 命令还强制要求可信的 Git for Windows Bash；WindowsApps `bash.exe` 和 WSL
不会作为回退。非标准路径可通过 `POLARIS_BASH_PATH` 指定。

## 可审阅与固定版本安装

不希望直接执行远程脚本时，可先下载同一 GitHub Release 的 `install.ps1`/`install.sh`、
`polaris-source.*` 和 `SHA256SUMS`，检查脚本和摘要后再运行。远程脚本同样会在解压源码前验证
release 中的 SHA-256。

固定版本：

```powershell
.\install.ps1 -Version v0.1.0
```

```bash
bash install.sh --version v0.1.0
```

## 参数

| PowerShell | shell | 行为 |
| --- | --- | --- |
| `-Dev` | `--dev` | 仅 checkout 内：创建 `.venv` 并安装 `.[all,dev]` |
| `-Upgrade` | `--upgrade` | 升级 Polaris 和安装器记录为自己安装的组件 |
| `-Check` | `--check` | 只检测，不修改主机 |
| `-DryRun` | `--dry-run` | 打印计划执行的命令 |
| `-SkipSandbox` | `--skip-sandbox` | 明确接受不安装容器运行时 |
| `-NonInteractive` | `--non-interactive` | 不弹出交互；缺权限或前置条件时失败 |
| `-Uninstall` | `--uninstall` | 从安装脚本进入恢复卸载流程 |
| `-PurgeData` | `--purge-data` | 卸载后额外删除用户级 Polaris 数据和安装状态 |
| `-Yes` | `--yes` | 无提示确认已经显示的卸载计划 |

重复运行默认只补缺失项，不强制升级已有工具。安装状态位于
`%LOCALAPPDATA%\Polaris\install-state.json`（Windows）或
`${XDG_STATE_HOME:-~/.local/state}/polaris/install-state.json`，其中只记录步骤、来源和所有权。

新安装收据还记录安装类型、CLI/环境根目录、uv 路径和外部 worker Python。记录中不含 API 密钥或
其他凭据。旧 schema 1 收据不会被直接信任；只有重新探测能精确匹配 uv tool 或源码 `.venv` 时
才允许卸载，否则按外部安装处理。

Windows 上的成功条件是安装后的命令确实可执行，而不是 WinGet 单独返回成功。若 WinGet 已有
Git、rg 或 Podman 的记录，但当前终端尚未获得新 PATH，安装器会合并刷新 PATH 后复检；若记录
存在但可执行文件或 portable 命令链接已经损坏，则自动强制重装一次。恢复后仍不可用时，错误
信息会给出对应的 `winget uninstall --id ... --exact` 清理命令，不会把未验证的组件写入安装状态。

默认沙箱档可能需要 UAC 来安装 Podman 或启用 WSL2；Windows 首次启用 WSL2 后需要重启，并以
退出码 `20` 提示重新运行相同命令。只需 Python/Node/主机命令而明确不需要容器沙箱时，可使用
`-SkipSandbox`；这属于显式选择的降级安装。若用户取消 UAC，安装器会把它明确映射为 Windows
错误 `1223 (0x000004C7)` 并立即停止，不会继续尝试其他 WSL 系统修改。

安装器会先探测 WSL 状态：尚未安装时先执行 `wsl --install --no-distribution`；命令失败且重新
探测后 WSL 仍不可用时，再自动尝试一次 `wsl --install --no-distribution --inbox`，使用 Windows
组件而非 Store 路径。已有 WSL 但状态异常时才执行 `wsl --update`。这些参数适用范围见
[Microsoft WSL 命令说明](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)。安装器不会
自动运行 DISM 或系统映像修复。

提权子进程的 stdout/stderr 会分别写入仅本次调用使用的临时文件，读取后始终清理。WSL 在本地化
Windows 上可能输出无 BOM 的 UTF-16 文本；安装器会按实际字节编码解码，不依赖当前 Conda/Python
环境的默认代码页。安装器把 `0`、`1641` 和 `3010` 视为成功或需要重启，但每次都会重新运行
`wsl --status`：即使其他退出码也会以重新探测结果为准。操作已被 Windows 接受但 WSL 尚不可用时，
安装器写入可恢复步骤并以退出码 `20` 提示重启；两种安装路径都失败时，错误会同时列出两条命令、
十进制/十六进制退出码以及解码后的 stdout/stderr。诊断内容不会写入安装状态。

退出码：`0` 就绪、`2` 参数错误、`10` 安装/卸载/验证失败、`20` 需要重启后重跑、`30` 不支持的
平台或架构。

## 卸载与数据保留

首选由 CLI 展示并确认卸载计划：

```console
polaris uninstall
polaris uninstall --dry-run
polaris uninstall --yes
polaris uninstall --purge-data --yes
```

`polaris uninstall` 在当前进程退出前把计划复制到权限受限的唯一临时目录，再由目标环境之外的
uv Python 完成删除。Windows 上不会尝试删除仍被当前进程占用的 `polaris.exe`；成功调度后返回
`0` 并打印完成日志路径。日志中的 `[uninstalled]` 表示最终完成，`[error]` 会保留可恢复诊断。
Scheduler 服务仅在安装状态和独立服务收据同时精确匹配时删除；默认保留调度历史，只有
`--purge-data` 才删除 Scheduler SQLite 数据库。

若 CLI 命令已损坏，下载/保留与目标版本对应的 release 脚本后使用同步恢复入口：

```powershell
.\install.ps1 -Uninstall
.\install.ps1 -Uninstall -DryRun
.\install.ps1 -Uninstall -PurgeData -Yes
```

```bash
bash install.sh --uninstall
bash install.sh --uninstall --dry-run
bash install.sh --uninstall --purge-data --yes
```

卸载模式只查找已经存在的 uv 和 uv Python 3.12，不安装或下载它们，也不会进入 WSL/Podman 准备
流程。`-Uninstall`/`--uninstall` 不能与安装、升级、检查或沙箱选项组合；非交互卸载必须提供
`-Yes`/`--yes`，但 dry-run 不需要确认。

默认删除内容：

- 安装器拥有的 `agent-with-llm` uv tool 环境、`polaris` 启动器及环境内专用 Python 依赖；
- 安装器创建且具有匹配所有权标记的开发 `.venv`，但保留源码 checkout；
- 安装在 Polaris 私有 runtime 根目录中、路径收据完全匹配的 Node 和对应 PATH/符号链接；
- Polaris 程序和私有 runtime 的状态收据。

默认保留 `~/.polaris` 配置、信任信息、技能、计划和会话，以及项目内 `.polaris`、`agent.toml`、
`.env`、`runs`、`memory`。同时始终保留 WSL、Podman/Docker/nerdctl、容器镜像、Git、ripgrep、
系统 Node、uv、uv Python 和包管理器。`--purge-data` 仅额外删除用户级 `~/.polaris` 与安装状态，
仍不扫描或删除任何项目目录。

Conda、普通 pip、手工 editable 或仅被安装器复用的 Polaris 没有安装器所有权，自动卸载会以
退出码 `10` 安全拒绝，并打印绑定当前解释器的命令，例如：

```console
<当前环境的 python> -m pip uninstall agent-with-llm
```

路径越界、符号链接逃逸、收据被修改或活动 `polaris` 与收据不一致时同样拒绝，不提供绕过所有权
检查的强制参数。重复运行恢复卸载且目标已经不存在时返回 `0`。

## 健康检查

```console
polaris health --provider fake --profile runtime
polaris health --provider fake --profile runtime --json
polaris health --provider fake --profile dev
```

JSON 输出固定包含 `status`、`profile` 和 `checks`；每个检查项包含 `name`、`required`、
`status`、`version`、`detail`。运行档要求全部 Python extras、Git、rg、Node/npm/npx、容器引擎
和默认镜像均可用；开发档再要求 pytest、ruff 和 mypy。

社区 MCP npm 包不会被预下载；启用对应配置后，`npx` 会按服务器声明获取它们。Kata、Lima、
专用 Hyper-V VM 等替代后端也不在默认安装档内。
