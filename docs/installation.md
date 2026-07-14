# 安装与升级

## 默认安装内容

完整运行档包含：uv 管理的 Python 3.12、项目的 `[all]` Python extras、Git、ripgrep、
Node.js 24 LTS、npm/npx、一个可用的 Podman/Docker/nerdctl 运行时，以及
`docker.io/library/debian:stable-slim` 沙箱镜像。

Node 使用 nodejs.org 官方二进制和 `SHASUMS256.txt` 校验，安装在用户目录；不会覆盖系统
Python。Git、rg 和 Podman 使用 WinGet、Homebrew/官方 macOS 包、apt 或 dnf 安装。脚本只在
需要系统变更时请求 UAC/sudo，不会读取或写入 API 密钥。

正式支持范围：

- Windows 10 build 19043+ / Windows 11 x64，使用 WSL2 Podman Machine；
- macOS Intel 与 Apple Silicon；
- Ubuntu 22.04+、Debian 12+、Fedora/RHEL 9+ 的 x86_64/arm64。

Windows 需要可用的 WinGet（Microsoft App Installer）。企业代理环境可使用标准的
`HTTPS_PROXY`、证书和 uv 索引环境变量。

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

重复运行默认只补缺失项，不强制升级已有工具。安装状态位于
`%LOCALAPPDATA%\Polaris\install-state.json`（Windows）或
`${XDG_STATE_HOME:-~/.local/state}/polaris/install-state.json`，其中只记录步骤、来源和所有权。

退出码：`0` 就绪、`2` 参数错误、`10` 安装/验证失败、`20` 需要重启后重跑、`30` 不支持的
平台或架构。

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
