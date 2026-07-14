# Polaris / Agent with LLM

Polaris 是一个带工具、权限、MCP、记忆和沙箱能力的 Python ReAct Agent。普通用户不需要先安装
Python：项目安装器会准备隔离的 Python 环境，并补齐 Git、ripgrep、Node/npm/npx 和容器沙箱。

## 一键安装

Windows 10/11（PowerShell）：

```powershell
irm https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.ps1 | iex
```

macOS、Ubuntu/Debian、Fedora/RHEL：

```bash
curl -fsSL https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.sh | bash
```

安装器会复用已可用的 Podman、Docker 或 nerdctl；三者都不可用时安装 Podman，并预拉取默认
沙箱镜像。Windows 首次启用 WSL2 后可能返回退出码 `20` 并要求重启；重启后重新运行同一条命令
即可从已完成步骤继续。

安装完成后验证：

```console
polaris health --provider fake --profile runtime
polaris run "Say hello without tools" --provider fake
```

安装器只准备沙箱能力，不会擅自修改项目的 `agent.toml`。需要使用沙箱时传入 `--sandbox`，或在
配置中设置 `[sandbox] enabled = true`。

## 源码开发

先下载/克隆本仓库，然后在仓库根目录执行：

```powershell
.\install.ps1 -Dev
```

```bash
bash install.sh --dev
```

开发档在 `.venv` 中执行 editable install，并包含 `pytest`、`ruff` 和 `mypy`。普通安装不会安装
这些开发工具。更多选项、平台范围、固定版本安装和安全校验方式见
[安装指南](docs/installation.md)。

## 仅 Python 安装（高级用法）

以下命令仍受支持，但**只安装 Python 包**，不会安装 Git、rg、Node 或沙箱运行时：

```console
pip install -e .
pip install -e ".[all]"
pip install -e ".[all,dev]"
```
