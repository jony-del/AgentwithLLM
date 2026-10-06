# Polaris 安装与发布

## 普通用户：不需要下载仓库

Windows PowerShell：

```powershell
irm https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.ps1 | iex
```

macOS/Linux：

```bash
curl -fsSL https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.sh | bash
```

这些入口以成功发布的 GitHub Release 为前提。程序在本机运行；默认模型服务仍通过 API
访问，安装 CLI 不等于下载或部署本地大语言模型。

安装器下载 `polaris-installer.zip` 或 `polaris-installer.tar.gz`，校验 SHA-256，再准备 uv、
Python 3.12 和完整运行依赖。小型安装包仅包含 wheel、依赖约束、发布清单及安装/卸载辅助
程序，不包含开发仓库、项目配置、测试、历史会话或 `.env`。wheel 内包含 Python 程序代码；
这种分发方式不提供源码保密。

普通安装使用 `uv tool` 独立环境。无需激活 `.venv`，可以在任意工作目录运行 `polaris`。
安装完成后重新打开终端，或根据安装器显示的命令目录配置 PATH。

```console
polaris --version
polaris health --provider fake --profile runtime
polaris run "Say hello without tools" --provider fake
```

完整安装仍包括 Git、ripgrep、Node/npm/npx、容器沙箱、记忆模型和用户级调度服务。
记忆模型约 1.17 GB，另有容器镜像和 Python 依赖。Windows 首次启用 WSL2 可能需要管理员
确认和重启：退出码 `20` 表示重启后重跑同一条命令，不是安装已完成。
安装器不会自动改写工作目录中的 `agent.toml`；运行时的沙箱启用方式见
[WSL2 沙箱指南](sandbox-wsl2.md)。

## 首次连接模型服务

例如使用 OpenAI Responses provider，Windows：

```powershell
$env:OPENAI_API_KEY = "你的 API Key"
polaris --provider openai --model "你的服务支持的模型名"
```

macOS/Linux：

```bash
export OPENAI_API_KEY="你的 API Key"
polaris --provider openai --model "你的服务支持的模型名"
```

`--model` 应使用你的服务实际支持的模型名。其他 OpenAI-compatible 服务使用
`--provider openai-compat`，通过 `OPENAI_COMPAT_API_KEY`、`OPENAI_COMPAT_BASE_URL` 配置；
Claude 使用 `--provider claude` 和 `ANTHROPIC_API_KEY`。环境变量可按你的系统习惯持久化；
不要把个人密钥提交到仓库。用户设置文件为 `~/.polaris/settings.toml`，项目可选使用
`agent.toml` 和 `agent.local.toml`，均不要求获取 Polaris 的源码。

## 升级、指定版本和可选跳过

升级 Windows：

```powershell
& ([scriptblock]::Create((irm https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.ps1))) -Upgrade
```

升级 macOS/Linux：

```bash
curl -fsSL https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.sh | bash -s -- --upgrade
```

升级保留用户配置和会话，仅替换安装器拥有的 Polaris 环境。未传升级参数时，已有命令会
复用；普通 pip/Conda 等其他安装不会被自动覆盖。wheel 和约束文件来自同一个确定版本，
不要用 `uv tool upgrade` 对 Release URL 安装进行跨版本升级。失败后可重跑安装器；
更新 CLI 后，后续模型/服务步骤失败可能留下部分完成状态，不保证整体事务回滚。

指定版本时将入口中的 `latest/download` 改为 `download/vX.Y.Z`，例如已发布的
`v0.1.1` 使用 `releases/download/v0.1.1/install.ps1`。已发布脚本内嵌其版本标签，后续
下载不再访问 `latest`，避免更新期间混用不同 Release。

可显式跳过容器准备和记忆模型下载，Python 依赖仍使用完整的 `[all]`：

```powershell
& ([scriptblock]::Create((irm https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.ps1))) -SkipSandbox -SkipMemoryModels
```

```bash
curl -fsSL https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.sh | bash -s -- --skip-sandbox --skip-memory-models
```

跳过记忆模型后使用词法检索降级；需要完整模型能力时重跑安装器，不传跳过选项。
跳过沙箱的安装不能提供沙箱隔离能力。已有离线记忆模型包可通过 `-ModelBundle PATH` /
`--model-bundle PATH` 指定，不能同时传跳过记忆模型选项。

## 卸载与损坏后的恢复

```console
polaris uninstall --dry-run
polaris uninstall
```

普通卸载依据精确的安装所有权收据删除 CLI、专用环境和私有 runtime，保留用户配置/会话
以及共享的 Git、uv、Python、容器工具。清理用户数据必须显式使用
`polaris uninstall --purge-data --yes`。

CLI 损坏时可以使用远程恢复入口，不需要下载源码：

```powershell
& ([scriptblock]::Create((irm https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.ps1))) -Uninstall -DryRun
```

```bash
curl -fsSL https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.sh | bash -s -- --uninstall --dry-run
```

移除 `DryRun` / `--dry-run` 后执行实际卸载。恢复卸载不会安装缺失的 uv/Python；需要可用的
现有 uv-managed Python。检查 PATH 上是否还有 Conda/pip 的其他 Polaris 安装，避免把这些
独立环境误认为同一次安装。

## 源码开发

仅开发者需要持久的 checkout：

```powershell
.\install.ps1 -Dev
```

```bash
bash install.sh --dev
```

`--dev` 安装 editable 软件和开发依赖；普通用户的 wheel 安装不会引用开发目录。
直接从仓库执行安装脚本仍兼容本地源码安装。未经过发布构建的脚本没有内嵌版本标签，不能
直接从 raw/main 管道安装；使用正式 Release 入口。

## 维护者发布

1. 修改 `pyproject.toml` 的版本，更新并提交 `uv.lock`，检查内置资源和发布测试。
2. 运行 `python tools/build_release_assets.py --tag vX.Y.Z`。构建工具调用 uv 构建 wheel，并
   从 `uv.lock` 导出 `[all]` 的跨平台依赖约束；标签必须与项目版本一致。
3. 检查 `dist/` 中的 wheel、两个安装包、两个安装脚本、`release.json`、约束文件和
   `SHA256SUMS`。构建器校验入口和必要资源，使用明确的辅助文件白名单。
4. 推送版本标签触发 `.github/workflows/release.yml`。流水线检查两个沙箱锁文件一致，验证
   已锁定的双架构镜像可匿名下载，执行安装器回归、Linux OCI E2E 和
   Windows/macOS/Linux 的 wheel 安装测试；三平台安装测试没有源码 checkout。
5. 检查全部门禁通过。流水线先上传完整资产到草稿 Release，再公开，随后一行入口可用。

可复用本地已经构建的 wheel 和约束文件检查打包，无需联网：

```console
python tools/build_release_assets.py --wheel PATH.whl --constraints PATH.txt --tag vX.Y.Z
```

发布仍需要 GitHub 仓库和 GHCR 的相应权限。GHCR 镜像必须允许目标用户拉取；Release
下载也必须允许目标用户访问。校验和用于检测文件损坏，分发信任来自受信任的 HTTPS 仓库
和发布权限；当前流程没有额外的离线签名验证。

沙箱镜像通过 `.github/workflows/sandbox-image.yml` 单独构建和执行 Linux OCI 验证。
修改沙箱协议、镜像内运行代码、工具链或依赖时，先运行该工作流，提供新的镜像标签；验证
通过后，将输出的不可变 digest 同时写入 `agent_core/sandbox/sandbox-image.lock.json`
和 `installer/manifest.json`，提交后再发布 CLI。普通 CLI 发布复用已锁定的镜像。

Windows WSL2/Podman 的完整资格验证保留在
`.github/workflows/sandbox-windows.yml`，通过手动运行、提供不可变镜像 digest 来执行，需要
带 `polaris-wsl2` 标签的现成测试机。CLI 发布不等待该专用机器；三平台 CLI 验证不等同于
新镜像在 Windows WSL2 上已完成完整验证，宣称该组合经过验证前应完成此工作流。

首次新电脑验收还应验证：不预装 Python、无源码目录、路径含中文/空格、网络失败重试、
WSL2 重启后续装、升级保留配置，以及卸载仅删除拥有的安装。远程入口回归使用隔离的假
下载器和惰性 worker，避免自动测试替换开发电脑上的工具、容器或调度服务。
