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

安装完成后运行交互式向导完成首次配置：

```console
polaris setup
```

向导依次选择 provider（claude / openai / openai-compat）和模型，openai-compat 还会询问
Base URL；随后以隐藏输入接收 API Key。密钥只写入操作系统凭据管理器（macOS Keychain、
Windows 凭据管理器、Linux Secret Service，经 `keyring` 包；默认安装已包含 `credentials`
extra），不会写入任何文件；provider、模型等非密钥配置写入 `~/.polaris/settings.toml`。
向导结束前可做一次最小真实请求验证连通性。直接运行 `polaris` 而凭据缺失时，交互终端
也会现场提供该向导。

非交互环境（管道、CI）不提问，直接用标志写入配置：

```console
polaris setup --provider claude --model claude-opus-4-8 --api-key-env ANTHROPIC_API_KEY
polaris setup --provider openai-compat --model <模型名> --base-url http://localhost:8000 --api-key-env OPENAI_COMPAT_API_KEY
```

`--api-key-env VAR` 声明密钥已在环境变量中；`--verify` 立即发一次请求验证；
`--non-interactive` 明确不提问（stdin 不是终端时自动生效）。非交互调用未提供
`--provider` 时只打印配置指引并以退出码 2 结束，不写入任何内容。

`~/.polaris/settings.toml` 支持 `provider`、`model`、`base_url` 键。`base_url` 主要面向
openai-compat（vLLM、LM Studio、DeepSeek 等自托管或兼容端点）；同名环境变量
（`OPENAI_COMPAT_BASE_URL`、`OPENAI_BASE_URL` 等）优先级更高。可用
`POLARIS_SETTINGS_PATH` 指向其他设置文件。

### 手动环境变量方式

密钥解析顺序始终是环境变量优先、操作系统凭据管理器其次（`POLARIS_DISABLE_KEYRING=1`
可完全停用凭据管理器）。不使用向导时按系统习惯持久化环境变量即可，Windows：

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

已由安装器安装的 CLI 可以直接在命令行内升级或回滚：

```console
polaris upgrade --check
polaris upgrade
polaris upgrade --to v0.1.1
polaris upgrade --force
```

`polaris upgrade` 校验安装器所有权收据，下载目标 Release 并做 SHA-256 校验，随后由独立
后台进程在本命令退出后完成 `uv tool` 环境替换；日志写入 `~/.polaris/upgrade-last.log`，
完成后打开新终端运行 `polaris --version` 确认。`--check` 只对比当前与最新版本；`--to`
指定更旧的版本即回滚；`--force` 强制重装当前版本。非安装器安装（源码、pip/Conda）会被
拒绝并给出对应升级指引。升级保持安装时的依赖宽度：收窄默认之前安装的完整环境升级后
仍为 `[all]`，收窄后的默认安装升级后保持默认集合（加回 SWE-bench 依赖见下文的
`-Full` / `--full`）。后台升级失败时先查看 `~/.polaris/upgrade-last.log`（升级前的健康
快照存于 `~/.polaris/upgrade-last-health.json`），再重跑下方的安装脚本即可恢复。
以下脚本方式仍然可用。

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

回滚即"用旧版本执行一次升级"：旧 Release 资产保留在 GitHub 上，将入口指向旧版本并加
升级参数即可，例如 `-Version v0.1.0 -Upgrade`（或 `bash -s -- --version v0.1.0 --upgrade`
对应的下载地址）。回滚只替换安装器拥有的 CLI 环境；`~/.polaris` 下的用户配置、凭据、
会话、索引和记忆模型与程序版本分离，升级和回滚都不会触碰。升级或回滚失败后可重跑同一
命令，安装器是幂等的。

默认安装只包含日常运行所需的 Python 依赖（`web`、`mcp`、`terminal`、`memory`、`codeintel`、
`credentials`），不再包含仅用于基准测试的 SWE-bench 依赖（`datasets`、`docker`、`swebench`
等，体积较大）。缺少这些依赖时，`polaris swebench` 会提示用 `--full` 重装；`polaris health`
不检查基准依赖，默认安装的健康结论不受影响。需要官方 SWE-bench Harness 时，用 `-Full` /
`--full` 重跑安装器即可装回完整的 `[all]` 集合：

```powershell
& ([scriptblock]::Create((irm https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.ps1))) -Full
```

```bash
curl -fsSL https://github.com/jony-del/AgentwithLLM/releases/latest/download/install.sh | bash -s -- --full
```

可显式跳过容器准备和记忆模型下载：

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
   本地 `dist/` 是构建输出目录（全部被 git 忽略），其内容可能缺失或过期；权威产物始终以
   发布流水线在标签上的全新构建为准，不要以本地 `dist/` 的新旧判断发布状态。
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

上述验收场景中，以下已由自动化锁定：不预装 Python 且无源码目录的三平台 wheel 安装由
`.github/workflows/release.yml` 的 cli-install 车道覆盖（无 checkout，且只使用 uv 管理的
Python 3.12，拒绝回退到系统解释器）；路径含中文/空格、重复安装幂等且不重复写 PATH、升级
保留 `~/.polaris` 用户数据、网络下载失败不产生部分变更且重跑可完成，由
`tests/test_installer_acceptance.py` 覆盖；WSL2 重启后续装与卸载仅删除拥有的安装分别由
`tests/test_installer.py` 与 `tests/test_uninstall.py` 覆盖。首次新电脑验收仍应在真实
机器上组合验证这些场景。远程入口回归使用隔离的假下载器和惰性 worker，避免自动测试替换
开发电脑上的工具、容器或调度服务。
