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

普通安装由 uv tool 提供用户级 `polaris` 命令，不需要激活虚拟环境；安装完成后重新打开终端或直接
运行 `polaris` 即可。只有源码开发流程需要激活仓库中的 `.venv`。

安装器会复用已可用的 Podman、Docker 或 nerdctl；三者都不可用时安装 Podman，并预拉取默认
沙箱镜像。Windows 首次启用 WSL2 后可能返回退出码 `20` 并要求重启；重启后重新运行同一条命令
即可从已完成步骤继续。

WSL 尚未安装时，安装器先使用标准 `wsl --install --no-distribution`；若该路径失败且 WSL 仍
不可用，会自动尝试一次 Windows inbox 组件路径。取消 UAC 会明确报告 Windows 错误 `1223` 并
立即停止；两条安装路径都失败时会显示各自的十进制/十六进制退出码和真实 stdout/stderr，不再
只显示笼统的 `command failed (1)`。

在 Windows 上，安装器会验证 Git、ripgrep 和 Podman 的命令是否真正可用。若 WinGet 留有安装
记录但命令链接缺失，安装器会自动修复一次，而不会因“已安装、无可用升级”提前终止。

安装完成后验证：

```console
polaris health --provider fake --profile runtime
polaris run "Say hello without tools" --provider fake
```

安装器只准备沙箱能力，不会擅自修改项目的 `agent.toml`。需要使用沙箱时传入 `--sandbox`，或在
配置中设置 `[sandbox] enabled = true`。

## 卸载

正常卸载会删除 Polaris CLI、`uv tool` 的独立 Python 环境及安装器创建的私有 runtime；不会删除
用户配置/会话，也不会卸载 WSL、Podman、容器镜像、Git、ripgrep、uv 或共享 Python：

```powershell
polaris uninstall
polaris uninstall --dry-run
```

需要同时清除 `~/.polaris` 用户数据和安装状态时，必须显式确认：

```powershell
polaris uninstall --purge-data --yes
```

CLI 已损坏时，可在源码/release 脚本旁使用恢复入口：

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

旧版 uv tool 中还没有 `polaris uninstall` 时，直接交给 uv 删除启动器、独立环境和其中的全部专用
依赖：

```powershell
uv tool uninstall agent-with-llm
```

安装器只自动删除有精确所有权收据的 uv tool 或开发 `.venv`。Conda、普通 pip 和手工 editable
安装不会被猜测性删除；必须使用拥有该安装的解释器，例如：

```powershell
python -c "import sys; print(sys.executable)"
& "$env:CONDA_PREFIX\python.exe" -m pip uninstall agent-with-llm
```

PowerShell 中可用以下命令检查 PATH 上是否还有其他安装：

```powershell
where.exe polaris
Get-Command polaris -All -ErrorAction SilentlyContinue
```

每个 Conda 环境、仓库 `.venv` 和 uv tool 都是独立安装；从一个环境卸载不会删除其他环境中的
Polaris。完整安全边界见[安装与卸载指南](docs/installation.md)。

## 权限模式

交互式 `polaris chat` 中运行 `/permissions` 可从六种策略中选择，或直接运行
`/permissions <mode>`。Shift+Tab 在 `default → acceptedits → plan → auto` 之间循环，底部状态栏
始终显示当前模式。输入框在 Agent 流式输出和工具执行期间保持可用，因此 Shift+Tab 的修改会从
下一次权限判断/模型请求开始生效；Windows 终端还可使用 BackTab（`ESC[Z`）或 Alt+M。

- `default`：读取自动允许，编辑和外部动作需要确认。
- `acceptedits`：额外自动允许框架原生的工作区文件编辑工具。
- `plan`：严格只读，只调查、提问和制定计划。
- `auto`：普通读写走安全快速路径，其他动作由 AI 分类器允许或拒绝。
- `dontask`：任何本应询问的动作直接拒绝。
- `bypass`：允许未命中的动作，但 deny/ask 规则和敏感路径保护仍然有效。

`auto`、`dontask`、`bypass` 没有真实沙箱时需要交互式明确确认；无头运行仍默认拒绝。AI 分类器
超时、报错或返回无法解析的结果时，顶层交互会话回退人工确认，无头和子 Agent 拒绝。中央安全策略、工具级
`check_permissions()`、确定性决策顺序、plan artifact 与审计契约见
[权限系统架构](docs/permission-system.md)。

模式参数同时兼容 `acceptEdits`、`dontAsk`、`bypassPermissions`，日志仍只输出规范名。交互弹窗可把
精确规则授权到 session、`agent.local.toml`、项目 `agent.toml` 或用户 `~/.polaris/agent.toml`；持久化
授权需要二次确认。系统管理员可通过平台默认路径或 `POLARIS_MANAGED_POLICY_PATH` 部署只读的
`[managed.permissions]` 策略。`auto` 分类器故障只在顶层交互会话回退人工确认，无头和子 Agent 拒绝。

## 流式交互与队列

Agent 运行时仍可继续输入：Enter 会把消息放入无限内存队列；同优先级按 FIFO 处理。普通消息会在
完整工具结果批次之后安全注入当前轮，slash command 则留到轮次边界逐条派发。空输入框按 ↑ 可一次
取回所有可编辑队列项。

- `Esc`：协作式中止当前 Agent run。
- `Ctrl+B`：把当前前台 Bash/PowerShell 任务转入后台。
- `Ctrl+O`：查看最近 transcript。
- `Ctrl+T`：查看 todos 和输入队列。
- `Ctrl+R`：搜索输入历史；`Ctrl+L`：重绘终端。

常用会话命令包括 `/rename`、`/effort`、`/fast`、`/sandbox`、`/model` 和 `/status`。模型、
effort 与 fast mode 只在当前会话生效；sandbox 变更原子应用并写入 gitignored
`agent.local.toml`。

## Claude 兼容插件

`/plugin` 支持 install/manage/uninstall/enable/disable/validate，以及 marketplace 的
add/remove/update/list。Polaris 不预装 marketplace；安装记录和缓存位于 `~/.polaris/plugins`，
项目启用状态默认写入 `agent.local.toml`。支持 `.claude-plugin/plugin.json`、skills/commands、
agents、hooks（含 `${CLAUDE_PLUGIN_ROOT}`）和 `.mcp.json`。可执行 hooks/MCP 启用前需要确认。

安装或启用不会修改正在运行的组件代；在 Agent 空闲时运行 `/reload-plugins` 才会构建并原子切换，
失败时继续使用旧代。插件组件使用 `plugin:component` 命名空间，安装 ID 使用
`plugin@marketplace`。

## 源码开发

开发者应使用仓库根目录下的 `.venv` editable 安装，而不是在多个 Conda 环境中分别安装。普通
Python 源码修改会直接生效，只需退出并重新启动正在运行的 Polaris 进程。

### Windows PowerShell

如果提示符包含 `(base)` 或其他 Conda 环境，先执行 `conda deactivate`；必要时重复执行，直到
提示符不再显示 Conda 环境。然后在仓库根目录安装并激活开发环境：

```powershell
cd C:\path\to\AgentwithLLM
.\install.ps1 -Dev
.\.venv\Scripts\Activate.ps1
```

不需要容器沙箱时可以改用 `.\install.ps1 -Dev -SkipSandbox`。激活成功后提示符应只显示
`(.venv)`；如果显示 `(base) (.venv)`，先运行 `deactivate` 退出 `.venv`，再运行
`conda deactivate`，最后重新激活 `.venv`。可选地关闭 Conda base 的自动激活：

```powershell
conda config --set auto_activate_base false
```

验证当前命令确实来自仓库 `.venv`：

```powershell
(Get-Command python).Source
(Get-Command polaris).Source
python -c "import sys; print(sys.executable)"
```

以上路径都应位于 `...\AgentwithLLM\.venv\Scripts\`。

### macOS/Linux

```bash
cd /path/to/AgentwithLLM
bash install.sh --dev
source .venv/bin/activate
```

### 日常修改与验证

激活 `.venv` 后直接修改仓库代码并重新启动命令，无需重新安装：

```powershell
polaris --help
polaris health --provider fake --profile dev
polaris run "Say hello without tools" --provider fake
python -m pytest -q
```

Python 进程不会热加载已经导入的模块，因此每次修改后要退出旧的 `polaris` 进程再运行。只有修改
`pyproject.toml`、依赖、extras、CLI entry point 或包元数据时，才需要刷新 editable 安装：

```powershell
uv pip install --python .\.venv\Scripts\python.exe -e ".[all,dev]"
```

开发档包含 `pytest`、`ruff` 和 `mypy`。结束当前开发 shell 时运行 `deactivate`；需要删除整个开发
安装时，先退出 `.venv`，再在仓库根目录执行 `.\install.ps1 -Uninstall -Yes`。更多选项、平台
范围、固定版本安装和安全校验方式见[安装指南](docs/installation.md)。

## 仅 Python 安装（高级用法）

以下命令仍受支持，但**只安装 Python 包**，不会安装 Git、rg、Node 或沙箱运行时：

```console
pip install -e .
pip install -e ".[all]"
pip install -e ".[all,dev]"
```
工业级 tool lifecycle、Bash/PowerShell 后台任务、LSP、Notebook、Git Worktree 与 Scheduler
的配置和安全语义见 [Tool platform](docs/tool-platform.md)。从旧 shell 规则升级时见
[Shell permission migration](docs/shell-migration.md)。
