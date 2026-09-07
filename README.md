# Polaris / Agent with LLM

## Reliability and retention defaults

Session cleanup is enabled by default. Resumable transcripts are retained for 90 days
and capped at 200 per project. Transcript cleanup runs at most once per day and
protects the current session, active sessions, tagged and malformed transcripts.
Preview or apply the transcript plan with:

```console
polaris sessions prune
polaris sessions prune --apply
```

Set `[session.retention] enabled = false` to disable transcript automatic deletion.
Automatic terminal-journal cleanup is suspended during P0 containment. The journal
retention settings remain readable for compatibility and explicit library maintenance.
The remaining limits and protection switches are documented in `agent.toml.example`.

### Recovery containment

Startup and `/resume` inspect recovery state without deleting overlays, restoring
workspace files, writing recovered transcript rounds, repairing indexes, or pruning
journals. Only a separate recovery audit record may be appended. Unfinished or
unverifiable recovery state pauses the selected session before model/tool execution;
`/resume` keeps the current session when its target is blocked.
Active parent/sibling tool rounds are recognized only through this process's real
locked journal objects, so subagents can continue normally. On-disk PID or process
token fields never exempt an abandoned journal from the startup check. Legacy open
indexes are no longer updated; recovery scans the session's shallow run partitions
so an incomplete index cannot conceal an unfinished journal.

From the owning project directory, review and explicitly apply recovery:

```console
polaris recovery --session-id SESSION_ID
polaris recovery --session-id SESSION_ID --dry-run --json
polaris recovery --session-id SESSION_ID --apply
```

The recovery command does not construct an Agent or start providers, hooks, MCP or
sandbox processes. It examines only that project's specified session, across prior
runs. Supply `--session-dir PATH` when the session used a custom transcript root;
the default is `AGENT_SESSION_DIR` or `~/.polaris/projects`. This standalone command
does not load repository configuration. An empty `--session-dir ""` disables history
writes. The journal's recorded transcript target must match the runtime-derived path.
`--apply` never overrides validation; busy, foreign, corrupt or incomplete recovery
returns exit code 1, invalid arguments return 2, and a valid preview or successful
apply returns 0. Preview and apply re-read the journal; close other processes using
the session before applying.

Journals stay under `POLARIS_HOME/recovery-journals` (default
`~/.polaris/recovery-journals`), partitioned by project/session/run. The state directory
must be outside the workspace, privately writable, and free of symlink/junction
redirects. An unavailable private directory fails explicitly without a shared-temp
fallback. Existing workspace `runs/.turn-journals` files are never imported, executed
or removed automatically. v3/v4 checksum chains remain readable; checksums detect
corruption and do not authenticate the author.

Embedding APIs `recover_all()` and `recover_turn_journals()` now default to
`dry_run=True`; actual recovery requires an explicit `dry_run=False` call. Use
`inspect_recovery()` for a structured `RecoveryReport`; blocked `run()` calls raise
`RecoveryRequiredError`. Recovery never retries external tool calls. Failures preserve
the journal and backups, and leave a diagnostic in `recovery-audit.log`. A durable
versioned recovery checkpoint permits retrying cleanup after workspace/history
actions succeeded. Unverifiable journals and indeterminate external effects require
manual investigation; do not delete their evidence to bypass the startup check.

Automatic memory extraction processes model output item by item. Permanently invalid or
secret-bearing items go to a bounded 1,000-entry diagnostic dead-letter queue without
their original content; valid siblings are still stored and infrastructure failures do
not advance the extraction cursor. Scheduler deliveries use three attempts with 30s/60s
exponential retry (capped at 600s) and a 1,800s lease. An agent can inspect and explicitly
redrive only its own dead letters with `cron_delivery_list` and `cron_delivery_retry`.

OpenAI-compatible streaming requests ask for usage with
`stream_options.include_usage=true`, including support for usage-only terminal chunks.
If an endpoint explicitly rejects that option, Polaris retries once without it and caches
the endpoint capability; unrelated 4xx responses are never retried as compatibility
fallbacks.

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

安装器会复用已通过完整挂载探针的 Podman、Docker 或 nerdctl；三者都不可用时安装 Podman，并
拉取带 OCI index 摘要的 GHCR 工具链镜像。Windows 首次启用 WSL2 后可能返回退出码 `20` 并要求重启；重启后重新运行同一条命令
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
配置中设置 `[sandbox] enabled = true`。Windows Guest 协议、路径映射、fail-closed 状态和安全
边界见 [WSL2 OCI sandbox](docs/sandbox-wsl2.md)。

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

常用会话命令包括 `/rename`、`/effort`、`/fast`、`/sandbox`、`/model` 和 `/status`。`/sandbox`
会显示 requested/effective/prepared、Runtime、完整镜像摘要、Guest OS、能力表和失败原因；退出隔离
只允许在启动新会话时显式传入 `--no-sandbox`。

## 长期记忆

长期记忆以独立主题 Markdown 为权威数据源，主代理私有记忆保存在用户目录，
团队记忆保存在当前 checkout 的 `.polaris/memory/team/`。系统支持中文检索、
跨进程原子写入、秘密扫描、可恢复遗忘和旧 JSONL 无损迁移；召回内容始终按
不可信历史数据处理，涉及当前代码或配置的说法需要重新验证。命令、目录结构、
迁移和隐私边界见 [长期记忆文档](docs/long-term-memory.md)。

检索使用可重建的本地 SQLite 索引，默认执行 exact → BM25 → 自适应 BGE-M3
dense（小集合分块精确扫描，大集合 USearch HNSW 召回后 FP32 精确重算）→
加权 RRF → BGE reranker，只注入命中的完整片段。`MEMORY.md` 的 200 行上限仅是
人类摘要限制，不限制主题枚举。可用 `polaris memory index status` 和
`polaris memory models status` 检查覆盖率；离线安装模型使用
`--model-bundle PATH`，只有安装器的 `--skip-memory-models` 会显式允许词法降级。

## Claude 兼容插件

Marketplace source、不可变制品、依赖、组件策略和旧配置迁移详见
[Claude Marketplace 兼容说明](docs/claude-marketplace-compatibility.md)。

`/plugin` 支持 install/update/manage/details/uninstall/enable/disable/validate/configure/prune，
以及 marketplace 的 add/remove/update/list。三个 Anthropic catalog 由能力配置按需同步；安装记录和
不可变缓存位于 `~/.polaris/plugins`，项目启用状态默认写入 `agent.local.toml`。Manifestless、根目录
skill、agents、hooks、MCP、LSP、workflows、monitors、channels 和展示/配置组件均参与统一校验；
会执行代码、注入消息或改变提示的组件在启用前需要确认。

安装或启用不会修改正在运行的组件代；在 Agent 空闲时运行 `/reload-plugins` 才会构建并原子切换，
失败时继续使用旧代。插件组件使用 `plugin:component` 命名空间，安装 ID 使用
`plugin@marketplace`。

## 运行时能力发现

Agent 可通过 `capability_search` 自行检索当前 skills、已连接但尚未暴露的 MCP tools、已安装
plugins，以及明确列入信任名单的 marketplace 元数据。仓库示例配置启用
`"autonomous-trusted"` 并声明三个真实 Anthropic catalog；改为 `"local"` 可关闭远程下载。
可用 `/capabilities <query>` 检查运行时目录，或用 `polaris capabilities search <query>` 检查
当前配置的发现来源。

将模式设为 `"autonomous-trusted"` 后，Agent 会自动搜索配置的 Marketplace 和 MCP Registry。搜索结果必须先由
`capability_plan` 解析为固定 source identity、commit 和内容 digest；Marketplace snapshot 本身不再被当作插件制品
完整性证明，也不能把任意 URL、路径或命令传给安装器。激活在当前工具批次结束的回合边界原子提交，候选组件
验证失败时保留旧 generation。默认无人值守激活只允许 Anthropic 第一方、不可变、低风险且只包含 skills/agents
的制品；MCP、LSP、hooks、workflows、bin、依赖安装、社区内容及所有 Registry 条目都需要宿主确认。

可增加组织自己的受信 Marketplace；表名必须与远端 manifest 的 `name` 一致：

```toml
[capabilities]
mode = "autonomous-trusted"
trusted_marketplaces = ["team"]
require_integrity = true
auto_components = ["skills", "agents", "mcp"]
allowed_hooks = []
```

Capability v3 uses a three-step host-owned activation protocol:
`capability_search` discovers component-level metadata, `capability_plan` freezes the
source and artifact digests, and `capability_activate` commits the verified generation.
The default unattended allowlist is limited to immutable low-risk Anthropic first-party
skills/agents. Third-party/community content and every MCP Registry connection/package
(remote, npm, PyPI, NuGet, OCI, MCPB) require direct host approval.

Old plugin state is intentionally not trusted after this upgrade. Interactive startup
offers an exact reset preview; for headless use `polaris plugins reset --dry-run` and
then `polaris plugins reset --yes`. `polaris plugins status`, `polaris plugins errors`,
`/plugin status`, and `/plugin errors` expose state and redacted failures.

Marketplace 的 source、snapshot、制品 digest、依赖图、组件选择和安全决策都会进入安装审计。
该配置扩大仓库权限边界，因此仍受项目 TOFU 信任检查保护。

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

## SWE-bench Lite runner

The benchmark integration is isolated from the normal interactive Agent path. Install
the optional dependencies (Docker Engine/Desktop must also be available):

```powershell
pip install -e ".[swebench]"
```

Add `,terminal` (or use `[all]`) when using the optional `--live` console trace.

List safe task metadata first. The loader never writes `patch`, `test_patch`,
`FAIL_TO_PASS`, or `PASS_TO_PASS` into the Agent workspace or prompt:

```powershell
polaris swebench list --dataset SWE-bench/SWE-bench_Lite --split dev --limit 20
```

For a manual 3-5 task smoke run, repeat `--instance-id` or provide a YAML/JSON
selection manifest:

```powershell
polaris swebench smoke --split dev --instance-id task-id-1 --instance-id task-id-2 --provider claude --evaluate --live --keep-workspaces
```

For a complete Lite test split, selection is intentionally explicit:

```powershell
polaris swebench run --dataset SWE-bench/SWE-bench_Lite --split test --all --provider claude --evaluate --solve-workers 1 --evaluation-workers 1
```

Each run is stored below `swebench_runs/<run_id>/` with public task rows, per-instance
state/logs, `patch.diff`, official-format `predictions.jsonl`, Harness output, and
`summary.json`/`summary.csv`. Use `--resume --run-id <id>` to continue a stopped run
(the prior ID-only selection manifest is reused automatically); add `--retry-failed` to
retry failed instances. `--runtime local` is an explicit
development/test fallback and is not an isolation substitute for the default Docker
runtime.
