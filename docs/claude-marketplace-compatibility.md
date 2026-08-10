# Claude Marketplace 与插件兼容

Polaris 的兼容基线锁定为 2026-08-09 的 Claude Marketplace、Plugin、MCP、Plugin Dependencies 与 Dynamic Workflows 规范。Marketplace catalog 只负责发现；每次安装都会把可变来源解析成固定 commit 或内容 digest，运行时只加载已缓存制品。

## 默认受信 catalog

`agent.toml` 与 `agent.toml.example` 声明三个按需同步的 catalog：

```toml
[capabilities]
mode = "autonomous-trusted"
trusted_marketplaces = [
  "claude-plugins-official",
  "anthropic-agent-skills",
  "knowledge-work-plugins",
]
require_integrity = true
auto_components = ["skills", "agents", "mcp", "lsp"]
allowed_hooks = []
max_results = 8
marketplace_refresh_ttl_seconds = 86400
```

它们第一次参与 `capability_search` 时同步，之后按 TTL 刷新。配置表名必须等于 catalog 内的 `name`；刷新失败继续使用 last-known-good snapshot。Catalog 的 source 配置纳入项目 TOFU 指纹。

## Source 与 manifest

Marketplace source 接受 `owner/repo@ref`、Git URL、远程 HTTPS JSON、`file`、`directory` 和内联 `settings`。Plugin source 统一为 `relative`、`github`、`url`、`git-subdir`、`npm` 或 `archive`，并保留 `ref`、`path`、`sha`、`version`、`registry`、`headers` 与 `sha256`。

插件可以没有 `.claude-plugin/plugin.json`；根目录 `SKILL.md`、标准组件目录和 Marketplace entry 会组成最终 manifest。`strict: true` 合并 entry 与 plugin manifest，`strict: false` 由 entry 完整定义组件并拒绝冲突。未知字段在校验时警告，路径逃逸会拒绝。

Git 安装固定到 commit；npm 使用 `npm pack --ignore-scripts`；ZIP/tar archive 限制为 256 MiB 下载、1 GiB 解压、100,000 个条目，并拒绝私网重定向、zip-slip、特殊文件及恶意符号链接。插件内部相对链接保留，同一 Marketplace 的共享链接解引用，越界链接跳过。

## 运行时策略

- Skills、commands 与 agents 自动注册，支持官方 frontmatter、参数替换、supporting files、模型、effort、工具限制、memory 与 worktree 隔离。
- MCP 支持 stdio、streamable HTTP、SSE 与 WebSocket；LSP 读取扩展映射、初始化参数、settings、workspace folder、超时、重启和 diagnostics。自动激活的本地进程必须进入真实 Polaris 沙箱。
- Hooks、workflows、monitors、channels、themes、`bin` 和 plugin settings 属于确认组件。Workflow 在隔离的 Node 24 permission runtime 中仅获得 `args`、`agent()` 与 `pipeline()`，最多 16 并发、1,000 个 agent。
- Monitor 进程可取消、逐行限长；channel 只可绑定插件自有 MCP server。两者进入下一轮时都带来源标记，并作为不可信数据处理。
- Output styles 与 themes 只注册，不会静默替换用户当前选择。Plugin `bin` 不写进全局 `PATH`，plugin settings 不覆盖 Polaris 的安全提示或权限规则。
- `userConfig` 仅从用户级 Polaris 配置读取。敏感值存入 Windows Credential Manager、macOS Keychain 或 Linux Secret Service；系统密钥库不可用时只接受 `${ENV_VAR}`，明文不会落盘。非敏感值执行类型、范围和 required 校验。项目目录中的 plugin config 不参与命令替换。

自动激活采用两阶段 generation：先安装、校验并启动候选组件，再在模型调用边界原子切换；任何失败都保留上一代。`capability_activate` 会返回 `activated_next_turn`、`confirmation_required`、`configuration_required` 或结构化错误。

## 依赖

依赖解析支持 Node-semver 风格的 caret、tilde、比较器、hyphen、OR 与 prerelease 范围，检测循环、范围冲突和跨 Marketplace 引用。跨源依赖只有在根 Marketplace 的 `allowCrossMarketplaceDependenciesOn` 中列出时才允许。Git Marketplace 使用 `{plugin-name}--v{version}` tag 选择满足全部约束的最高版本，并把 tag commit 单独记录。依赖图安装只确认一次；失败恢复安装记录；`/plugin prune` 只清理不再被引用的自动依赖。

## 命令与迁移

```text
/plugin marketplace add <source>
/plugin marketplace add <name> <source>
/plugin marketplace update <name>
/plugin marketplace remove <name>
/plugin install <plugin@marketplace>
/plugin update|enable|disable|details|uninstall <plugin@marketplace>
/plugin configure <plugin@marketplace> key=value ...
/plugin validate <path|plugin@marketplace> [--strict]
/plugin prune [--dry-run]
/reload-plugins
```

## Capability state v3 and MCP Registry

Capability v3 deliberately does not migrate v1/v2 marketplace trust or enabled-plugin
state. On the first interactive start Polaris shows the exact managed targets and asks
before clearing them. A headless deployment must use `polaris plugins reset --dry-run`
followed by `polaris plugins reset --yes`; until then legacy state is ignored and remote
capability activation stays unavailable. `/plugin status`, `/plugin errors`, and
`polaris plugins status` expose the gate and recent redacted failures.

The model-facing flow is `capability_search` -> `capability_plan` ->
`capability_activate`. A plan pins source identity, marketplace snapshot, downloaded
bytes and dependency/permission requirements, expires after 15 minutes by default, and
cannot be altered by the model. Only immutable Anthropic first-party prompt-only
skills/agents may activate unattended. Community skills fork into a child with a real
tool allowlist; MCP, LSP, workflows, hooks, package scripts and every MCP Registry entry
require host approval and an enforcing sandbox where applicable.

The MCP Registry provider is search-only by default and normalizes all official
distribution methods: remote, npm, PyPI, NuGet, OCI and MCPB. Package resolvers use
exact versions plus content digests (OCI repository digest and MCPB `fileSha256`), and
activation consumes only the content-addressed plan artifact. Registry-origin tools have
a local `DANGEROUS` risk floor regardless of publisher annotations.

Marketplace records are keyed by a canonical source hash. The three reserved Anthropic
marketplace names cannot be rebound, and an official catalog entry that points to a
third-party repository remains community trust. Refresh state records last attempt,
last success, retry backoff and last-known-good snapshot. Plugin records, marketplaces,
configuration and audit appends use cross-process locks and atomic replacement.

旧版 `marketplaces.json`、`installed.json` 及插件启用状态不会继承到 capability v3。交互启动必须确认清理，无头部署必须执行显式 reset；直接 `[mcp.servers]` 配置不受此迁移门禁影响。人工安装后可在 Agent 空闲边界运行 `/reload-plugins`，通过 Activation Plan 激活的组件会在下一次模型调用前自动提交。
