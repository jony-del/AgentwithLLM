# Gap Analysis：参考项目（Claude Code）agent runtime vs. AgentwithLLM

> 参考项目：Claude Code 的 TypeScript 源码（`src/` ~1900 文件）。
> 本报告只读取了与 agent runtime 直接相关的核心文件（`query.ts`、`QueryEngine.ts`、
> `Tool.ts`、`context.ts`、`Task.ts`、`services/compact/*`、`utils/permissions/*`、
> `skills/*`），对照本项目的 `react.py`、`compression.py`、`permissions.py`、
> `session.py`、`cli.py` 等。分析仅做架构对比，不复制源码。

---

## 1. 参考项目的核心 agent loop

核心循环在 `src/query.ts` 的 `async function* queryLoop(params)` —— 一个 **async
generator**，`yield` 出 `StreamEvent | Message | ToolUseSummaryMessage |
TombstoneMessage`，最终 `return` 一个 `Terminal`（带 `reason`）。`QueryEngine.ts`
在外层包装 `query()`，把 stream 转成 SDK 消息、累计 usage/cost。

| 阶段 | 参考项目做法 | 关键代码 |
|---|---|---|
| **输入进入** | UI / SDK 收集用户消息 → `QueryEngine` 组装 `QueryParams`（`messages`、`systemPrompt`、`userContext`、`systemContext`、`canUseTool`、`toolUseContext`）→ `query()` | `QueryEngine.ts`, `query.ts:181` |
| **构造上下文** | 三段式分离：`systemPrompt`（指令）＋ `systemContext`（git status，`context.ts:getSystemContext`，memoized）＋ `userContext`（CLAUDE.md / memory files / 当前日期，`getUserContext`）。`prependUserContext()` / `appendSystemContext()` 每次调用前拼接；`getMessagesAfterCompactBoundary()` 只取 compact 边界之后的消息 | `context.ts`, `query.ts:365,449,660` |
| **调用模型** | `deps.callModel({...})` 流式；每个 delta 立刻 `yield`；`tool_use` 块边到达边喂给 `StreamingToolExecutor`（模型还在出 token 时工具已经开跑） | `query.ts:659,841` |
| **选择/执行工具** | 两条路径：`StreamingToolExecutor`（gate 开时边流边执行）或 `runTools()`（批量）。每个工具经 `canUseTool`（权限）→ 执行 → `yield` 结果消息 | `query.ts:1366,1380` |
| **处理工具结果** | `normalizeMessagesForAPI()` 把结果规范成 `user` 消息（tool_result block）追加；`applyToolResultBudget()` 对超大结果做 per-message 预算裁剪；可选生成 `toolUseSummary`（Haiku 异步生成，下一轮 yield） | `query.ts:379,1395,1411` |
| **继续下一轮** | 不是递归而是 `while(true)` + 显式 `state = {...}; continue`。`State` 是显式可变结构体，7 个 continue 点各自重建（fallback / 413 恢复 / max_output_tokens 恢复 / stop-hook 阻断 / token budget 续跑） | `query.ts:204,307,1192` |
| **何时终止** | `needsFollowUp === false`（无 tool_use 块）→ 跑 stop hooks → 不阻断则 `return {reason:'completed'}`。其它终止：`blocking_limit`、`aborted_streaming`、`model_error`、`prompt_too_long`、`image_error`、`stop_hook_prevented` | `query.ts:1062,1357` |

**循环最显著的设计取向（与本项目同源）**：没有固定 step cap，靠"模型不再请求工具"
自然终止 + 一组显式安全闸（cancel、token budget、wall-clock）。

**参考项目独有的循环机制**：

- **多级 token 恢复管线**：proactive autocompact → snip → microcompact →
  context-collapse → 命中 413 后 reactive compact / collapse drain →
  max_output_tokens 升档重试（8k→64k）→ 多轮 recovery 提示。全部在循环内、靠
  `continue` 重入。
- **流式工具执行**：工具不等模型整轮结束就开跑。
- **fallback 模型切换**：`FallbackTriggeredError` 时切模型、吐 tombstone 清理孤儿消息、
  重试整个请求。
- **thinking block 不变量**：流式 fallback 时对 assistant 消息打 tombstone，因为
  partial thinking block 的 signature 失效会让 API 报错。

---

## 2. 关键模块边界对照

| 层 | 参考项目 | 本项目 | 状态 |
|---|---|---|---|
| **CLI / UI** | `main.tsx` + Ink (React TUI)，`src/ink/`、`src/components/` 几百个文件，富交互 | `cli.py`（argparse：run/chat/dream/memory/health/mcp）+ `ui.py`（事件 sink：NullUI/ConsoleUI） | 架构对齐，规模天差 |
| **session / transcript** | `bootstrap/state.js`（sessionId、全局状态）、`history.ts`（命令历史、paste/image ref、`/resume` 持久化）、session ingress | `transcript.py`（按 `session_id` 命名的追加式 Message JSONL，`uuid`/`parent_uuid` 消息树 + **内建 compact-boundary**）、`session.py`（per-run SessionContext + `session_id` 贯通）、CLI `--resume`/`--continue`/`--fork-session`/`sessions list` | **✅ 已实现**（见 §3.2-2、§3.4-3） |
| **context mgmt / compaction** | `services/compact/`（autoCompact、microCompact、snipCompact、reactiveCompact、contextCollapse、sessionMemoryCompact、cachedMicrocompact）—— LLM 摘要 + cache 编辑 | `compression.py`（snip / microcompact / context_collapse 三段，**纯字符串截断**，无 LLM 摘要） | 流程同构，**手段简化** |
| **tool registry / execution** | `Tool.ts`（统一 contract：`isEnabled`、`validateInput`、`checkPermissions`、`call` async gen、`mapToolResultToToolResultBlockParam`、`backfillObservableInput`）；`StreamingToolExecutor` + `runTools` | `tools/`（`@builtin_tool` 自注册、`registry`、`executor` 带 wave 分区并发、`_invoke`/async `run`）、`ToolRisk` | **设计对齐良好** |
| **permission / approval** | **规则驱动**：`alwaysAllow/Deny/AskRules` + `bashClassifier`、`pathValidation`、`dangerousPatterns`、`yoloClassifier`、shadowed-rule 检测；PermissionMode 是其中一维 | **模式驱动**：`PermissionPolicy` 按 `PermissionMode` + `ToolRisk` 决策，session allowlist | **设计哲学不同**（见 §3） |
| **config / memory / 项目指令** | `utils/settings/`、`utils/config.js`、CLAUDE.md 发现（`claudemd.ts`）、`memdir/`（auto memory） | `config.py`（defaults→toml→env→flag 四级）、`memory/`（store/retrieval/extraction/dreaming，opt-in）、**无 CLAUDE.md 注入** | config 对齐；memory 本项目更完整；**缺项目指令注入** |
| **扩展层** | subagent（`AgentTool`）、skill（`skills/bundled/` 17 个 + `loadSkillsDir` 用户技能）、hook（pre/post tool、stop、postSampling、userPromptSubmit，**settings.json 外部进程加载**）、MCP（完整 client+transport+oauth）、plugin marketplace、Task/swarm 多 agent | subagent（`dispatch_agent`）、team（多 agent 协作）、hook（pre/post tool **+ 生命周期：Stop/PostSampling/UserPromptSubmit/Pre·PostCompact，编程式注入**）、MCP（client/adapter/config） | subagent/MCP 对齐；**hook 事件面已补齐但仍是编程式（无外部配置加载器，见 §4）**；无 skill 系统、无 plugin、无 slash command |

---

## 3. Gap Analysis

### 3.1 已经类似 ✅

- **核心循环骨架**：async、流式、无固定 step cap、自然终止 + 安全闸。`react.py` 与
  `query.ts` 在哲学和主干流程上高度一致。
- **跨层数据契约**：`Message/ToolCall/ToolResult/LLMResult` ↔ 参考的
  `Message`/tool_use/tool_result，都是稳定 contract。
- **thinking block 保真**：已保留 `thinking_blocks` 并在下一轮回放——正是参考项目反复
  强调的"thinking 不变量"。
- **工具自注册 + Risk 分级 + 工作区限制**：`@builtin_tool` / `ToolRisk` /
  `WorkspacePathMixin` 对应得很干净。
- **并发预算**：`GatedProvider`（semaphore+token bucket）+ executor wave 分区 ≈ 参考的
  API 并发 gate + 资源冲突串行化。
- **compaction 三段管线 + 进度上报**：结构与参考同构。
- **subagent 防递归**：排除子 agent 的 `dispatch_agent` + depth ceiling，对应参考的
  限制思路。

### 3.2 缺失的部分 ❌（按缺口大小）

1. ~~**LLM 驱动的 compaction**：参考用 forked agent 调模型生成 `<analysis>+<summary>`
   摘要；本项目是纯字符串截断（snip/omit）。**这是上下文质量的最大差距**——长任务里
   截断会丢语义，摘要不会。~~ **✅ 已实现**：`compression_summary.py` 实现了完整的
   Track A LLM 摘要器——9 节 prompt（移植自参考 `BASE_COMPACT_PROMPT`）、`<analysis>+<summary>`
   解析、无工具强制前导（`_NO_TOOLS_PREAMBLE`/`_NO_TOOLS_TRAILER`）、不可信 transcript
   定界符（`<transcript>`）、升档 `_budget_ladder`（steady-state → compact_max → 模型硬限），
   全部共享 `GatedProvider` 预算。`react.py:214` 在初始化时构建 `self._summarizer`；
   proactive 与 reactive 两路压缩均传入该 summarizer；`FakeProvider` / 无 key / `use_llm_summary=False`
   时自动降级到 Track B 字符串截断，保持离线/测试环境 byte-stable。
2. ~~**会话持久化 / resume**：参考有 session 文件 + `/resume` + 历史。本项目只有
   `runs/*.jsonl` 单向日志，不能续接会话。~~ **✅ 已实现**：新增 `transcript.py`——
   按 `session_id` 命名的追加式 Message round-trip JSONL（`uuid`/`parent_uuid` 消息树，
   含 thinking_blocks/tool_call 关联保真），CLI `--resume`/`--continue`/`--fork-session`
   + `sessions list`，`chat` 跨轮记忆，跨项目定位。**并已对标参考的 transcript 内建
   compact-boundary**：一次 fold 把边界 + 摘要写盘，`--resume` 直接加载压缩后状态（含
   `>5MB` fd 级字节截断 + 边界前元数据抢救扫描）；可经 `persist_compaction_boundary`
   开关回退到忠实全量历史。`runs/*.jsonl` 事件日志保持不变、各司其职。
3. ~~**项目指令注入（CLAUDE.md）**：参考把 CLAUDE.md / memory files 作为 `userContext`
   每轮注入。本项目有 memory 系统但**不读取/注入项目级 CLAUDE.md**。~~ **✅ 已实现**：
   `context.py:build_project_instructions()` 从 workspace 向上发现并合并所有 CLAUDE.md（VCS
   根 → workspace 优先级顺序 + 可选 `~/.claude/CLAUDE.md`），注入为 pinned `<system-reminder>`
   userContext 中的 `claudeMd` 条目（`react.py:672-682`），截断至 `claudemd_max_chars`（默认
   32000）。`build_git_status()` 同步实现 git 快照注入（`systemContext.gitStatus`），带单次
   `asyncio.wait_for` 超时 + subprocess kill-await，对标参考 §3.5-P0-1/2。
4. **规则驱动权限 + bash 分类器**：参考能对单条命令做细粒度 allow/deny/ask（按命令
   前缀、路径、危险模式）。本项目粒度到"工具级"，无法表达"允许 `git status` 但 ask
   `git push`"。
5. ~~**更丰富的 hook 类型**：参考有 stop hook（可阻断/续跑）、postSampling hook、
   userPromptSubmit hook、preCompact hook。本项目只有 pre/post tool。~~ **✅ 已实现
   （编程式生命周期钩子）**：新增 `UserPromptSubmit`/`PostSampling`/`PreCompact`/
   `PostCompact`/可阻断的 `Stop` 五类钩子，在 `react.py` 循环边界按固定顺序触发，
   `hooks.py:HookPipeline.run_*` 折叠执行；Stop 续跑由 `config.max_stop_blocks` 封顶。
   **保持编程式（构造期注入），尚未实现参考的 settings.json 外部进程加载器**——两者的
   实现细节、差别与演进指引见 **§4**。
6. ~~**skill / slash command 系统**：参考有 bundled skills + 用户技能目录 + slash
   command。本项目无。~~ **✅ 已实现**：新增 `agent_core/skills/`（纯 stdlib）——
   Markdown `SKILL.md` + frontmatter 加载器（`loader.py`，bundled→用户 `~/.polaris/skills`
   →项目 `./.polaris/skills`→额外目录的优先级覆盖、realpath 去重、`disabled` 过滤）、
   `SkillRegistry`、slash 解析/渲染（`dispatch.py`）、自写极简 frontmatter 解析器
   （不引 PyYAML）、`bundled/*.md` 内置技能（`commit`/`review`）。**人类**经 chat 的
   `/command` 触发（`cli.py` 派发 + `/help`/`/skills`）；**模型**经新增 `tools/skill.py`
   的 `skill` 工具自主调用（`schema_for_llm` 动态列出可调用技能，无可调用技能时从 registry
   隐藏）。支持 `inline`（注入指令）与 `fork`（复用 depth-limited `subagent_factory` 的隔离
   子 agent）两种执行上下文；frontmatter 的 `user-invocable`/`disable-model-invocation`
   分别控制人/模型可见性。子 agent 排除 `skill` 工具防递归。`[skills]` 配置 + `AGENT_SKILLS`
   开关，加载失败降级为空 registry 不崩 run。
   **Phase 2（与参考务实对齐）**：frontmatter 解析切换到 **PyYAML**（接口不变、坏 YAML 降级）；
   bundled 静态技能扩到 8 个（commit/review/verify/simplify/remember/init/pr-review/security-review，
   含把参考 prompt-type 命令 `/init`·`/review`·`/security-review` 实现为技能）；新增
   **programmatic（Python 可调用）技能接缝**（`skills/programmatic.py` 的 `@programmatic_skill`
   自注册 + `SkillPromptContext` + `build_skill_prompt`，对标参考 `getPromptForCommand`），
   移植切题的动态技能 `lorem-ipsum`/`skillify`/`debug`；新增 `agent_core/chat_commands.py`
   提供约 11 个**接本项目真实子系统**的内置 chat 命令（`/help`·`/skills`·`/clear`·`/status`·
   `/context`·`/cost`·`/compact`·`/model`·`/mcp`·`/memory`·`/resume`，`ChatTurn` 返回支持
   有状态命令；参考的 TUI/账号/云端命令无对应物故不做）。支撑改动：`ReActAgent.compact_now()`
   强制压缩、会话累计用量计数。全套 **550 绿**。
7. **流式工具执行**：参考边流边跑工具；本项目是整轮 LLM 结果出齐后再 `execute_many`。
8. **413 / max_output_tokens 的分级恢复**：本项目有 `LLMContextTooLongError` →
   reactive_compact 一次；参考有 collapse-drain → reactive → 升档重试 → 多轮 recovery
   的完整链。
9. **fallback 模型切换**与 tombstone 清理。
10. ~~**tool result 预算裁剪 + tool-use summary**（Haiku 异步摘要降低上下文占用）。~~
    **✅ 已实现**。**订正**：参考的两件事职责不同 ——
    (a) **预算裁剪**（`applyToolResultBudget`）才是真正「降低上下文占用」的部分，本项目早已由
    `hooks.py:MaxOutputPostHook` 的 pointer 模式（`<tool_output_ref>` 结构化预览指针，对标参考
    `toolResultStorage`）完成；
    (b) **tool-use summary**（`toolUseSummaryGenerator.ts` / `query.ts:1410-1482`）其实**不缩
    上下文** —— 它是每个工具批次后由 Haiku 异步生成的一句 ~30 字进度标签，**只发 SDK/UI**，
    `normalizeMessagesForAPI` 从不含它（不进 API、不进 transcript）。原条目括号「降低上下文占用」
    系对 (b) 的误标。
    现新增 `tool_use_summary.py`（镜像 `compression_summary.py` 的注入闭包 seam）忠实复刻 (b)：
    fire-and-forget 异步 Haiku 标签、下一轮 model 调用期间并发跑完、`ui.on_tool_use_summary` +
    `runs/*.jsonl` 观测日志、**绝不进 API messages / transcript**；`[tool_use_summary]` 配置
    （默认关、可选模型、单次非堆叠超时、leader-only）、live-UI + 非 Fake 闸、离线 byte-stable。
    共 15 个新测试（`tests/test_tool_use_summary.py`），全套 423 绿。

### 3.3 设计不同但可以保留 🔵

- **模式驱动 vs 规则驱动权限**：本项目 5 种 `PermissionMode`（含 `dontask`/`auto`）+
  session allowlist 简单清晰，对中型 agent 完全够用。**建议保留**，未来在其上叠加一个
  可选的"命令级规则"层即可，不必推翻。
- **纯字符串 compaction**：作为"无 API key 也能跑"的 deterministic 兜底**很有价值**。
  建议演进成"默认 LLM 摘要，降级到字符串截断"的双轨，而非替换。
- **team 多 agent 抽象**：`team_create/teammate_spawn/task_update` 是参考 swarm 的
  轻量版，方向一致，保留。
- **memory 系统**：store/retrieval/extraction/**dreaming** 比参考的 memdir 更完整
  （dreaming 是本项目特色），保留并继续投入。
- **三段式 config 精确层级**：清晰，保留。

### 3.4 高风险重构点 ⚠️

1. **把 compaction 改成 LLM 摘要**：触及 `compression.py` + `react.py` 的
   reactive/auto 调用点，且摘要要走 `GatedProvider`（异步、可能失败）。风险：摘要失败
   时的兜底、摘要本身吃 token/费用、与 memory extraction 的职责重叠。**务必保留字符串
   截断作为降级路径**，并确保"compact 失败不能让 run 崩"。
2. **流式工具执行**：要求 provider 在流式过程中逐块暴露 tool_use，并引入"工具执行器
   消费流"的并发模型。会改动 `providers/claude.py` + `react.py` + `executor`。风险高、
   收益对中型 agent 有限——**建议靠后**。
3. ~~**会话持久化/resume**：需要定义 transcript 序列化格式（含 thinking blocks
   signature、tool_result 关联、compact 边界）。`Message` 契约可能要扩展
   `uuid`/`parent_uuid`。**契约变更影响所有层**，需 deliberately + 更新测试。~~
   **✅ 已实现**：`Message` 末尾加 `uuid`/`parent_uuid`（`compare=False`，等值语义不变、
   位置构造合法、provider 不受影响）+ `from_dict`；序列化格式见 `transcript.py`。
   compact-boundary 用 **before/after uuid 差集**推断 fold 后写「summary 根（`parent_uuid=None`
   + `compact_boundary` 标记）+ 轻量 relink 记录」，不侵入 `compression.py` 内部结构、
   不重写已落盘消息；持久化失败不崩 run。共 18 个新测试（`tests/test_transcript.py` +
   `tests/test_transcript_boundary.py`），全套 408 绿。
4. **规则驱动权限**：引入 bash 命令解析/分类器是一个独立的复杂子系统（参考有 24 个
   文件）。风险在于解析的安全正确性（命令注入、shell 拼接绕过）。**若做，先做小：只做
   命令前缀 allow/deny 规则，不做完整 classifier**。

### 3.5 建议优先级

**P0（高收益 / 低风险，先做）**

1. ~~**CLAUDE.md 注入**：在 `react.py` 组装初始 messages 时，读取 workspace 的
   `CLAUDE.md` 注入为 system/pinned 块（仿 `context.ts:getUserContext`）。改动局部、
   收益立竿见影。~~ **✅ 已完成**（见 §3.2-3）。
2. ~~**git status 上下文块**：仿 `context.ts:getGitStatus`，run 开始时注入一次性 git
   快照。小、独立。~~ **✅ 已完成**（见 §3.2-3）。
3. ~~**LLM 摘要式 compaction（双轨）**：默认走模型摘要、降级到现有字符串截断。这是缩小
   与参考差距的**最高杠杆**项。~~ **✅ 已完成**（见 §3.2-1）。

**P1（中收益，中风险）**

4. ~~**stop hook**（可阻断/可续跑）——让"任务未完成时强制继续/收尾"成为可能。~~
   **✅ 已完成**（连同 UserPromptSubmit/PostSampling/Pre·PostCompact 一并补齐，见 §3.2-5、§4）。
5. ~~**413 分级恢复**——在现有 `reactive_compact` 上补多轮 prompt-too-long 恢复。~~
   **✅ 已完成**：`react.py` 在 `LLMContextTooLongError` 后先 `reactive_compact`，
   再用 `truncate_head_for_ptl_retry` / `shrink_oversize_messages` 做有界多轮重试
   （见 `tests/test_react.py` Phase 3D）。**max_output_tokens 续跑**仍未完成：主循环尚未对
   `LLMResult.stop_reason == "max_tokens"` 做续跑；目前只在 compaction summarizer 内有输出预算升档。
6. ~~**tool result 预算裁剪**——超大工具输出按 per-message 预算截断（已有
   MaxOutputPostHook，可扩展）。~~ **✅ 已完成**：预算裁剪由 `MaxOutputPostHook` pointer 模式
   承担；参考的 tool-use summary（UI 进度标签，非上下文缩减）也已忠实复刻（见 §3.2-10）。

**P2（架构投入，按需）**

7. ~~会话持久化 / resume（需契约扩展，谨慎）。~~ **✅ 已完成**（含 transcript 内建
   compact-boundary，见 §3.2-2、§3.4-3）。
8. ~~skill / slash command 系统。~~ **✅ 已完成**（见 §3.2-6）。
9. 流式工具执行（收益/复杂度比对中型 agent 偏低）。
10. 规则驱动权限（从命令前缀规则起步）。

---

## 4. 钩子机制：当前实现 + 配置驱动加载器演进指引

> 本节记录本项目当前（编程式）钩子的完整实现，并对照参考项目的「settings.json 外部配置
> 钩子」，作为后续设计「配置驱动的 command/http 钩子加载器」的指导文档。

### 4.1 当前实现（编程式生命周期钩子）

本项目原先只有 pre/post **工具**钩子（`HookPipeline` 的 `PreToolHook`/`PostToolHook`）。
现已补齐一组**生命周期钩子**，与参考的事件面对齐，但保持**编程式**（构造期注入），不引入
参考的 settings.json 外部进程机制。落点：`agent_core/hooks.py` + `agent_core/react.py` +
`agent_core/compression.py`。

**两类钩子的分工**：

| 类别 | 事件 | 同步/异步 | 触发处 |
|---|---|---|---|
| 工具钩子（原有） | PreToolUse / PostToolUse | 同步 | `ToolExecutor`（每个工具调用前后） |
| 生命周期钩子（新增） | UserPromptSubmit / PostSampling / PreCompact / PostCompact / Stop | 异步 | `ReActAgent.run()` 循环边界 |

工具钩子在 executor 内**同步**运行（执行器把阻塞工具卸载到线程，故钩子保持同步）；生命周期
钩子在 async 主循环里直接 `await`，可做真正的工作（调模型、跑 verifier）而不阻塞循环线程。

**核心数据结构**（`hooks.py`）：

- `HookEvent`：事件枚举。
- `HookContext`：单一上下文形状 + 事件相关可选字段（`prompt` / `trigger` / `summary` /
  `last_assistant_message` / `stop_hook_active`），镜像参考的 per-event 输入但不做类爆炸。
- `HookOutcome`：单一 `block` 决策位（**语义按事件而定**）+ `additional_context`（注入文本）
  + `reason`。
- `HookPipeline.run_*`：按序折叠钩子，遇首个 `block` 短路；`run_post_sampling` 为 fire-each、
  无返回（观测性）。
- `compression.should_compact()`：无副作用谓词，镜像 `auto_compact` 自身闸门，让 `PreCompact`
  只在 fold 真正迫近时触发。

**钩子在循环中的逻辑顺序**（已写入 `CLAUDE.md` 的「Agent Loop」契约）：

0. **UserPromptSubmit** —— 任务就位、首次 model 调用之前。可中止整个 run（`block`），或注入
   `additional_context`（作为 `<system-reminder>` 让模型本轮看到）。
1. **PreCompact** —— 仅当 proactive fold 真正迫近时（`should_compact()`）→ 压缩 →
   **PostCompact**（仅当真的发生 fold、带新 summary）。`PreCompact.block` 跳过 proactive
   fold；在被迫的 **reactive（413 恢复）路径上 block 被忽略**（压缩是强制的）。
2. 调 model → **PostSampling**（fire-and-forget；assistant 轮落历史后触发，与下一次 model
   调用并发；在终止返回处 reap）。
3. 无 tool_use 时 → **Stop**：钩子可 **block 本次停止**强制续跑（"可阻断/可续跑"），由
   `config.max_stop_blocks`（默认 3）封顶，注入续跑指令后重入循环；超过上限则照常停止。
4. 经 `ToolExecutor` 执行工具（其内部对每个调用跑同步 pre/post 工具钩子）。
5. 工具观察追加为 `tool` 消息，continue。

**安全性 / 兜底**：

- 默认无生命周期钩子时，所有 seam 都是廉价 no-op，行为与改动前**完全一致**。
- Stop 续跑有硬上限（`max_stop_blocks`），且 `stop_hook_active` 在首次 block 后置 True ——
  比参考多一道防失控护栏（参考只有 `preventContinuation`，无次数上限）。
- PostSampling 是观测性 fire-and-forget，异常只记日志，绝不 sink run。
- 测试：`tests/test_lifecycle_hooks.py` 11 个用例（管线折叠/短路、UserPromptSubmit 阻断+注入、
  PostSampling 触发、Stop 阻断续跑+封顶+零上限禁用、Pre·PostCompact fold 触发+block 跳过），
  全套 467 绿。

### 4.2 与参考项目「settings.json 外部配置钩子」的差别

参考项目的钩子是**配置驱动的外部进程调用**：在 `settings.json` 按事件 + matcher 声明，每个
钩子是 `command`（子进程，stdin/stdout JSON，退出码 2 = block）/ `http`（POST 端点）/
`prompt`（再发一次 LLM）/ `agent`（跑 verifier agent）之一。两者本质是**「钩子怎么被定义和
加载」**的两条路线：进程内代码注入 vs 配置驱动的外部进程调用。

| 维度 | 编程式钩子（本项目） | settings.json 外部钩子（参考） |
|---|---|---|
| 定义位置 | Python 代码，构造 `HookPipeline` 时注入 | `settings.json`，事件 + matcher |
| 谁能改 | 开发者（改代码、重部署） | 终端用户/运维（改配置即可，无需改源码） |
| 运行形态 | 同进程、同 Python 运行时 | 子进程 / HTTP / 再一次 LLM / verifier agent |
| 语言 | 必须 Python | 任意（bash / Node / curl / …） |
| 数据传递 | 直接传**活对象**（`HookContext`/`ToolCall`/`Message`） | 序列化 **JSON 快照**（stdin/stdout 或 HTTP body） |
| 阻断协议 | 返回 `HookOutcome.block` | 退出码 2 / JSON `{continue:false}` |
| 状态共享 | 能读写进程内对象/闭包/共享 store | 无共享内存，只有 JSON 快照 |
| 延迟/开销 | 进程内 await，近零 | spawn / 网络 / LLM，量级更高 |
| 安全面 | 由代码作者控制 | "配置即可执行任意命令/HTTP"，需信任边界/超时/沙箱 |

**取舍小结**：

- **外部钩子优势**：免改代码即可配置、语言无关、进程隔离、可分发可审计、天然支持
  `prompt`/`agent` 这类"重"钩子。
- **外部钩子劣势**：延迟大（高频的 PreToolUse/PostToolUse 尤甚）、只能传 JSON 快照（改不了
  进程内状态）、安全面大、失败模式多（超时/僵尸进程/网络/退出码解析）、实现量大。
- **编程式钩子优势**：零开销、能传/改富对象、类型安全、`try/except` 兜底、实现极简、契合
  现有 async 模型。
- **编程式钩子劣势**：改钩子 = 改代码 = 重部署、只能 Python、终端用户无法在不动源码下自定义。

> 当前阶段编程式钩子更合适：本项目定位是 Python agent 框架，钩子的主要消费者是**开发者**；
> 且 PreToolUse 改写参数、PostToolUse 改写结果、Stop 注入续跑这些场景**依赖进程内活对象**，
> JSON 边界反而会限制能力。配置驱动加载器是面向**终端用户免改源码扩展**时才值得加的独立特性。

### 4.3 配置驱动加载器的演进指引（不互斥，叠加为适配器层）

**关键判断**：外部加载器应作为**编程式钩子之上的一个适配器实现**，而非替换。本次的
`HookPipeline` 接口正是那一层的落点 —— 把每个外部钩子适配成实现对应 Protocol 的回调对象，
与编程式钩子在同一管线里并存、同序折叠。

建议落地步骤：

1. **配置解析**：在 `config.py` 增加 `[[hooks]]` 表解析（事件名 + matcher + type + 具体
   字段），遵循现有 defaults→toml→env→flag 层级；事件名复用 `HookEvent`。
2. **适配器**：为每种 type 写一个适配器类，实现对应 Protocol（如 `StopHook.on_stop`）：
   - `command`：`asyncio.create_subprocess_exec`，stdin 喂 `HookContext` 的 JSON 投影，
     stdout 解析 JSON / 退出码 2→block；**必须**有单次非堆叠超时 + kill-await（对齐
     `plan-review-standards` 与 `compression_summary` 的超时纪律）。
   - `http`：异步 POST `HookContext` JSON，解析响应为 `HookOutcome`。
   - `prompt` / `agent`：经 `GatedProvider` / `dispatch_agent` seam 复用现有并发预算。
3. **上下文投影**：定义 `HookContext` → JSON 的稳定投影（`messages` 可能要裁剪/摘要，避免把
   整段历史塞进每个外部钩子）。这是 JSON 边界的主要设计点，也是和进程内"活对象"路线最大的
   能力分界。
4. **装配**：构造期把解析出的适配器 `append` 进 `HookPipeline` 的对应钩子列表 —— 编程式钩子
   与外部钩子在同一管线里并存、同序折叠，循环侧（`react.py`）无需任何改动。
5. **安全**：外部 `command`/`http` 钩子是强力能力，需信任边界（仅项目内 `agent.toml`、不读
   不可信远程配置）、超时、并发上限，并把失败降级为"放行 + 记日志"而非 sink run。

> 即：当前已有编程式 `HookPipeline` + 生命周期 seam（已完成），配置驱动加载器只需再写
> 「解析 + 适配器 + 投影」三块，把外部进程包装成同一接口的回调。**契约不变，循环不动，
> 风险可控。**

---

## 结论

本项目在**循环骨架、契约设计、并发模型、工具/权限/memory 架构**上与 Claude Code
**同构且取向正确**——这是难的部分，已经做对了。真正的差距集中在**上下文质量**
（LLM 摘要 compaction、CLAUDE.md/git 注入）和**会话连续性**（resume）。建议从 P0 三项
入手：改动小、风险低、最快拉近与参考项目的体感差距，且都不触碰高风险的契约变更。

> **进展更新（2026-06）**：
> - **P0 三项全部完成**：LLM 双轨 compaction（§3.2-1）、CLAUDE.md 注入（§3.2-3）、
>   git status 注入（§3.5-P0-2）均已实现，`compression_summary.py` + `context.py` 是
>   主要新文件。
> - **会话连续性（resume）已完成**（§3.2-2 / §3.4-3）——包括 `transcript.py` 持久化、
>   `--resume`/`--continue`/`--fork-session`/`sessions list`、`chat` 跨轮记忆，以及
>   transcript 内建 compact-boundary（fold 写边界 + 摘要、`>5MB` 截断、`persist_compaction_boundary`
>   开关）。
> - **tool result 预算裁剪 + tool-use summary**（§3.2-10）**已完成**：预算裁剪由
>   `MaxOutputPostHook` pointer 模式承担；参考的 tool-use summary（异步 Haiku **UI 进度标签**，
>   非上下文缩减）由新增 `tool_use_summary.py` 忠实复刻（只进 UI + `runs/*.jsonl`，不进
>   API/transcript）。同时订正了原条目「降低上下文占用」对 tool-use summary 的误标。
> - **更丰富的 hook 类型（§3.2-5）已完成**：补齐 UserPromptSubmit/PostSampling/
>   Pre·PostCompact/可阻断 Stop 五类**编程式**生命周期钩子（`hooks.py` + `react.py`，
>   `tests/test_lifecycle_hooks.py` 11 例，全套 467 绿）。**剩余**：参考的 settings.json
>   **外部配置钩子加载器**（command/http/prompt/agent）尚未实现——其差别与演进指引见 **§4**。
> - **skill / slash command 系统（§3.2-6）已完成**：新增 `agent_core/skills/`（Markdown
>   `SKILL.md` 加载器 + bundled 内置技能 + 用户/项目技能目录）、chat 的 `/command` 派发、
>   可被模型调用的 `skill` 工具（inline/fork 两种上下文），57 个新测试，全套 523 绿。
> - **仍未实现**：流式工具执行（§3.2-7）、413 分级恢复完整链（§3.2-8）、fallback 模型
>   切换（§3.2-9）、规则驱动权限（§3.2-4）、配置驱动外部钩子加载器（§4.3）。
