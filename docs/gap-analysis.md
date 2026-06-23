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
| **扩展层** | subagent（`AgentTool`）、skill（`skills/bundled/` 17 个 + `loadSkillsDir` 用户技能）、hook（pre/post tool、stop、postSampling、userPromptSubmit）、MCP（完整 client+transport+oauth）、plugin marketplace、Task/swarm 多 agent | subagent（`dispatch_agent`）、team（多 agent 协作）、hook（pre/post tool）、MCP（client/adapter/config） | subagent/MCP/hook 对齐；**无 skill 系统、无 plugin、无 slash command、stop/postSampling hook 缺失** |

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

1. **LLM 驱动的 compaction**：参考用 forked agent 调模型生成 `<analysis>+<summary>`
   摘要；本项目是纯字符串截断（snip/omit）。**这是上下文质量的最大差距**——长任务里
   截断会丢语义，摘要不会。
2. ~~**会话持久化 / resume**：参考有 session 文件 + `/resume` + 历史。本项目只有
   `runs/*.jsonl` 单向日志，不能续接会话。~~ **✅ 已实现**：新增 `transcript.py`——
   按 `session_id` 命名的追加式 Message round-trip JSONL（`uuid`/`parent_uuid` 消息树，
   含 thinking_blocks/tool_call 关联保真），CLI `--resume`/`--continue`/`--fork-session`
   + `sessions list`，`chat` 跨轮记忆，跨项目定位。**并已对标参考的 transcript 内建
   compact-boundary**：一次 fold 把边界 + 摘要写盘，`--resume` 直接加载压缩后状态（含
   `>5MB` fd 级字节截断 + 边界前元数据抢救扫描）；可经 `persist_compaction_boundary`
   开关回退到忠实全量历史。`runs/*.jsonl` 事件日志保持不变、各司其职。
3. **项目指令注入（CLAUDE.md）**：参考把 CLAUDE.md / memory files 作为 `userContext`
   每轮注入。本项目有 memory 系统但**不读取/注入项目级 CLAUDE.md**。
4. **规则驱动权限 + bash 分类器**：参考能对单条命令做细粒度 allow/deny/ask（按命令
   前缀、路径、危险模式）。本项目粒度到"工具级"，无法表达"允许 `git status` 但 ask
   `git push`"。
5. **更丰富的 hook 类型**：参考有 stop hook（可阻断/续跑）、postSampling hook、
   userPromptSubmit hook、preCompact hook。本项目只有 pre/post tool。
6. **skill / slash command 系统**：参考有 bundled skills + 用户技能目录 + slash
   command。本项目无。
7. **流式工具执行**：参考边流边跑工具；本项目是整轮 LLM 结果出齐后再 `execute_many`。
8. **413 / max_output_tokens 的分级恢复**：本项目有 `LLMContextTooLongError` →
   reactive_compact 一次；参考有 collapse-drain → reactive → 升档重试 → 多轮 recovery
   的完整链。
9. **fallback 模型切换**与 tombstone 清理。
10. **tool result 预算裁剪 + tool-use summary**（Haiku 异步摘要降低上下文占用）。

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

1. **CLAUDE.md 注入**：在 `react.py` 组装初始 messages 时，读取 workspace 的
   `CLAUDE.md` 注入为 system/pinned 块（仿 `context.ts:getUserContext`）。改动局部、
   收益立竿见影。
2. **git status 上下文块**：仿 `context.ts:getGitStatus`，run 开始时注入一次性 git
   快照。小、独立。
3. **LLM 摘要式 compaction（双轨）**：默认走模型摘要、降级到现有字符串截断。这是缩小
   与参考差距的**最高杠杆**项。

**P1（中收益，中风险）**

4. **stop hook**（可阻断/可续跑）——让"任务未完成时强制继续/收尾"成为可能。
5. **413 / max_output_tokens 分级恢复**——在现有 `reactive_compact` 上补
   max_output_tokens 续跑。
6. **tool result 预算裁剪**——超大工具输出按 per-message 预算截断（已有
   MaxOutputPostHook，可扩展）。

**P2（架构投入，按需）**

7. ~~会话持久化 / resume（需契约扩展，谨慎）。~~ **✅ 已完成**（含 transcript 内建
   compact-boundary，见 §3.2-2、§3.4-3）。
8. skill / slash command 系统。
9. 流式工具执行（收益/复杂度比对中型 agent 偏低）。
10. 规则驱动权限（从命令前缀规则起步）。

---

## 结论

本项目在**循环骨架、契约设计、并发模型、工具/权限/memory 架构**上与 Claude Code
**同构且取向正确**——这是难的部分，已经做对了。真正的差距集中在**上下文质量**
（LLM 摘要 compaction、CLAUDE.md/git 注入）和**会话连续性**（resume）。建议从 P0 三项
入手：改动小、风险低、最快拉近与参考项目的体感差距，且都不触碰高风险的契约变更。

> **进展更新（2026-06）**：**会话连续性（resume）已完成**——包括 `transcript.py`
> 持久化、`--resume`/`--continue`/`--fork-session`/`sessions list`、`chat` 跨轮记忆，
> 以及对标参考的 **transcript 内建 compact-boundary**（fold 写边界 + 摘要、`>5MB` 字节
> 截断、`persist_compaction_boundary` 开关回退）。详见 §2 表「session / transcript」行、
> §3.2-2、§3.4-3、§3.5-7。其余 §3.2/§3.5 各项状态未在本次更新范围内，仍按原文。
