# P0 目标设计文档

> 依据 `docs/gap-analysis.md` §3.5 的 P0 三项。本文档只描述**目标设计**，不修改任何
> 代码实现。三项：
>
> 1. CLAUDE.md 注入
> 2. git status 上下文块
> 3. LLM 摘要式 compaction（双轨）
>
> 设计约束（遵循 `CLAUDE.md`）：
> - `import agent_core` 不得依赖可选重依赖。
> - 公共执行 API 一律 `async def` 干净命名，阻塞 IO 只作为 `_xxx_sync` 内部实现。
> - 跨层契约 `Message/ToolCall/ToolResult/LLMResult` 谨慎变更并更新测试。
> - 上下文管理不得隐藏重要近期证据；compaction 失败不得让已完成的 run 崩。
> - `NullUI` 默认静默、非交互测试稳定。

---

## 公共设计原则（三项共用）

- **一次性上下文注入点统一**：CLAUDE.md 与 git status 都是"run 开始时构造、整轮不变"
  的上下文，统一在 `ReActAgent.run()` 组装初始 `messages` 的位置注入，紧贴现有
  `_recall()` 的注入方式（`react.py:346` 把 memory 块 `insert(1, ...)`）。
- **pinned / 不被压缩**：注入块打 `metadata` 标记，使 compaction 的
  `_context_collapse` 与未来 LLM 摘要都把它当作 system pinned 保留（参考
  `compression.py:96` 已用 `metadata.get("compressed")` 区分 system 块）。
- **可降级**：任一注入源不可用（无 CLAUDE.md / 非 git 仓库 / 无 API key）时静默跳过，
  绝不抛错。

---

## 1. CLAUDE.md 注入

### 目标行为

- run 启动时，自上而下发现并读取项目指令文件，作为一段 pinned system 上下文注入到
  `messages`，位置在主 system prompt 之后、memory recall 块与 user task 之前。
- 发现规则（对齐 Claude Code `context.ts:getUserContext` → `claudemd.ts`）：
  - 从 `session.workspace` 起向上逐级父目录查找 `CLAUDE.md`，就近到根；多个文件按
    "从根到 workspace"顺序拼接（越靠近 workspace 越靠后、优先级越高）。
  - 额外包含 `~/.claude/CLAUDE.md`（用户级，若存在）。
- 开关：环境变量 `AGENT_DISABLE_CLAUDE_MD` 为真值时整体关闭（对齐参考的
  `CLAUDE_CODE_DISABLE_CLAUDE_MDS`）。
- 注入块带前缀说明文字（"以下是项目指令，需遵守"）+ 文件来源路径，便于模型分辨。
- 文件过大时按字符上限截断并标注（默认 32k 字符），避免单文件吃光上下文。

### 现有代码入口

- `agent_core/react.py:210-213` —— `run()` 构造初始 `messages`（system + user）。
- `agent_core/react.py:346-357` —— `_recall()`，已有的 `messages.insert(1, ...)` 注入
  范式，CLAUDE.md 注入紧邻其后复用同款思路。
- `agent_core/session.py:79` —— `SessionContext.workspace`，发现起点。
- `agent_core/config.py:104` —— `resolve_config()`，开关/上限的解析落点。

### 需要修改的文件

| 文件 | 改动 |
|---|---|
| `agent_core/context.py` *(新增)* | 项目指令发现/读取/拼接（见下） |
| `agent_core/react.py` | `run()` 中调用 `await build_project_instructions(...)` 并注入；新增私有 `_inject_project_context()` 统一处理 CLAUDE.md + git |
| `agent_core/config.py` | `ReActConfig` 增 `project_instructions: bool = True`、`claudemd_max_chars: int = 32000`；env `AGENT_DISABLE_CLAUDE_MD` 解析 |
| `tests/test_context.py` *(新增)* | 单测 |

### 新增模块设计：`agent_core/context.py`

```python
# 纯标准库，无重依赖；所有 IO 为内部 _sync，公共入口 async。
async def build_project_instructions(
    workspace: Path,
    *, max_chars: int = 32000, include_user_home: bool = True,
) -> str | None:
    """发现并拼接 CLAUDE.md，返回注入文本；无内容返回 None。"""

def _discover_claude_md_sync(workspace: Path, include_user_home: bool) -> list[Path]:
    """workspace 向上逐级 + ~/.claude/CLAUDE.md，去重、保序（根→workspace）。"""

def _read_and_join_sync(paths: list[Path], max_chars: int) -> str | None:
    """读取、按来源加标题、整体超限时截断并标注。"""
```

- 读取走 `asyncio.to_thread(self._..._sync, ...)`，不阻塞事件循环。
- 返回文本由 `react.py` 包成 `Message("system", text, metadata={"pinned": "claudemd"})`。

### 数据流

```
run(task)
  └─ workspace = session.workspace
  └─ text = await build_project_instructions(workspace, max_chars=cfg.claudemd_max_chars)
        └─ _discover_claude_md_sync  → [/proj/CLAUDE.md, /proj/sub/CLAUDE.md, ~/.claude/CLAUDE.md]
        └─ to_thread(_read_and_join_sync) → "## 项目指令 (来源: ...)\n<content>"
  └─ if text: messages.insert(<after system, before recall/user>,
                Message("system", text, metadata={"pinned": "claudemd"}))
  └─ logger.write("project_instructions", {"sources": [...], "chars": n})
```

注入顺序（自顶向下）：`system_prompt` → `claudemd`(pinned) → `git_status`(pinned) →
`memory recall`(若有) → `user task`。即 CLAUDE.md/git 在 `_recall` 之前注入，保持
`_recall` 的 `insert(1,...)` 仍把 recall 放在 user 之前。

### 异常兜底

- 开关关闭 / 无文件 / 全部读失败 → 返回 `None`，不注入，run 照常。
- 单文件读失败（权限/编码）→ 跳过该文件，其余继续；日志记 `skipped` 与原因。
- 超大文件 → 截断到 `max_chars` 并追加 `\n...(truncated)`。
- 任何异常都被模块内吞掉并降级，绝不冒泡到 `run()`。

### 测试计划（`tests/test_context.py`）

- 无 CLAUDE.md → 返回 None，messages 不变。
- workspace 链上多个 CLAUDE.md → 顺序为"根→workspace"，内容都在。
- `AGENT_DISABLE_CLAUDE_MD=1` → 即便存在也不注入。
- 超大文件 → 被截断且带标注。
- 注入块 `metadata["pinned"]=="claudemd"`；经 `compression.compact` 后仍保留。
- 不可读文件 → 跳过且不抛错。

### 验收标准

- [ ] 存在 `CLAUDE.md` 时，`run()` 后 `messages` 含一条 pinned system 块且内容匹配。
- [ ] 多级/用户级文件按既定顺序拼接。
- [ ] 开关可关闭；关闭后无注入。
- [ ] 任何读失败都不影响 run 结果。
- [ ] compaction 后注入块不丢。
- [ ] `import agent_core` 仍无新重依赖。

---

## 2. git status 上下文块

### 目标行为

- run 启动时（仅一次）采集 git 快照，作为一段 pinned system 上下文注入，紧随 CLAUDE.md
  之后。对齐 Claude Code `context.ts:getGitStatus` 的字段：
  - 当前分支、主分支（用于 PR）、git user、`git status --short`、最近 5 条
    `git log --oneline`。
  - 文案首句声明"这是会话开始时的快照，整轮不更新"。
- 非 git 仓库 / git 不可用 → 跳过。
- `git status` 超 2000 字符时截断并提示用模型用 bash 自查（对齐参考的 `MAX_STATUS_CHARS`）。
- 开关：`ReActConfig.git_context: bool = True`；env `AGENT_DISABLE_GIT_CONTEXT`。

### 现有代码入口

- `agent_core/react.py:210-213` —— 与 CLAUDE.md 同一注入点，复用
  `_inject_project_context()`。
- `agent_core/session.py:79` —— `workspace` 作为 git 命令的 cwd。
- 项目已有 PowerShell/Bash 工具经验，但此处用 `asyncio.create_subprocess_exec` 直调
  `git`，不经工具层（这是上下文采集，不是模型可见的工具调用）。

### 需要修改的文件

| 文件 | 改动 |
|---|---|
| `agent_core/context.py` | 增 `build_git_status(workspace) -> str | None` |
| `agent_core/react.py` | `_inject_project_context()` 内追加 git 块注入 |
| `agent_core/config.py` | `ReActConfig.git_context: bool`；env 解析 |
| `tests/test_context.py` | git 相关单测（用临时 git 仓库 / monkeypatch git 调用） |

### 新增模块设计（并入 `agent_core/context.py`）

```python
async def build_git_status(
    workspace: Path, *, max_status_chars: int = 2000,
) -> str | None:
    """非 git 仓库或失败返回 None。"""

async def _git(workspace: Path, args: list[str]) -> str | None:
    """asyncio.create_subprocess_exec('git', '--no-optional-locks', *args, cwd=workspace);
    超时/非零退出返回 None。带整体超时（如 5s）。"""
```

- 并行采集 branch / main / status / log / user.name（`asyncio.gather`，互不依赖）。
- 输出格式与参考一致，便于模型沿用既有习惯（"Main branch (you will usually use this
  for PRs)" 等）。

### 数据流

```
_inject_project_context(messages):
  git_block = await build_git_status(workspace)   # None 即跳过
  if git_block:
     messages.insert(<after claudemd>, Message("system", git_block,
                       metadata={"pinned": "git_status"}))
     logger.write("git_status", {"chars": len(git_block)})
```

### 异常兜底

- 非 git / `git` 不在 PATH / 子进程超时 → `None`，跳过。
- 单条 git 命令失败 → 该字段留空（如 user.name 缺失则省略行），其余照常。
- 整体 `asyncio.wait_for` 超时保护，避免大仓库卡住 run 启动。
- 测试环境（`AGENT_ENV=test` 或注入 fake）默认跳过真实 git 调用，避免 CI 抖动
  （对齐参考 `context.ts:37` 的 `NODE_ENV==='test'` 短路）。

### 测试计划

- 临时 git 仓库（init + commit）→ 块含分支、log、status 字段。
- 非 git 目录 → 返回 None，不注入。
- `git status` 超限 → 截断 + 提示文案。
- git 不可用（monkeypatch `_git` 抛错/超时）→ None，不影响 run。
- 注入块 `metadata["pinned"]=="git_status"`，compaction 后保留。

### 验收标准

- [ ] git 仓库内 run，`messages` 含 git pinned 块且字段正确。
- [ ] 非 git / git 缺失 → 无块、无异常。
- [ ] 超长 status 被截断并提示。
- [ ] 块有 5s 量级超时保护。
- [ ] 测试默认不打真实 git，可显式开启。

---

## 3. LLM 摘要式 compaction（双轨）

> 这是 P0 中**杠杆最高、风险最高**的一项。核心难点：现有 `CompressionPipeline` 是
> **同步**的（`maybe_auto_compact` / `reactive_compact` 为 `def`，在 `react.py:251`
> 与 `react.py:270` 同步调用），且**不持有 provider**；LLM 摘要需要异步 + provider。

### 目标行为

- compaction 的"折叠历史前缀"环节支持两条轨：
  - **Track A（LLM 摘要，默认）**：把待折叠前缀交给模型生成结构化摘要
    （`<analysis>`/`<summary>` 风格，对齐 Claude Code `services/compact/prompt.ts`），
    保留语义而非粗暴截断。
  - **Track B（字符串截断，降级）**：现有
    `_snip`/`_microcompact`/`_context_collapse` 行为，作为无 API key / 摘要失败 /
    deterministic 测试时的兜底。
- 选轨规则：当注入了可用的 `summarizer`（有真实 provider）且 `use_llm_summary=True` →
  Track A；否则 Track B。FakeProvider / 无 key → 自动 Track B（保证"无 key 也能跑"）。
- LLM 摘要只替换 `_context_collapse` 阶段的前缀压缩；`_snip`/`_microcompact`（按
  tool 结果体积裁剪）仍先跑，便宜且确定。
- pinned 块（CLAUDE.md / git / memory recall）与近期 N 条消息永不进摘要。
- 摘要走 forked、no-tools 调用（参考 `prompt.ts:NO_TOOLS_PREAMBLE`），且**不计入**
  memory extraction。

### 现有代码入口

- `agent_core/compression.py:33-56` —— `maybe_auto_compact` / `reactive_compact` /
  `compact`（三段同步管线）。
- `agent_core/compression.py:90-118` —— `_context_collapse`（当前用 `" | ".join`
  朴素拼接，**这是 Track A 要替换的环节**）。
- `agent_core/react.py:251-256` —— auto 调用点（循环每轮）。
- `agent_core/react.py:269-275` —— reactive 调用点（`except LLMContextTooLongError`）。
- `agent_core/providers/base.py:28` —— `LLMProvider.complete(...)`，摘要复用它
  （`tools=[]`、专用 system prompt）。
- `agent_core/react.py:115` —— `self.provider`（已 gated），summarizer 的闭包来源。

### 需要修改的文件

| 文件 | 改动 |
|---|---|
| `agent_core/compression.py` | `CompressionConfig` 增 `use_llm_summary`/`summary_max_tokens`/`summary_keep_recent`；新增**异步**入口 `auto_compact`/`reactive_compact_async`（或带 `summarizer` 参数）；`_context_collapse` 拆出 `_collapse_prefix`，按是否有 summarizer 走 A/B |
| `agent_core/compression_summary.py` *(新增)* | 摘要 prompt 模板 + `Summarizer` 类型 + 解析/降级逻辑 |
| `agent_core/react.py` | 构造 `summarizer` 闭包并注入；compaction 调用点改 `await`；摘要失败回退 Track B |
| `agent_core/config.py` | `resolve_compression_config()`（`[compression]` toml 表）；env 开关 |
| `agent_core/models.py` | （不改契约）摘要块复用 `metadata={"compressed":"llm_summary"}` |
| `tests/test_compression.py` | 扩充：双轨、降级、pinned 保留、异步路径 |

### 关键 seam 设计：注入式 summarizer（不让 compression 依赖 provider）

沿用项目既有依赖注入范式（`session.subagent_factory` 是注入闭包，
`react.py:128-134`），compression **不导入 provider**，而是接收一个异步回调：

```python
# compression_summary.py
Summarizer = Callable[[list[Message]], Awaitable[str]]   # 入: 待折叠前缀; 出: 摘要文本

SUMMARY_SYSTEM = "..."   # no-tools preamble + <analysis>/<summary> 指令(精简版)

def build_summarizer(provider, provider_config) -> Summarizer:
    async def summarize(prefix: list[Message]) -> str:
        convo = render_prefix(prefix)                 # 前缀渲染为单条 user 文本
        msgs = [Message("system", SUMMARY_SYSTEM), Message("user", convo)]
        cfg = {**provider_config, "max_tokens": cfg.summary_max_tokens}
        result = await provider.complete(msgs, [], cfg)   # tools=[], 无流
        return extract_summary(result.content)        # 剥离 <analysis>，取 <summary>
    return summarize
```

`CompressionPipeline` 侧改为**异步公共入口**（遵循 CLAUDE.md：公共执行 API 用干净
async 名）：

```python
class CompressionPipeline:
    async def auto_compact(self, messages, *, summarizer=None, on_stage=None): ...
    async def reactive_compact(self, messages, *, summarizer=None, on_stage=None): ...
    # 内部：_snip/_microcompact 仍同步；_collapse_prefix 异步（A 用 summarizer，B 朴素）
```

> 说明：现有同步 `maybe_auto_compact` 的纯字符串行为整体保留为 Track B 的实现（被
> `_collapse_prefix` 在无 summarizer 时调用），因此 deterministic 测试与无 key 路径
> 行为不变。这降低了"高风险重构"的回归面。

### 数据流

```
react.py 循环每轮:
  messages = await compression.auto_compact(
                 messages, summarizer=self._summarizer, on_stage=reporter)
        ├─ _snip(messages)            # 同步, 按 tool 结果体积
        ├─ _microcompact(messages)    # 同步
        └─ _collapse_prefix(messages):
              split → [pinned] + [prefix(旧)] + [recent N 条]
              if summarizer and use_llm_summary:
                  try:  summary = await summarizer(prefix)        # Track A
                        block = Message("system", summary,
                                  metadata={"compressed":"llm_summary",
                                            "messages_collapsed": len(prefix)})
                  except Exception:  block = naive_collapse(prefix)  # → Track B 降级
              else:
                  block = naive_collapse(prefix)                  # Track B
              return [*pinned, block, *recent]

reactive(except LLMContextTooLongError):
  messages = await compression.reactive_compact(messages, summarizer=self._summarizer)
        # aggressive: 更小的 keep_recent / 更狠的前缀切分; 同样 A→B 降级
```

阈值/触发不变：`auto` 在 `char_count >= max_context_chars * auto_threshold_ratio` 时
触发（`compression.py:36`）；`reactive` 由 `LLMContextTooLongError` 触发
（`react.py:269`）。

### 异常兜底

- **summarizer 抛错 / 超时 / 返回空** → 立即降级 Track B（朴素折叠），该轮 compaction
  仍成功，run 不中断。日志记 `summary_fallback` + 原因。
- **摘要调用自身可能 413**：`render_prefix` 先经 `_snip`/`_microcompact` 已缩小；摘要
  调用对前缀再做硬字符上限保护（`summary_input_max_chars`），超限先朴素截断再喂模型。
- **reactive 路径尤须稳**：它已在 `except LLMContextTooLongError` 内，二次失败必须降级
  到 Track B 并继续，绝不能形成 413→摘要→413 的死循环（对齐参考的
  `hasAttemptedReactiveCompact` 单发保护）——本设计用"reactive 内摘要至多尝试一次，失败
  即 Track B"实现同等效果。
- **费用/递归**：摘要走 `GatedProvider`（已有并发/限流预算）；摘要调用 `tools=[]`、
  `stream=None`，不触发工具、不参与 memory extraction。
- **无 provider / FakeProvider**：`build_summarizer` 在无真实 key 时返回 `None`，全程
  Track B，保证 `--provider fake` 与离线测试 deterministic。

### 测试计划（`tests/test_compression.py` 扩充）

- **Track B 不变性**：不传 summarizer → 输出与现有同步实现逐字节一致（回归保护）。
- **Track A 正常**：注入 stub summarizer（返回固定串）→ 折叠块为该串、`metadata
  ["compressed"]=="llm_summary"`、`messages_collapsed` 正确。
- **降级**：summarizer 抛异常 / 超时 / 返回空 → 落 Track B，compaction 仍返回有效
  messages，run 不抛。
- **pinned 保留**：CLAUDE.md/git/recall 块在 A、B 两轨均不进摘要。
- **near-recent 保留**：最近 N 条原样保留，未被折叠。
- **reactive 二次失败**：stub summarizer 连续抛错 → 不死循环，单发后降级。
- **async 化回归**：`react.py` 全流程（fake provider）在 auto/reactive 触发下跑通；
  `test_react.py` / `test_limits.py` 绿。
- **选轨**：FakeProvider → summarizer 为 None → 始终 Track B。

### 验收标准

- [ ] 有真实 provider 且 `use_llm_summary=True` 时，折叠前缀为 LLM 摘要块。
- [ ] 无 key / FakeProvider / 关闭开关 → 行为与现状逐字节一致（Track B）。
- [ ] summarizer 任意失败都降级且 run 不崩；reactive 路径无 413 死循环。
- [ ] pinned 与近期 N 条永不进摘要。
- [ ] 摘要调用 `tools=[]`、不参与 memory extraction、受 `GatedProvider` 预算约束。
- [ ] compaction 公共入口为 async 干净命名；阻塞仅在内部。
- [ ] 全量 `pytest` 绿；`import agent_core` 无新增重依赖。

---

## 跨项落地顺序与回归面

1. **先做 1 + 2（注入类，低风险）**：只在 `run()` 增注入点 + 新增 `context.py`，不改
   契约、不改循环控制流。`test_context.py` 覆盖。
2. **再做 3（双轨 compaction，高风险）**：分两步——
   - 3a 先把 compaction 公共入口 async 化但仍只跑 Track B（纯重构，行为不变，靠
     "Track B 逐字节一致"测试护栏）。
   - 3b 再接入 summarizer 注入与 Track A + 降级。
3. 每步跑：`pytest tests/test_context.py tests/test_compression.py tests/test_react.py
   tests/test_limits.py`，再跑全量。

## 配置汇总（新增项，集中在 `agent.toml` 与 env）

```toml
[context]
project_instructions = true      # AGENT_DISABLE_CLAUDE_MD
git_context = true               # AGENT_DISABLE_GIT_CONTEXT
claudemd_max_chars = 32000

[compression]
use_llm_summary = true
summary_max_tokens = 2048
summary_keep_recent = 8
summary_input_max_chars = 16000
```

> 以上均不改 `Message/ToolCall/ToolResult/LLMResult` 契约，仅扩展 `metadata` 取值与
> 新增配置字段，符合 `CLAUDE.md` 的契约稳定性要求。
