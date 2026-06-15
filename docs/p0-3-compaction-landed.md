# P0-3 双轨 compaction —— 已落地记录 + 行号修正

> 配套文档：目标设计见 `docs/p0-target-design.md` §3；缺口来源见 `docs/gap-analysis.md`。
> 本文档记录 **§3 已实现的真实状态**：落地清单、与原设计的差异（含行号修正）、修正后的
> 数据流，以及**仍未做/延期**的项，作为后续逐步实现的路标。
>
> 约定：行号为本文写作时的实际代码位置，可能随后续提交漂移；以符号名定位更稳。

---

## 0. 状态总览

| 项 | 状态 |
|---|---|
| compaction 公共入口 async 化（删同步 twins） | ✅ 已落地 |
| 注入式 summarizer seam（不让 pipeline 依赖 provider） | ✅ 已落地 |
| `[compression]` 配置表 + env + CLI 接线 | ✅ 已落地 |
| Track A（LLM 摘要）替换前缀折叠 + 失败降级 Track B | ✅ 已落地 |
| reactive 单发护栏（无 413→摘要→413 死循环） | ✅ 已落地（结构性保证） |
| Track B 逐字节回归护栏 | ✅ 已落地（测试） |
| 全量测试 | ✅ 283 passed |

延期项见 §6。

---

## 1. 已落地清单（文件 → 符号 → 行为）

### `agent_core/compression.py`

| 符号 | 行 | 说明 |
|---|---|---|
| `Summarizer` 类型别名 | 16 | `Callable[[list[Message]], Awaitable[str]]`。**定义在此处**（不在 `compression_summary`），以打破"pipeline ↔ seam"循环依赖。 |
| `_COLLAPSE_MARKERS` | 20 | `frozenset({"context_collapse", "llm_summary"})`。两轨的折叠块都可被下一轮重新折叠，避免堆积。 |
| `CompressionConfig` | 32 | 新增 4 字段：`use_llm_summary=True` / `summary_max_tokens=2048` / `summary_keep_recent=8` / `summary_input_max_chars=16000`。原 4 字段不变。 |
| `async def auto_compact(...)` | 51 | 取代旧同步 `maybe_auto_compact`；阈值门仍在内部；签名 `(messages, *, summarizer=None, on_stage=None)`。 |
| `async def reactive_compact(...)` | 70 | 取代旧同步 `reactive_compact`；总是压缩；同签名。 |
| `async def _run_stages(...)` | 80 | 取代旧 `compact()`；显式按序跑 snip→microcompact→collapse；只有 collapse 是 `await`。 |
| `def _snip(...)` | 107 | **同步纯函数，行为不变**（tool 消息头尾截断）。 |
| `def _microcompact(...)` | 123 | **同步纯函数，行为不变**（非 pinned 超长截断）。 |
| `async def _context_collapse(...)` | 146 | 改 async，接 `summarizer`；切分 `[system/pinned] + [prefix] + [recent N]`，调 `_collapse_prefix`。 |
| `async def _collapse_prefix(...)` | 178 | **轨道分叉点**：有 summarizer 走 Track A（失败就地降级 naive），无则 Track B；返回 `(block, note)`。 |
| `@staticmethod _collapse_prefix_naive(...)` | 205 | Track B 朴素折叠（`" \| ".join(...[:160])`），与重构前**逐字节一致**。 |

要点：
- **同步/异步分工**：公共入口 + collapse 为 async；`_snip`/`_microcompact`/`_collapse_prefix_naive`/`_char_count` 为同步纯 helper（CPU 字符串操作，无需 `to_thread`）。
- **选轨**：`use_llm = summarizer is not None and config.use_llm_summary`（`_context_collapse` 内）。
- **recent 窗口**：Track A 用 `summary_keep_recent`，Track B 用 `collapsed_keep_recent`（保证 Track B 不变）；两者 `keep = max(4, base // (2 if aggressive else 1))`。
- **观测**：CompressionEvent.detail 现带轨道注记——`collapsed N msgs (llm_summary)` / `(track_b)` / `(summary_fallback: <ExcName>)`，经 `react.py` 写入 `runs/*.jsonl` 的 `compression` 事件。折叠块 metadata：Track A=`{"compressed":"llm_summary","messages_collapsed":N}`，Track B=`{"compressed":"context_collapse",...}`。

### `agent_core/compression_summary.py` *(新增)*

| 符号 | 行 | 说明 |
|---|---|---|
| `SUMMARY_SYSTEM` | 25 | no-tools 前导 + `<analysis>`/`<summary>` 九段指令（借 Open-ClaudeCode `prompt.ts`，精简）。 |
| `render_prefix(prefix, max_chars)` | 45 | 前缀渲染成单条文本，超 `summary_input_max_chars` 头尾硬截断（防摘要调用自身 413）。 |
| `extract_summary(text)` | 64 | 剥 `<analysis>`、取 `<summary>`；无标签时回退整段（容错）。 |
| `build_summarizer(provider, provider_config, config)` | 82 | 返回 `Summarizer \| None`。`use_llm_summary=False` 或 `FakeProvider`（含被 `GatedProvider` 包裹）→ `None`；否则闭包用**gated** provider 发 `tools=[]`、`stream=False`、关 thinking、`max_tokens=summary_max_tokens` 的调用。 |

> 注：`Summarizer` 注入的回调返回**已抽取的最终摘要文本**（`extract_summary` 在 `build_summarizer` 闭包内完成），pipeline 侧只负责包成 Message。单测里 stub 直接返回纯文本即可。

### `agent_core/config.py`

- `resolve_compression_config(config_file)` —— 266 行。读 `[compression]` 表（`overlay_dataclass` 按字段类型 coerce）+ `AGENT_DISABLE_LLM_SUMMARY`（真值强制 `use_llm_summary=False`，把整轮钉死在 Track B）。

### `agent_core/react.py`

- 12 行：`from agent_core.compression_summary import build_summarizer`。
- 133→137 行：`self.compression = CompressionPipeline(...)` 之后构造 `self._summarizer = build_summarizer(self.provider, self._provider_config(), self.config.compression)`（用 gated provider 共享预算）。
- 269 行：`await self.compression.auto_compact(messages, summarizer=self._summarizer, on_stage=...)`。
- 288 行：`await self.compression.reactive_compact(messages, summarizer=self._summarizer, on_stage=...)`。

### `agent_core/cli.py`

- 导入并在 `build_agent` 的 `ReActConfig(...)` 加 `compression=resolve_compression_config()`（150 行）——**补上原设计漏掉的 CLI 接线**。

### `agent.toml.example`

- 新增 `[compression]` 段，注释说明双轨与降级语义。

---

## 2. 与原设计文档（`p0-target-design.md` §3）的差异 / 行号修正

| 设计文档写的 | 实际落地 | 备注 |
|---|---|---|
| auto 调用点 `react.py:251` | **react.py:269** | 文档行号偏移；以符号 `auto_compact` 调用为准。 |
| reactive 调用点 `react.py:269` | **react.py:288**（`except LLMContextTooLongError:` 内） | 同上。 |
| `compression.py:33-56` 三段同步管线 | 已替换为 async（见 §1） | 旧 `maybe_auto_compact`/`compact` 已删除。 |
| `_context_collapse` 在 `compression.py:90-118` | **现 146 行**（且已 async） | 重构后行号变化。 |
| 摘要 prompt 模块名 `compression_summary.py` 内含 `Summarizer` 类型 | `Summarizer` **改放 `compression.py`** | 否则 `compression` ↔ `compression_summary` 循环导入。`compression_summary` 从 `compression` 导入它。 |
| 设计草图 `_collapse_prefix` 切分为"`[pinned]`+prefix+recent" | 实际保留**所有 system 消息**（非仅 pinned）不折叠 | **沿用现状语义**以保 Track B 逐字节一致；pinned（CLAUDE.md/git/recall）恰好都是 system，已被覆盖。 |
| 设计未提"prior 摘要块如何处理" | 用 `_COLLAPSE_MARKERS` 让 `context_collapse` **与** `llm_summary` 块都可被重新折叠 | 防多次压缩堆积多个摘要块；纯 Track B 历史不含 `llm_summary`，行为不受影响。 |
| 选轨：`build_summarizer` 在无真实 key 返回 None | 实现为 `isinstance(_unwrap(provider), FakeProvider)` 检测（`_unwrap` 剥 `GatedProvider`） | provider 在 agent 内已被 gate 包裹，需对内层判断。 |
| reactive 单发用类似 `hasAttemptedReactiveCompact` 标志 | **结构性保证**，无显式标志 | 摘要失败在 `_collapse_prefix` 内**就地降级 naive**，故 `reactive_compact` 永不因摘要抛错；`react.py` 的 `except` 内只重试一次 `complete`（未再包 try），二次 413 直接上抛——天然单发。 |
| 设计提到"摘要超时保护" | 暂未加显式 `asyncio.wait_for` | 当前靠 `_collapse_prefix` 的宽 `except Exception`（含 `TimeoutError`）兜底降级。如需硬超时，见 §6 延期项。 |

---

## 3. 修正后的真实数据流

```
react.py 循环每轮（react.py:269）:
  messages, events = await compression.auto_compact(
      messages, summarizer=self._summarizer, on_stage=reporter)
    └─ if char_count < max_context_chars * auto_threshold_ratio: return 不变
    └─ _run_stages(aggressive=False, summarizer):
         _snip(同步)  →  _microcompact(同步)  →  await _context_collapse(summarizer):
            use_llm = summarizer is not None and use_llm_summary
            split → [非折叠 system] + [prefix 旧] + [recent N]   # N: A 用 summary_keep_recent / B 用 collapsed_keep_recent
            if conversation <= keep+1: 不折叠
            block, note = await _collapse_prefix(prefix, summarizer if use_llm else None):
                A: summary = await summarizer(prefix)         # tools=[], stream=False, gated
                   空/异常 → naive + note="summary_fallback: <Exc>"
                   成功    → Message(system, "Earlier conversation summary: …",
                                     metadata={"compressed":"llm_summary","messages_collapsed":len(prefix)})
                B: naive + note="track_b"
            return [*system, block, *recent]

except LLMContextTooLongError（react.py:288）:
  messages, events = await compression.reactive_compact(
      messages, summarizer=self._summarizer, on_stage=reporter)   # aggressive=True，更小 keep；同样 A→B 降级
  # 之后只重试一次 complete；二次 413 直接上抛（单发）
```

`build_summarizer`（无真实 provider → None）保证 `--provider fake`/离线测试全程 Track B、deterministic。

---

## 4. 配置（已生效）

```toml
[compression]
max_context_chars = 24000
auto_threshold_ratio = 0.8
max_message_chars = 6000
collapsed_keep_recent = 8        # Track B 近期保留
use_llm_summary = true           # AGENT_DISABLE_LLM_SUMMARY=1 强制 Track B
summary_max_tokens = 2048
summary_keep_recent = 8          # Track A 近期保留
summary_input_max_chars = 16000  # 摘要输入硬上限
```

精度：`overlay_dataclass` 按字段声明类型 coerce；未知键忽略；`AGENT_DISABLE_LLM_SUMMARY` 真值覆盖。

---

## 5. 测试覆盖现状（`tests/test_compression.py`，18 例）

- Track B：阈值触发/未触发、reactive 更狠、保留 system、阶段上报、保留 pinned、detail、**逐字节快照**。
- Track A：正常折叠、`use_llm_summary=False` 退 B、异常降级、空串降级、pinned 不进摘要、重复压缩只留一个摘要块。
- seam：FakeProvider/被 gate 包裹 → None、禁用 → None、no-tools 有界调用、`extract_summary` 解析+回退。
- 间接：`tests/test_context.py::test_git_block_survives_compaction`、`tests/test_react.py` reactive 重试用例（FakeProvider→Track B）。

---

## 6. 未做 / 延期项（后续可逐步实现的路标）

> 以下**本轮未实现**，按优先级与依赖排列，供后续按需开工。每项标注切入文件与思路。

1. **摘要硬超时**（小）：在 `_collapse_prefix` 的 Track A 调用外包 `asyncio.wait_for(summarizer(prefix), timeout=...)`；为 `CompressionConfig` 加 `summary_timeout_seconds`，`resolve_compression_config` 解析。超时落 `except` 已有的降级路径。
2. **grouping 成对保护**（小-中）：`_context_collapse` 的 prefix/recent 切点可能切断 `assistant(tool_calls)` 与其 `role=="tool"`（带 `tool_call_id`）结果。可加一个确定性纯 helper，把切点向 round 边界吸附（参考 Open-ClaudeCode `grouping.ts` 按 assistant 分组思想）。Track B 现状未切此问题，Track A 摘要为自然语言、孤立 tool 结果不致命，故定为可选增强。
3. **前缀分块（map-reduce）摘要**（中）：超长前缀当前是"硬截断 + 单次摘要"。后续可分块摘要再合并；`summary_input_max_chars` 之上加 `summary_chunk_chars`。
4. **tool result 写盘 + preview 指针替换**（中-大）：对标 Open-ClaudeCode `toolResultStorage`（单条 >50K / 单消息聚合 >300K 移出 live context，留 2KB preview + 盘上指针 + 冻结决策保缓存）。本项目已有 `MaxOutputPostHook` 溢写 `runs/outputs`，可在其上演进为"上下文内 preview 引用"。
5. **cache-edit / time-based microcompact**（大，依赖外部能力）：需服务端 prompt-cache 编辑能力，当前 provider 抽象无此概念，**暂不纳入**。
6. **sessionMemoryCompact**（大）：复用持久化会话记忆替代 LLM 摘要。本项目 `agent_core/memory/` 是**跨会话事实抽取**，语义不同，若做需新设计，勿与现 memory 混用。
7. **max output tokens recovery（8K→64K 升档）**（中）：属"输出被截断"恢复，与上下文压缩正交，归到 provider/loop 另议。

---

## 7. 建议的后续切入顺序

1. 先做 §6.1（超时）+ §6.2（grouping 成对保护）——都是 `compression.py` 内小改、低风险、收益直接。
2. 再评估 §6.4（tool result 指针）——这是与 Open-ClaudeCode 差距最大、对长会话最有价值的一块，但牵涉 `hooks.py`/消息重建，单独立项。
3. §6.5/6.6/6.7 视真实需求与 provider 能力再定，不建议在没有明确场景时投入。
