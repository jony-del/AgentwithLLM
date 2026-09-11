# 核心 Contract（阶段 0 冻结）

本文档是 `SECOND_ROUND_INDEPENDENT_AUDIT.md` 路线图「阶段 0：冻结核心 contract」的落地物。
它汇总四个已冻结的核心 contract 的权威定义、schema 版本、不变量和所有权规则。
文档只描述**已冻结的现状**；任何变更必须先升 schema 版本并更新本文档。

钉住这些 contract 的测试集中在 `tests/test_contracts.py`。

## 版本总览

| Contract | 定义位置 | Schema 版本 |
|---|---|---|
| ExecutionScope | `agent_core/execution.py` | v1 |
| SessionDescriptor / DurableHead / SessionRuntime | `agent_core/session.py` | v1 |
| RecoveryState（turn journal 状态机） | `agent_core/tools/transaction.py` | journal schema v4（可读 v3, v4） |
| Message identity + compaction snapshot | `agent_core/models.py` / `agent_core/transcript.py` | transcript schema v4 |

## 1. ExecutionScope（执行范围）

一棵执行树（run loop → provider attempt → tool batch → hook/MCP/subagent）共享一个
`ExecutionScope`。不变量：

1. **`child()` 单调收紧**：deadline 取 min；`network` 不允许 `deny → allow`  widening；
   `workspace_writable` 不允许 `False → True` widening。
2. **同一棵树共享同一 `CancellationToken` 与 `ExecutionTaskRegistry`**（`child()` 不换这两个字段）。
3. **取消语义三层关系固定**：`cancel_probe`（外部异步信号，如 Esc）首次观测时折叠进 token；
   token 提供协作取消（`raise_if_cancelled` 抛 `asyncio.CancelledError`）；
   `asyncio.Task.cancel` 是结构性取消，仅限 executor 对 `safely_cancellable` 任务使用。
   run loop 只把「本 scope 的 token 取消」转换为 interrupted 结果，其余重抛。
4. `remaining_budget(requested) = min(requested, 剩余墙钟)`；`run_awaitable` 以该值为上界，
   超时抛 `TimeoutError`。
5. 谁创建谁 `close()`：`close()` 取消 token 并收割注册的任务与 cleanup 回调。

**已知旁路（阶段 3 的接入对象，本阶段不处理）**：httpx `ProviderConfig.timeout`、
MCP client 线程侧 future、prompt/agent hook 的裸 `wait_for`、`ProcessSupervisor._enforce_timeout`。

## 2. SessionDescriptor / DurableHead / SessionRuntime

- **SessionDescriptor**（frozen）：一个可恢复会话的稳定身份 = `session_id` + `workspace` +
  `transcript_path` + `project_id`。生命周期内不可变；使用方不得在使用时从当前 cwd 重新推导。
- **DurableHead**：三分离——`memory_head_id`（内存链头，可领先于磁盘）、`durable_head_id`
  （最后确认落盘的链头，写失败后不得推进）、`persistence_degraded`（sticky 降级标志，
  置位后 durable head 冻结并上报 `transcript_persistence_degraded` 事件）。
- **SessionRuntime**：一个活跃会话的全部可变状态与持有资源。resume = 构造新 runtime →
  关闭旧 runtime → 原子替换 agent 指针（`ReActAgent.resume_loaded_session`）。
  `ReActAgent` 的 `session_id`/`session`/`transcript`/`permissions`/`process_supervisor`
  是转发 `runtime.*` 的派生只读 property，单一数据源为 SessionRuntime。
  以下 session 级状态**一律不跨 resume 存活**：session 权限规则与 allow-list、todo、
  plan state、approved workflow digests、read-file state、scheduler jobs、session 计数器。

## 3. RecoveryState（turn journal 状态机）

`agent_core/tools/transaction.py` 的 `RecoveryState` 枚举是 journal `state` 字段的唯一权威：

- **写入端强制**：`record()` / `record_telemetry()` 校验 state 必须是枚举成员，
  非法值抛 `JournalWriteError`——journal 里只可能出现 contract 内的状态。
- **读取端兼容**：`_load_verified` 在 checksum 校验通过后才按 `LEGACY_STATE_ALIASES`
  归一化（v3 的 `external_outcome` → `external_outcome_committed`），磁盘字节不变。
- **终态集合** `TERMINAL_RECOVERY_STATES`：`history_persisted` / `rolled_back` /
  `journal_closed` / `indeterminate_external_effect` / `external_effect_history_missing`
  （后两个为 v3 遗留终态，只读不写）。终态 journal 被恢复扫描跳过。
- **所有权**：每条记录携带 `owner{project_id, session_id, run_id, workspace, nonce}` +
  sha256 checksum 链；foreign journal 只能报告为 `foreign`，不得被当前 session 消耗或终结。
- **外部效应安全**：含未决 `external_intent` 的 journal 绝不重放、绝不终结，
  报告 `IndeterminateExternalEffect` 等待人工对账。
- **提交顺序（journal-first）**：turn 收尾按 `history_ready`(最终 payload) →
  transcript `append_tool_round` → `history_persisted` 推进。journal payload 写失败时
  **必须跳过 transcript 写入**并保持 journal 未决——journal 绝不落后于 transcript。
  transcript 的 round checksum 是幂等提交点；`history_persisted` 仅是终态标记，
  它写失败时 journal 保持未决，恢复路径幂等回放，不产生重复行。

## 4. Message identity 与 compaction snapshot

**Message 四重身份**（`agent_core/models.py`）：

| 字段 | 语义 |
|---|---|
| `uuid` | 链上实例身份；压缩派生时保持不变；仅在载入缺失时重新生成 |
| `origin_id` | 跨压缩的逻辑身份，指向最初版本（缺省 = `uuid`） |
| `version` | 内容版本；`_evolve_message` 每次派生 +1 |
| `round_id` | provider 工具轮次身份（带 `tool_calls` 的 assistant 消息缺省 = `uuid`） |

Provider 序列化的 tool_use/tool_result 配对只依赖 `metadata.tool_calls` / `tool_call_id`；
上述身份字段纯属持久化/恢复层，两个面互不影响。

**Compaction snapshot**（transcript schema v4）：

- `compaction_snapshot` 是**唯一权威**的 compaction 边界记录：完整有序的 post-fold 链
  （parent 重链到新 summary 根）+ `source_head` + sha256 checksum。载入时最后一个有效
  snapshot 取代其之前的全部内容（last boundary wins）；transcript 本体 append-only，
  从不重写已写入内容。
- `relink` 记录为 **legacy 只读兼容**：读路径仍按 last-wins 应用，但无生产写入方，
  schema v5 时移除读支持。
- `_commit_compaction_boundary` 的 fold 检测用 `(uuid, version)` 差集；真正的
  「snip/microcompact 不写边界」守卫是「差集中没有 summary 即不视为 fold」。

## 5. resume / fork / 跨项目 continuation 产品契约

- `--resume` / `/resume`：**仅同项目**。跨项目命中一律拒绝并提示
  `--fork-session` / `/branch`；不做任何形式的静默续接或部分导入。
- `--continue`：仅当前项目最新 session。
- `--fork-session`：**唯一跨项目通道**。只携带 `fork_chain` 克隆的消息链
  （全部换新 uuid + 重链 parent），不携带权限、todo、plan 或任何其他运行时状态。
- 进程内 `/resume` 与进程间 `--resume` 都走同一个 `resume_loaded_session` 语义：
  整体替换 `SessionRuntime`，旧 runtime 的 session 级状态全部失效。

## 阶段外事项（明确不属于本 contract）

以下属于路线图的后续阶段，本文档不作规定：TOFU trust matrix、patch parser 单一化、
plugin capability manifest（阶段 1）；上述 ExecutionScope
旁路的接入（阶段 3）；prompt ingress 统一（阶段 4）；journal/transcript retention 接线、
MCP 版本矩阵与 CI 门禁（阶段 5）。

阶段 2 的两项已落地：external outcome/history 原子提交（journal-first 提交顺序不变量
见 §3）；ReActAgent 平行指针重构为 SessionRuntime 单一数据源（agent 侧同名指针均为
runtime 派生 property，见 §2）。
