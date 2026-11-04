# 超大规模代码仓库能力改造实施计划

> 交接文档：独立 Coding Agent 可仅依据本文档和当前仓库执行后续开发。
> 本文档只描述实施计划；截至本文档生成时，不包含 S1 及之后阶段的新代码修改。

## 1. 背景、目标与已确定决策

本项目是 Python 实现的 Coding Agent，已有本地代码检索、代码上下文读取、LSP 证据、代码编辑、快照、检查点和验证副本能力。目标不是把整个仓库塞进模型，而是在有限上下文、延迟、内存和工具调用预算下，可靠完成：

自然语言需求理解 → 分阶段检索 → 精确定位文件/符号/编辑范围 → 方案规划 → 局部或跨文件修改 → 静态检查和测试验证。

目标规模包括：

- 场景 A：约 100 万 LOC；
- 场景 B：约 1000 万 LOC；
- 高文件数、Monorepo、多模块、跨语言、复杂依赖和频繁变更。

此前已确定的技术原则：

1. 渐进式改造，优先复用现有 SQLite、快照、版本和工具调用管线。
2. 先解决可测量的瓶颈，不先引入向量数据库、物理分片或完整重写。
3. 检索成功、定位成功、补丁成功、测试通过和任务解决必须分别计量。
4. 不把未经实测的百万/千万行能力表述为已实现；真实大仓库和真实模型验收需要外部输入。
5. 所有自动检索、LSP、验证和编辑操作继续经过权限、锁、预算、取消、日志和沙箱管线。
6. 不执行用户未授权的全仓写入、依赖安装、网络下载或破坏性操作。

当前基线：分支 `main`，HEAD 为 `a946482c7dc57f8ace224b2297f70df3768c53ae`，工作树应保持干净。真实超过 20,000 文件的仓库和模型配置尚未提供，因此先完成离线工具、合成基准和代码改造。

## 2. 当前实现与代码定位

### 2.1 代码智能核心

- `agent_core/codeintel/models.py`：`FileVersion`、`CodeHit`、`SearchCoverage`、`SearchPage`、`SearchRequest`、`ChangeSet`、`symbol_reference` 等协议模型。当前 schema 主要为 1；游标为 generation-bound。
- `agent_core/codeintel/service.py`：`CodeIntelligenceService`。负责本地 SQLite 索引、全文/三元组检索、符号预筛选、文件版本读取、分页、预算、取消、并发准入、发布和关闭。
- `agent_core/codeintel/store.py`：`CodeStore`。维护 `files`、`symbols`、`edges`、`pending`、`events`、`scan_dirs`、`module_roots` 等表；WAL、短连接超时、线程本地连接和批量写入已存在。
- `agent_core/codeintel/catalog.py`：`FileCatalog`。用持久目录队列及 `scandir` 增量发现文件，处理删除和策略排除。
- `agent_core/codeintel/indexer.py`：`update_pending()`。按文件读取、哈希、静态解析并原子提交 facts；支持 audit 未变化快速路径和软截止时间。
- `agent_core/codeintel/changes.py`：`record_commit()`、`replay_commits()`。目前事件表仍保存单条 JSON paths，按事件原子失效；超大单事件在短预算下可能长期无法推进。
- `agent_core/codeintel/watcher.py`：`WatchChanges`，批量合并变更、版本 token、失败重试、目录/overflow 重扫。
- `agent_core/codeintel/backends.py`：Python AST 结构解析和 manifest 静态依赖解析；不执行项目代码。
- `agent_core/codeintel/modules.py`：包名、模块解析、边界更新和最近 manifest 选择；尚未形成完整 workspace/local-package 图。
- `agent_core/codeintel/candidates.py`：`validate_candidate()`，按 worktree/path/hash、精确锚点及可选 Python symbol ID 校验候选；当前只证明机械锚点有效，不证明语义相关性。
- `agent_core/codeintel/budget.py`：`QueryBudget`，统一 deadline、取消、文件/字节/候选/边/输出上限。
- `agent_core/codeintel/pagination.py`：版本 2 游标；`strategy.py`：探索状态、查询指纹、无增益计数和 coverage 引导。
- `agent_core/codeintel/working_set.py`：`TaskWorkingSet`，持久化受限证据引用、查询和工作区绑定；不持久化原始源码、查询全文和编辑版本。
- `agent_core/codeintel/runtime.py`：按 workspace/config/policy 共享服务和工作集，提交后失效读版本并触发发布。

### 2.2 工具、规划和上下文

- `agent_core/tools/codeintel.py`：`_CodeTool`、`code_search`、`code_relations`、`code_context`、`code_validate`；负责本地优先检索、权限、锁、版本记忆和结果记录。
- `agent_core/tools/lsp.py` 与 `agent_core/codeintel/lsp_evidence.py`：LSP 启动、Document/definition/references/implementation 证据整合、路径/权限/UTF-16/版本检查。LSP coverage 目前始终标记为 incomplete/server coverage unknown。
- `agent_core/planner.py` 与 `agent_core/tools/planner.py`：`plan_code_task()` / `PlanCodeTaskTool`，目前按 1–4 条 query 获取文本/符号/关系/上下文，生成 JSON 计划；尚未做自动检索策略路由。
- `agent_core/react.py`、`agent_core/session.py`、`agent_core/tools/task_control.py`：ReAct 工具注册、任务状态、代码证据页、LSP/验证工作区绑定和清理。
- `agent_core/compression.py`、`agent_core/rendering.py`：压缩和完整 JSON/引用渲染；需继续防止压缩丢失关键候选、游标和 recovery 信息。

### 2.3 编辑、快照、检查点和验证

- `agent_core/tools/transaction.py`、内置编辑工具：以 expected version、snapshot/precondition 和 transaction overlay 防止误改；持久提交后记录 change event，失败时进入恢复/重建路径。
- `agent_core/codeintel/source_snapshots.py`：不可变 manifest/CAS、分块流式存储、`SnapshotReader`、`SnapshotStore.materialize()`；支持独立可写验证副本。
- `agent_core/codeintel/checkpoints.py`：检查点 v2 引用 manifest，兼容旧版本，避免复制大正文。
- `agent_core/codeintel/verifier_workspace.py`：`VerifierWorkspace`，按任务/工作区/digest/snapshot 复用一份独占副本；每次 lease 做清单、哈希、模式和策略校验，脏或多余文件则整副本重建。
- `agent_core/verifier.py`：`VerifierConfig.reuse_workspace`、行为验证、独立模型 probe、最终 live capture；worker 取消 drain 仍需单独复核。

### 2.4 评估和基准

- `agent_core/benchmarks/repository.py`：版本化 gold/observation，计算 Precision@K、Recall@K、MRR、定位和 resolved；当前 aggregator 不执行测试、不做证明，只消费外部 attestation。
- `agent_core/benchmarks/repository_tasks.py`：仓库任务 CLI。
- `agent_core/benchmarks/codeintel.py`：合成检索/增量基准，默认约 100k LOC，可生成 1M/10M synthetic，但不是真实仓库验收。
- `agent_core/benchmarks/codeintel_maintenance.py`：高文件数维护基准，支持 2–20,000 文件、总量不超过 2M LOC、构建/audit/增量/并发查询指标。
- `agent_core/benchmarks/source_snapshots.py`：快照冷/热/变更/检查点/复制和复用基准。
- `agent_core/benchmarks/swebench/*`：`SWEbenchRunner`、运行时和官方 harness；不能直接视为通用仓库任务适配器。

## 3. 已实现、部分实现和未实现能力

### 已实现且有测试覆盖

- 本地索引、FTS/trigram、符号预筛选、版本绑定游标、分页和 bounded query。
- Python AST 结构事实、manifest 外部依赖事实、模块边界基础信息。
- LSP 定义/引用/实现证据的权限、路径、版本、UTF-16 和源代码复核。
- 精确候选机械校验、expected-version 编辑保护、事务 overlay。
- 不可变源码快照、检查点引用、验证副本复用及脏副本重建。
- watcher/cancel/close/锁等待预算、运行时服务复用和持久工作集引用。

### 部分实现且存在明确局限

- `replay_commits()` 对巨大单事件仍须完整解析，短预算可能饥饿；未做到 per-path 可恢复消费。
- Planner 有固定的检索流程，但没有根据需求、失败、歧义、无增益和可用能力动态路由。
- LSP 证据已整合，但没有统一候选池、角色（target/dependency/test）和冲突证据评估；调用层级存在 `prepared[0]` 选择限制。
- 工作集恢复能标记 `needs_revalidation`，但没有持久化结构化候选记录和验证状态。
- 模块解析只提供最近 manifest/边界，不提供完整 workspace/local-package 静态图。
- 查询缓存每次仍需验证文件 hash；快照/验证复用仍须完整清单检查，不能宣称 O(1) 复用。
- 现有维护数据没有真实 1M/10M 仓库、内存、私有 ACL、长尾 P95 或真实任务解决率证据。

### 尚未实现、需新增

- 文件/符号/编辑范围级别的金标准和定位指标。
- per-path giant outbox 与恢复游标。
- 统一候选模型、证据冲突、角色评估和可持久化上下文。
- 分阶段检索控制器及 Agent 路由。
- 结构化 LSP document symbols 和有界 call hierarchy。
- workspace/local-package 静态依赖图。
- 面向 1M/10M LOC 和 >20k 文件的合成压力矩阵及离线端到端验收适配器。
- 真实仓库/真实模型验收；需外部仓库路径、模型配置、依赖和隐藏测试。

## 4. 分阶段实施步骤

### S0：基线、门禁与文档修正（P0）

**依赖：无。**

1. 确认 HEAD、工作树、Python/venv、配置和当前测试基线；记录源 SHA、配置 hash、环境指纹。
2. 运行既有结构、维护、watcher、LSP、候选、事务、快照、验证、planner、React、任务运行时测试。
3. 修正 `docs/repository-task-evaluation.md`、`docs/code-intelligence.md` 中仍将下一阶段能力写成“已实现”的表述，明确 LSP/模块图/验证副本复用的边界。
4. 建立每次性能比较的 before/after 原始 JSON、SHA、配置、随机种子和失败分母规则。

**验收：**工作树干净；既有测试通过；文档与代码状态一致；任何环境错误单独标记，不伪装成通过。

### S1：定位级指标与对比协议（P0）

**依赖：S0。** 主要文件：`benchmarks/repository.py`、`repository_tasks.py`、新增/扩展 benchmark 测试。

1. 扩展 versioned gold：`relevant_files` 之外增加符号、定义/引用目标、允许的编辑 range、角色和证据来源。
2. 分开记录：检索返回候选、Agent 选中的候选、候选机械校验、实际 patch 位置、隐藏测试结果。
3. unknown 成本使用 `null`，不填零；所有失败（timeout、environment error、patch failure、test failure）进入分母。
4. 不跨工作区比较 `symbol_reference.id`（其中含 worktree id）；使用规范化 `(path, qualified, kind, range, content version)`。
5. 重新定义 compare：允许声明处理变量（模型、硬件、配置、任务顺序）和配对/非配对实验，不能继续要求所有环境字段绝对相等。

**验收：**旧 manifest v1 可读；新 schema 能计算 file/symbol/edit-range Precision@K、Recall@K、MRR；缺失标注不会被错误判定为失败或成功。

### S2：巨大变更 outbox 的逐路径可恢复消费（P0）

**依赖：S0。** 主要文件：`codeintel/store.py`、`changes.py`、数据库迁移/测试。

1. 增加版本化规范化表：事件、路径项、消费状态/代次，索引至少覆盖 `(event_seq, next_ordinal)`；保留 legacy v1 读路径。
2. `replay_commits()` 每次只消费有界路径数，事件只有在全部 paths 完成后才标记完成。
3. 在同一事务中重读消费游标、批量写入 pending、更新游标；取消、deadline、异常均回滚本批次，不能跳过路径。
4. 不要在第一批消费后把 dedupe 标记为完成；必须支持重复调用和新版本代次。
5. 查询/迁移前先检查 payload 大小；旧巨型 JSON 若无法安全拆分，应报告 unsupported/diagnostic，不得悄悄 advance 或 skip。
6. 先确认当前 SQLite/驱动实际支持的流式 API，再决定是否采用；不得只凭 API 名称假设可流式解析。

**验收：**短 budget 下巨型事件可持续前进；中断后从正确 ordinal 恢复；失败批次原子回滚；新旧 schema、重复事件、路径删除和并发 replay 均有测试。

### S3：候选记录、角色和证据整合（P0）

**依赖：S1；可与 S2 部分并行。** 主要文件：`models.py`、`candidates.py`、`working_set.py`、`task_control.py`、`lsp_evidence.py`。

新增受限 `CandidateRecord`（建议 schema v1）：

- `id`、`path`、`expected_version`、`symbol`、`range`；
- `role`: `target | dependency | test | unknown`；
- `evidence_refs`、supporting/contradicting evidence、assessment；
- `anchor_validation` 与 `relevance_validation` 分开；
- validation state、失效原因、来源操作。

复用 `TaskRun.code_exploration` 和工作集，新增 versioned working-set 子格式；读取旧 v1。历史中只保存 bounded 引用和摘要，不保存完整源码、原始 query 或编辑凭据。恢复候选默认 `needs_revalidation`；旧 `code_validate` 参数保持兼容。

**验收：**文本、符号、LSP、关系证据可以合并为同一候选；冲突可见；工作区/版本变化只使候选失效而不继续用于编辑；旧任务状态可恢复。

### S4：分阶段检索控制器与 Agent 路由（P0）

**依赖：S3。** 主要文件：`strategy.py`、`planner.py`、`tools/planner.py`、`react.py`、`tools/codeintel.py`。

控制器输入：显式路径/标识符、自然语言 query、错误文本、当前候选、歧义数、可用 LSP/模块能力、重复/无增益状态、cursor/coverage 和总 budget。输出下一阶段动作：

1. 路径/模块范围确认；
2. 词法/正则检索；
3. 符号定义与引用；
4. 语义检索（仅当词法/结构召回不足）；
5. 关系/调用/依赖扩展；
6. 受限上下文读取；
7. 候选验证或停止并请求澄清。

所有子调用共享一个总 `QueryBudget`，而不是每次重新获得独立额度。延迟新增 `code_locate` 工具，先保持 `plan_code_task(queries)` 和旧工具兼容；不要改动现有 `code_search` 游标排序而破坏恢复。

重要并发约束：`PlanCodeTaskTool` 持有任务/文件锁时不得嵌套 dispatch executor。应在释放外层锁后由正常 ReAct 循环执行阶段动作，或验证一个安全的执行 seam；自动 LSP 也不能绕过权限、hook、锁、日志和沙箱。

**验收：**重复 query 会收敛而非无限重试；无结果、证据冲突、候选过多时可停止/升级；工具调用次数和预算可观测；旧 planner 行为回归通过。

### S5：结构化 LSP symbols 与有界调用层级（P1）

**依赖：S4 的候选/控制器接口；服务器能力部分可并行。** 主要文件：`tools/lsp.py`、LSP manager 相关模块、`lsp_evidence.py`。

1. 支持 DocumentSymbol 和 SymbolInformation，保留旧 raw LSP shape；新集成操作使用显式 structured mode。
2. 所有 range 使用服务器声明的 UTF-16；保留 surrogate boundary 检查、当前版本 reread、路径 containment 和权限检查。
3. 修复 call hierarchy `prepared[0]` 的歧义：按候选/符号匹配，若多项无法唯一确定则返回 ambiguity，不静默取第一项。
4. 为 LSP manager 的 `open_document` 读源纳入外层总 budget；当前会创建独立默认 `QueryBudget`，需消除预算旁路。
5. 未配置、服务器不支持、路径不在 workspace 或覆盖范围未知时，返回结构化 unavailable/coverage unknown，而非伪造 complete。

**验收：**定义/引用/实现/文档符号结果可转换为候选证据；UTF-16、多符号、服务器不支持、取消和源文件变化均有测试；不承诺完整跨语言 dataflow。

### S6：静态 workspace/local-package 依赖图（P1）

**依赖：S3；与 S5 实现可并行。** 主要文件：`backends.py`、`modules.py`、`store.py`、manifest 解析及测试。

扩展现有四类 manifest 的静态事实，增加边：来源 manifest、from/to module/package、version/path、unresolved reason、最近 module 兼容性：

- npm workspaces 和 name/path refs；
- Cargo workspace/path dependencies；
- Go module/local replace；
- 明确限定的 Python local package/project refs。

禁止执行 package/build 脚本；不在查询时遍历全文件系统，使用有界索引 inventory。暂不强制 Bazel/Gradle，除非基准明确显示其必要性。

**验收：**同名模块、缺失 manifest、循环依赖、路径越界、解析失败均可解释；关系扩展受 `max_edges/max_depth` 约束；不把外部 registry 依赖误报为本地源码边。

### S7：百万/千万 LOC 与高文件数离线压力验证（P0/P1）

**依赖：S1–S6 中对应接口稳定。**

测试矩阵至少包括：

- 1M LOC、10M LOC；
- 低文件数和 >20,000 文件；重复/唯一/混合内容；
- 深层模块、相同符号名、多语言 manifest；
- 变更、保持时间戳、删除、并发服务、publish burst、cancel/close、overlay/worktree/resume；
- 本地存储与私有 ACL/权限场景。

每类至少 50 个查询样本，3 次独立重建，随机化并交替执行 before/after；保存原始 JSON、配置、种子、源 SHA 和 hash。现有 benchmark 的生成配额和总量上限需显式扩展，不能只把参数名字改成 10M。

采集：P50/P95 检索/定位/索引/增量延迟、Recall/Precision/MRR、峰值内存、SQLite/manifest/快照磁盘、token、工具/LLM 调用次数、patch 成功和测试通过率。

**验收：**只报告实测值；理论可行性和当前实现能力分开。若环境无法容纳 10M，必须报告资源限制和可复现的小规模替代，而不是推算真实指标。

### S8：离线任务适配器与真实仓库/模型验收（外部依赖）

**依赖：S7；需要用户提供仓库、模型配置、依赖和隐藏测试。**

可以复用 `SWEbenchRunner`、`InstanceRuntime`、`OfficialHarnessAdapter` 的部分运行逻辑，但不能假设它们已经是通用 manifest adapter。先开发离线执行适配器：任务 gold/hidden tests 位于 solver workspace 外；记录 base/config/model/deps/budgets；独立记录 patch SHA、测试结果和失败原因。

验收顺序：每类任务先 smoke；正式配对集至少 100 个经人工审核任务；阈值在实验前固定。真实 solved-rate 未运行前保持 unknown，不得由检索指标推导。

### S9：可选增强（仅在测量证明必要时）

可选项：验证副本增量修复、embedding/rerank、物理索引分片、常驻 daemon、对象 pack。优先复用当前验证副本检查；若目录/文件/哈希/权限/链接状态不确定，整副本重建。复核 `verifier_workspace.bounded_thread` 的重复取消 drain；不要把服务层的取消修复自动套用过去。

embedding 只有在词法/结构 Recall 明显不足时才引入；分片/daemon/pack 只有在资源、并发或磁盘实测证明单库不可接受时才引入。

## 5. 数据流与兼容性要求

目标数据流：

```text
用户需求
  -> ExplorationController（共享 budget/coverage）
  -> lexical/symbol/LSP/module/dependency evidence
  -> CandidateRecord（target/dependency/test + conflicts）
  -> bounded context + latest version reread
  -> code_validate / transaction edit
  -> Git diff + syntax/static checks + targeted tests
  -> versioned observation / task result
```

兼容性必须满足：

- 旧 `SearchRequest`、cursor v2、`code_validate` 入参、LSP raw JSON、TaskStore v1–v5 均可读。
- 新字段采用 additive schema；旧记录读取后标记需要重新验证，不直接编辑。
- 不改变已有权限、hook、锁、取消、日志和 sandbox 语义。
- 不把候选证据、快照或工作集中的完整源码复制到长期任务历史。
- 每个版本迁移有明确版本号、回滚/诊断路径和损坏数据处理。

## 6. 验证命令与测试策略

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B -m ruff check agent_core tests installer
.\.venv\Scripts\python.exe -B -m mypy agent_core
git diff --check
git status --short
```

Windows 沙箱可能在最小 `asyncio.run(sleep(0))` 处卡在 Proactor/socketpair，而非项目代码。遇到此情况，使用已验证的私有隔离 harness：先创建 scratch `POLARIS_HOME`、`AGENT_TRUST_STORE`、`TMP/TEMP` 和短 `--basetemp`，再运行目标测试；记录为环境限制，不弱化断言。保持临时目录在仓库 `tmp/` 或系统临时目录内，避免长路径。

阶段测试重点：

- S1：gold schema、跨工作区符号规范化、missing/unknown 分母、paired comparison。
- S2：巨型事件短预算、恢复 ordinal、重复/取消/失败原子性、legacy migration。
- S3：候选合并、冲突、角色、工作集恢复和版本失效。
- S4：策略收敛、共享 budget、锁/取消/权限回归、旧 planner。
- S5：UTF-16、DocumentSymbol、多候选 call hierarchy、server unavailable、源变化。
- S6：四类 manifest、循环/缺失/越界/外部依赖。
- S7：合成规模矩阵、并发、内存、长尾、磁盘和原始数据完整性。
- S8：patch、隐藏测试、环境错误、超时、模型失败的独立归因。

## 7. 风险、假设与待确认事项

已由代码验证的结论：当前索引、LSP 证据、候选机械校验、快照/检查点/验证副本复用、watcher/取消/权限基础能力存在；对应测试文件已在仓库中。

仍属假设、必须测量：

- 1M/10M LOC 的真实 P95、内存、磁盘和索引构建时间；
- SQLite 单库在目标并发和私有 ACL 下是否成为瓶颈；
- 词法+结构召回是否足以支撑目标任务，是否真的需要 embedding/rerank；
- LSP 服务器覆盖率、跨语言解析质量和 call hierarchy 稳定性；
- 验证副本完整清单检查在真实大仓库中的成本；
- 真实模型工具调用数、token 和任务解决率。

执行前需进一步检查的位置：

- `changes.py` 中实际 SQLite 驱动能力及迁移兼容性；
- `LSPManager.open_document` 的预算注入 seam；
- `PlanCodeTaskTool` 持锁期间的调用栈，确认不能嵌套 executor；
- `verifier_workspace.py` 重复取消时 worker drain 是否会残留；
- `benchmarks/repository.py::compare` 的 treatment 设计；
- 目标真实仓库的语言、构建系统、权限、模型和隐藏测试入口。

主要风险及缓解：

1. schema 迁移破坏旧任务：保留 v1 读路径，先双读/只读迁移，加入损坏诊断。
2. 候选池膨胀：所有证据、关系、上下文和持久字段均受预算/数量上限约束。
3. 自动路由死循环：query fingerprint、no-gain、重复候选和最大阶段数必须是硬门禁。
4. 过期上下文误改：编辑前重新读取并验证 expected version；旧候选恢复必须 revalidate。
5. 性能结果不可比：固定基线、随机种子、配置 hash、环境指纹和 paired order。
6. 复杂组件过早引入：只有 S7 的实测瓶颈达到预设阈值，才进入 S9。

## 8. 完成定义与优先级

### 必须优先完成

- S0 基线/文档/实验门禁；
- S1 定位级指标；
- S2 巨大 outbox；
- S3 候选和证据模型；
- S4 分阶段控制器；
- 至少完成 S7 的离线 1M/10M 合成验证。

### 重要优化

- S5 结构化 LSP 与有界调用层级；
- S6 静态 workspace/local-package 图；
- 高文件数、并发、取消、恢复和资源成本的完整压力矩阵。

### 外部验收/可选增强

- S8 真实仓库和真实模型任务验收，取决于用户提供输入；
- S9 embedding、分片、daemon、object pack、增量 verifier 修复，仅凭实测需求启用。

整体任务只有在以下条件同时满足时，才能声称“离线改造完成”：S0–S7 的代码、迁移、测试和可复现实验完成，报告明确区分检索/定位/修改/测试结果，且所有关键指标都有原始数据。不能因为搜索速度改善，就声称已具备真实千万行仓库的端到端任务能力；S8 仍应标为待外部验收。

建议下一步严格从 **S0 → S1/S2** 开始，先建立可比较的定位指标和可恢复变更消费，再推进候选、路由、结构化 LSP 与模块图。
