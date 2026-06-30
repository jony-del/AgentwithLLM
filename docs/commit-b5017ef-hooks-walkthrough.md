# `b5017ef` 深度阅读指南：配置驱动外部钩子 + 内建生命周期钩子

本文档带你从「为什么」到「怎么做」，完整理解这个提交在 Hook 体系上做了什么。

---

## 目录

1. [本次提交解决的问题](#1-本次提交解决的问题)
2. [全貌：五个生命周期事件](#2-全貌五个生命周期事件)
3. [数据层：三个新配置类（hooks.py）](#3-数据层三个新配置类hookspy)
4. [内建钩子层（builtin_hooks.py）](#4-内建钩子层builtin_hookspy)
5. [外部适配器层（hook_adapters.py）](#5-外部适配器层hook_adapterspy)
6. [配置解析层（config.py）](#6-配置解析层configpy)
7. [装配层：`_build_hook_pipeline`（react.py）](#7-装配层_build_hook_pipelinereactpy)
8. [用户配置界面（agent.toml.example）](#8-用户配置界面agenttomlexample)
9. [一次完整运行的时序图](#9-一次完整运行的时序图)
10. [设计决策：为什么这样做](#10-设计决策为什么这样做)

---

## 1. 本次提交解决的问题

### 提交之前的世界

在这个提交之前，`HookPipeline` 的生命周期列表默认都是空的：

```python
# 旧代码（示意）
self.hooks = HookPipeline(
    post_hooks=[MaxOutputPostHook(...)],  # 只有工具输出截断
    # user_prompt_hooks = []     ← 空的
    # post_sampling_hooks = []   ← 空的
    # pre_compact_hooks = []     ← 空的
    # post_compact_hooks = []    ← 空的
    # stop_hooks = []            ← 空的，模型停下来就停，没人检查
)
```

这意味着：
- 模型有未完成的待办事项时，仍然可以直接退出，没有人阻止
- 每次采样完成后没有任何观测数据写入日志
- 压缩前后没有任何记录
- 用户要扩展行为，必须修改 Python 代码

### 提交之后的世界

```
                    ┌─────────────────────────────────────┐
                    │           HookPipeline               │
                    │                                     │
agent.toml ─────→  │  stop_hooks:                        │
 [hooks]           │    [StopCompletionHook, ...]         │
 [[hooks.external]]│                                     │
                   │  post_sampling_hooks:               │
内建代码 ────────→  │    [PostSamplingObserverHook, ...]   │
builtin_hooks.py   │                                     │
                   │  pre_compact_hooks / post_compact:  │
                   │    [CompactionLoggerHook, ...]       │
                   └─────────────────────────────────────┘
```

现在 Hook Pipeline 有了真实的默认行为，而且用户可以通过 `agent.toml` 增删外部钩子，**无需修改代码**。

---

## 2. 全貌：五个生命周期事件

在深入代码之前，先理解这个体系监听的五个事件（定义在 `hooks.py` 的 `HookEvent`）：

```
用户输入任务
    │
    ↓
[UserPromptSubmit] ← 可以：拦截空输入、注入上下文
    │
    ↓
┌──────────────────────────────┐
│       ReAct 主循环           │
│                              │
│  [PreCompact]  ← 可以：跳过压缩、记录日志
│  ↓ 压缩历史                  │
│  [PostCompact] ← 可以：记录压缩结果     │
│                              │
│  调用 LLM → 得到回复          │
│  [PostSampling]← 只观测，不阻断        │
│                              │
│  if 无工具调用:               │
│    [Stop]  ← ★ 可以：阻止退出，让模型继续
│    return 最终答案            │
│                              │
│  执行工具（pre/post tool hook）│
│  继续下一轮                  │
└──────────────────────────────┘
```

> [!IMPORTANT]
> `Stop` 是本次提交最关键的事件。它允许钩子检查「模型是否真的完成了任务」，如果没有则强制模型继续工作。

---

## 3. 数据层：三个新配置类（hooks.py）

这个提交在 [`hooks.py`](../agent_core/hooks.py) 里新增了三个配置数据类，它们是整个体系的「数据契约」。

### `BuiltinHooksConfig` — 内建钩子开关

```python
@dataclass(slots=True)
class BuiltinHooksConfig:
    stop_completion: bool = True        # ← 默认开：检查待办事项
    post_sampling_observer: bool = True # ← 默认开：观测每轮采样
    compaction_logger: bool = True      # ← 默认开：记录压缩事件
    user_prompt_context: bool = False   # ← 默认关：提示词注入
```

**为什么 `user_prompt_context` 默认关？**
因为 `context.py` 已经在 run 开始时注入了 `userContext`（包含 CLAUDE.md 和日期）。如果这个钩子也开着，同样的信息会注入两次（double-grounding），浪费 token 还可能让模型困惑。

### `ExternalHookSpec` — 一条外部钩子的完整描述

```python
@dataclass(slots=True)
class ExternalHookSpec:
    event: str          # 挂在哪个事件上（"Stop", "UserPromptSubmit" 等）
    type: str           # 用什么 transport（"command"/"http"/"prompt"/"agent"）
    matcher: str | None # 仅 Pre/PostCompact 有效，匹配 "auto"|"reactive"
    command: str | None # type="command" 时用
    url: str | None     # type="http" 时用
    prompt: str | None  # type="prompt"/"agent" 时用
    model: str | None   # type="prompt"/"agent" 时，指定用哪个模型
    headers: dict | None# type="http" 时，额外的 HTTP 请求头
    timeout: float = 30.0 # 所有 transport 共用的超时时间（秒）
```

### `HooksConfig` — 整个 `[hooks]` 表的顶层容器

```python
@dataclass(slots=True)
class HooksConfig:
    enabled: bool = True                              # 主开关
    builtin: BuiltinHooksConfig = field(...)         # 内建钩子开关组
    external: list[ExternalHookSpec] = field(...)    # 外部钩子列表
```

**设计要点**：这三个类都是纯数据（`@dataclass`），不持有任何「活着的对象」（如 session、logger），所以它们可以在 `config.py` 里安全地从 TOML 文件解析出来，**在 `ReActAgent` 构造之前就完成**。

---

## 4. 内建钩子层（builtin_hooks.py）

[`builtin_hooks.py`](../agent_core/builtin_hooks.py) 是新增的文件，包含四个钩子类。

### 为什么叫「内建」？

因为这些钩子持有「活着的对象」——运行中的 `session` 和 `logger`，只有 agent 实例创建后才能存在。它们做的事情是配置驱动的外部进程（`hook_adapters.py`）做不到的：**直接读取模型的待办事项列表**。

```
builtin_hooks.py 的钩子                 hook_adapters.py 的钩子
     │                                           │
     ↓                                           ↓
持有 session / logger 对象              只收到 JSON 快照
可以读 session.todos                    看不到 todos
直接在进程内运行                         跨越进程/网络边界
```

### `StopCompletionHook` — 最重要的内建钩子

```python
class StopCompletionHook:
    def __init__(self, session: SessionContext) -> None:
        self.session = session

    async def on_stop(self, ctx: HookContext) -> HookOutcome:
        try:
            if ctx.stop_hook_active:       # ① 本次 run 已经阻断过一次了？放行
                return HookOutcome()
            open_items = [
                todo for todo in self.session.todos.items()
                if todo.status in _OPEN_TODO_STATUSES  # "pending" 或 "in_progress"
            ]
            if not open_items:             # ② 没有未完成项？正常退出
                return HookOutcome()
            listing = ...
            return HookOutcome(            # ③ 有未完成项！阻断停止
                block=True,
                reason=f"{len(open_items)} unfinished to-do item(s) remain.",
                additional_context="...",  # 注入续跑指令到对话
            )
        except Exception:
            return HookOutcome()           # ④ 任何异常 → 放行，绝不崩溃
```

**三道安全网**：
1. `ctx.stop_hook_active`：本次 run 内只阻断一次（第一次阻断后此字段变为 True）
2. `react.py` 的 `max_stop_blocks`：全局上限，无论如何最多阻断 N 次
3. `except Exception`：钩子代码的任何 bug 都被吞掉，run 照常继续

### `PostSamplingObserverHook` — 纯观测钩子

```python
class PostSamplingObserverHook:
    async def after_sampling(self, ctx: HookContext) -> None:  # 注意：返回 None，不是 HookOutcome
        try:
            await self.logger.write("hook_observe", {
                "hook": "PostSampling",
                "messages": len(ctx.messages),
                "assistant_chars": len(ctx.last_assistant_message or ""),
            })
        except Exception:
            pass  # 观测失败不影响任何事
```

**关键设计**：`after_sampling` 返回 `None` 而不是 `HookOutcome`，因为 PostSampling 是**纯观测性**的，它不能阻断任何东西。这在 Protocol 层面就固定了（见 `hooks.py` 的 `PostSamplingHook`）。

### `CompactionLoggerHook` — 一个类实现两个时间点

```python
class CompactionLoggerHook:
    async def before_compact(self, ctx: HookContext) -> HookOutcome:
        # 记录「压缩前」的消息数量
        await self.logger.write("hook_observe", {"hook": "PreCompact", "messages": len(ctx.messages)})
        return HookOutcome()  # 不阻断

    async def after_compact(self, ctx: HookContext) -> HookOutcome:
        # 记录「压缩后」的摘要长度
        await self.logger.write("hook_observe", {"hook": "PostCompact", "summary_chars": ...})
        return HookOutcome()  # 不阻断
```

在 `react.py` 装配时，**同一个实例**被同时加入 `pre_compact_hooks` 和 `post_compact_hooks`：

```python
compaction_hook = CompactionLoggerHook(self.logger)
pipeline.pre_compact_hooks.append(compaction_hook)   # ← 同一个对象
pipeline.post_compact_hooks.append(compaction_hook)  # ← 同一个对象
```

### `UserPromptContextHook` — 示范性注入钩子（默认关）

```python
class UserPromptContextHook:
    async def on_user_prompt(self, ctx: HookContext) -> HookOutcome:
        try:
            if not (ctx.prompt or "").strip():
                return HookOutcome(block=True, reason="Empty prompt rejected.")  # 拦截空输入
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            return HookOutcome(additional_context=f"Prompt submitted at {stamp}.")  # 注入时间戳
        except Exception:
            return HookOutcome()
```

这个钩子主要是**教学性质**——展示 `UserPromptSubmit` 事件的两种能力：阻断 + 注入上下文。

---

## 5. 外部适配器层（hook_adapters.py）

[`hook_adapters.py`](../agent_core/hook_adapters.py) 是另一个新文件，解决「用户不想写 Python 代码，但想扩展 hook 行为」的问题。

### 核心思想：统一接口，四种 transport

```
                         ┌──────────────────┐
agent.toml 配置          │  _ExternalHookAdapter  │
[[hooks.external]]  →   │  (基类，实现所有 Protocol)│
                         └────────┬─────────┘
                                  │ 派生
              ┌───────────────────┼────────────────────┐
              ↓                   ↓                    ↓                    ↓
  CommandHookAdapter    HttpHookAdapter    PromptHookAdapter    AgentHookAdapter
  (spawn 子进程)         (POST HTTP 请求)   (re-prompt LLM)      (运行子 Agent)
```

每个适配器的外部接口都是 `HookPipeline` 里的同一套 Protocol，所以 `react.py` 的循环对它们完全无感知——外部 hook 和内建 hook 放进同一个列表里。

### `project_hook_input` — 跨边界的 JSON 投影

内建钩子可以直接读 `session` 对象，但外部进程不可能拿到 Python 对象。因此必须把 `HookContext` 序列化成 JSON 再传过去：

```python
def project_hook_input(ctx: HookContext, *, max_messages=20, max_content_chars=2000):
    data = {
        "hook_event_name": ctx.event.value,   # 事件名称
        "session_id": ctx.session_id,
        "stop_hook_active": ctx.stop_hook_active,
        # 事件特有字段
        "prompt": ctx.prompt,          # 仅 UserPromptSubmit
        "trigger": ctx.trigger,        # 仅 Pre/PostCompact
        "summary": ctx.summary,        # 仅 PostCompact
        # 最近 20 条消息的截断版本
        "messages": [{"role": m.role, "content": m.content[:2000]} for m in tail],
    }
    return data
```

**关键约束**：最多 20 条消息，每条内容截断到 2000 字符。这防止了把整个对话历史（可能几万 token）通过 stdin/HTTP 传给外部进程。

### `CommandHookAdapter` — 进程通信

```python
class CommandHookAdapter(_ExternalHookAdapter):
    async def _invoke(self, ctx: HookContext) -> HookOutcome:
        payload = json.dumps(project_hook_input(ctx)).encode("utf-8")
        proc = await asyncio.create_subprocess_shell(
            self.spec.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(payload), timeout=self.spec.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()   # ← kill 后必须 await，否则产生「僵尸进程」
            return HookOutcome()
        return outcome_from_output(stdout.decode(), proc.returncode or 0)
```

**「退出码 2 = block」** 是参考实现（Claude Code）的约定：外部命令用退出码 2 告诉 agent「阻断这次操作」。也可以输出 JSON `{"continue": false}` 达到同样效果。

### `HttpHookAdapter` — HTTP 通信

用标准库 `urllib.request`（不依赖 requests/httpx）向指定 URL 发 POST 请求，响应体的 JSON 格式和命令行输出一致：

```json
{
  "continue": false,
  "stopReason": "安全检查未通过",
  "hookSpecificOutput": {
    "additionalContext": "发现敏感词 X，请修改后重试。"
  }
}
```

### `PromptHookAdapter` / `AgentHookAdapter` — LLM 辅助决策

这两个 transport 是**纯咨询性质**（advisory-only）：
- 它们调用 LLM 或子 Agent 获得文字建议，注入为 `additional_context`
- **永远不会 `block=True`**，因为让一个模型来决定是否阻断另一个模型的输出，风险太高

```python
class PromptHookAdapter(_ExternalHookAdapter):
    async def _invoke(self, ctx: HookContext) -> HookOutcome:
        snapshot = json.dumps(project_hook_input(ctx))
        result = await self.provider.complete([
            Message("user", f"{self.spec.prompt}\n\n<hook_input>\n{snapshot}\n</hook_input>")
        ], [], config)
        return HookOutcome(additional_context=result.content or None)
        # 注意：没有 block=True！
```

### 基类的「降级到放行」保障

所有适配器都继承 `_ExternalHookAdapter`，它的 `_run` 方法是所有外部调用的入口：

```python
async def _run(self, ctx: HookContext) -> HookOutcome:
    if not self._matches(ctx):         # matcher 检查
        return HookOutcome()
    try:
        return await self._invoke(ctx) # 真正的调用
    except Exception as exc:           # 任何失败
        await self._log("exception", ...) # 记录日志
        return HookOutcome()           # 降级为「放行」
```

**原则**：外部钩子的任何失败（超时、进程崩溃、HTTP 503、模型拒绝）都不能让整个 run 失败。

---

## 6. 配置解析层（config.py）

[`config.py`](../agent_core/config.py) 新增了 `resolve_hooks_config()` 函数，负责从 `agent.toml` 读取 `[hooks]` 表并构造 `HooksConfig`。

```python
def resolve_hooks_config(config_file="agent.toml") -> HooksConfig:
    table = load_agent_toml(config_file).get("hooks")
    config = HooksConfig()  # 从默认值出发

    if isinstance(table, dict):
        # 1. 主开关
        if "enabled" in table:
            config.enabled = coerce_to_type(bool, table["enabled"])

        # 2. 内建钩子开关
        builtin = table.get("builtin")
        if isinstance(builtin, dict):
            config.builtin = config.builtin.from_dict(builtin)

        # 3. 外部钩子列表（[[hooks.external]] 是 TOML 的数组写法）
        external = table.get("external")
        if isinstance(external, list):
            for entry in external:
                spec = _parse_external_hook(entry, valid_events)
                if spec is not None:  # None = 这条记录有问题，跳过
                    config.external.append(spec)

    # 环境变量可以覆盖主开关（AGENT_HOOKS=0 关闭所有钩子）
    env = os.getenv("AGENT_HOOKS")
    if env is not None:
        config.enabled = env.strip().lower() in {"1", "true", "yes", "on"}

    return config
```

### `_parse_external_hook` — 宽松验证

```python
def _parse_external_hook(entry, valid_events) -> ExternalHookSpec | None:
    if not isinstance(entry, dict):         # 不是字典 → 跳过
        return None
    event = str(entry.get("event", ""))
    hook_type = str(entry.get("type", ""))
    if event not in valid_events:           # 未知事件名 → 跳过
        return None
    if hook_type not in _HOOK_TYPES:        # 未知类型 → 跳过
        return None
    # 每种 type 有它必须有的字段
    required = {"command": "command", "http": "url", "prompt": "prompt", "agent": "prompt"}
    if not entry.get(required[hook_type]):  # 缺少必填字段 → 跳过
        return None
    # ... 构造 ExternalHookSpec
```

**「降级，不崩溃」原则**：一条格式错误的 `[[hooks.external]]` 只会让那一条被静默跳过，不会让整个 agent 构造失败。

---

## 7. 装配层：`_build_hook_pipeline`（react.py）

这是所有东西汇聚的地方。`ReActAgent` 新增了 `_build_hook_pipeline` 方法：

```python
def _build_hook_pipeline(self) -> HookPipeline:
    # 1. 底线：工具输出截断钩子（原来就有的）
    pipeline = HookPipeline(
        post_hooks=[MaxOutputPostHook.from_config(...)]
    )

    hooks_config = self.config.hooks
    if not hooks_config.enabled:     # 主开关关闭 → 只保留工具钩子
        return pipeline

    # 2. 内建钩子（按配置开关决定是否加入）
    builtin = hooks_config.builtin
    if builtin.stop_completion:
        pipeline.stop_hooks.append(StopCompletionHook(self.session))
    if builtin.post_sampling_observer:
        pipeline.post_sampling_hooks.append(PostSamplingObserverHook(self.logger))
    if builtin.compaction_logger:
        compaction_hook = CompactionLoggerHook(self.logger)
        pipeline.pre_compact_hooks.append(compaction_hook)
        pipeline.post_compact_hooks.append(compaction_hook)  # 同一个实例
    if builtin.user_prompt_context:
        pipeline.user_prompt_hooks.append(UserPromptContextHook())

    # 3. 外部适配器（每条 spec 转成一个 Adapter，加入对应的列表）
    for spec in hooks_config.external:
        attr = LIFECYCLE_EVENT_ATTRS.get(spec.event)  # e.g. "stop_hooks"
        if attr is None:
            continue
        try:
            adapter = build_external_adapter(
                spec,
                logger=self.logger,
                provider=self.provider,
                base_config=self._provider_config(),
                subagent_factory=self.session.subagent_factory,
            )
        except Exception:
            adapter = None  # 构造失败 → 跳过这个 hook
        if adapter is not None:
            getattr(pipeline, attr).append(adapter)  # 动态属性访问

    return pipeline
```

### 两个重要的调用时机

```python
class ReActAgent:
    def __init__(self, ..., hooks: HookPipeline | None = None):
        # ...
        # session 和 logger 在这里已经就绪
        self.hooks = hooks or self._build_hook_pipeline()
        #              ↑              ↑
        #     测试/库用的覆盖     默认从 config 装配
```

- `hooks=` 参数：测试代码传入 mock pipeline，绕过装配逻辑
- `_build_hook_pipeline()`：正常运行时的默认路径

在 `cli.py` 里，现在会先解析配置再构造 agent：

```python
# cli.py 的 build_agent 函数（示意）
hooks_config = resolve_hooks_config(config_file)
config = ReActConfig(..., hooks=hooks_config)
agent = ReActAgent(config, ...)
# 此时 ReActAgent.__init__ 调用 _build_hook_pipeline，
# 把 config.hooks 里的配置变成真实的 HookPipeline
```

---

## 8. 用户配置界面（agent.toml.example）

用户通过 `agent.toml` 控制整个 hook 体系，无需写代码：

```toml
[hooks]
enabled = true                 # false 或 AGENT_HOOKS=0 关闭整个 hook 子系统

[hooks.builtin]
stop_completion = true         # 有未完成待办就阻断退出
post_sampling_observer = true  # 记录每轮采样的遥测数据
compaction_logger = true       # 记录压缩前后的消息数
user_prompt_context = false    # 提交空 prompt 时报错 + 注入时间戳（默认关）

# 外部钩子示例（取消注释即启用）
# [[hooks.external]]
# event = "UserPromptSubmit"
# type = "command"
# command = "python3 ./.polaris/hooks/validate_prompt.py"
# timeout = 5

# [[hooks.external]]
# event = "Stop"
# type = "prompt"           # 让 LLM 帮忙检查任务是否真的完成
# prompt = "Is the user's task complete? Answer briefly."
# model = "claude-haiku-4-5-20251001"
# timeout = 30
```

---

## 9. 一次完整运行的时序图

```
agent.toml                    config.py              ReActAgent.__init__
  [hooks]            ──→   resolve_hooks_config()   _build_hook_pipeline()
  [[hooks.external]]    →   HooksConfig               │
                                                       │
                                          ┌────────────┴──────────────────┐
                                          │  HookPipeline                 │
                                          │  stop_hooks:                  │
                                          │    [StopCompletionHook,       │
                                          │     CommandHookAdapter(Stop)] │
                                          │  post_sampling_hooks:         │
                                          │    [PostSamplingObserver]     │
                                          │  pre/post_compact_hooks:      │
                                          │    [CompactionLogger]         │
                                          └───────────────────────────────┘
                                                       │
                                                       ↓
                                               ReActAgent.run()
                                                       │
        ┌──────────────────────────────────────────────│──────────────┐
        │ ReAct 循环                                    │              │
        │                                              │              │
        │  on start:  pipeline.run_user_prompt(ctx)   ←── HookPipeline│
        │                                              │              │
        │  each turn: pipeline.run_pre_compact(ctx)   ← (if needed)  │
        │             call LLM                         │              │
        │             pipeline.run_post_sampling(ctx) ← (fire-forget)│
        │                                              │              │
        │  no tools:  pipeline.run_stop(ctx)          ←── ★ 关键点   │
        │               │                              │              │
        │               ├── StopCompletionHook        │              │
        │               │   读 session.todos           │              │
        │               │   有未完成项 → block=True    │              │
        │               │   注入续跑指令               │              │
        │               │                              │              │
        │               └── CommandHookAdapter(Stop)  │              │
        │                   spawn 子进程               │              │
        │                   读 exit code / JSON        │              │
        │                                              │              │
        │               if block: 继续下一轮           │              │
        │               if not block: return 最终答案  │              │
        └──────────────────────────────────────────────│──────────────┘
```

---

## 10. 设计决策：为什么这样做

### Q1：为什么要两个来源（builtin + external），而不是只用一种？

| 维度 | 内建钩子（builtin_hooks.py） | 外部钩子（hook_adapters.py） |
|------|--------------------------|--------------------------|
| 持有对象 | 可以持有 session、logger 等活对象 | 只能收到 JSON 快照 |
| 扩展方式 | 需要修改 Python 代码 | 只需改 agent.toml |
| 能读 session.todos？ | ✓ 可以 | ✗ 不行 |
| 跨语言？ | ✗ 只能 Python | ✓ 任何语言 |
| 安全性 | 在进程内，可信 | 跨进程，需要 JSON 边界 |

两者互补，合并进同一个 `HookPipeline`，循环对区别无感知。

### Q2：为什么 `StopCompletionHook` 只阻断一次？

用 `ctx.stop_hook_active` 标记「本次 run 内已经阻断过一次停止」。原因：
- 模型可能有正当理由放弃某些待办（例如发现需求已变更）
- 如果反复阻断，模型会陷入死循环
- `max_stop_blocks`（默认 3）是最后一道防线

### Q3：`project_hook_input` 为什么要限制消息数量？

外部钩子（command/http/prompt）会收到完整的消息历史的**投影**。如果不限制：
- 整个对话历史可能有几万 token
- 通过 stdin 传给子进程，内存压力大
- 处理时间也会超过 timeout

所以限制为最近 20 条消息，每条内容截断到 2000 字符，这是一个合理的平衡点。

### Q4：为什么外部钩子失败要「降级为放行」而不是抛出异常？

Agent 是在执行用户的任务。钩子是辅助性质的（观测、验证），不是任务本身。如果一个用于记日志的钩子崩了，就让整个 run 失败，代价不对称。

```
钩子失败
  → 记日志（尽力而为）
  → 返回 HookOutcome()（等同于「没有意见，继续」）
  → run 照常进行
```

只有当一个钩子**明确返回 `block=True`** 时，run 才会被影响。

---

## 总结

这个提交的核心贡献是**填上了生命周期钩子的空洞**：

| 之前 | 之后 |
|------|------|
| `stop_hooks = []`，模型任意停止 | `StopCompletionHook` 检查待办，有未完成项则阻断 |
| `post_sampling_hooks = []`，采样完没有遥测 | `PostSamplingObserverHook` 写入 `runs/*.jsonl` |
| `pre/post_compact_hooks = []` | `CompactionLoggerHook` 记录压缩前后状态 |
| 想扩展钩子需要写 Python | 在 `agent.toml` 加 `[[hooks.external]]` 即可 |

所有这些都遵守同一条原则：**任何钩子的任何失败，都不能让 run 失败**。这是 Agent 框架健壮性的基本要求。
