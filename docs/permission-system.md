# Permission System Architecture

Polaris 的权限边界由三层组成：中央 `PermissionPolicy` 维护不可绕过的安全不变量；已迁移工具通过
`Tool.check_permissions(arguments, context)` 判断一次具体调用；`ToolRisk` 只为尚未迁移或返回
`PASSTHROUGH` 的工具提供保守 fallback。工具基类默认返回 `PASSTHROUGH`，不会默认放行。

## Permission contract

`PermissionResult` 的 `behavior` 只能是 `ALLOW`、`ASK`、`DENY` 或 `PASSTHROUGH`，并携带
`reason`、`decision_source`、可选的 `updated_arguments`/`metadata`、`classifier_approvable` 和
`bypass_immune`。`PermissionContext` 向工具提供当前 mode、workspace、interactive 状态、Sandbox
实际状态、带 provenance 的 rules、session grants、父子 Agent 信息、调用来源、Web domain policy、
managed policy 与 plan workflow 状态。

`updated_arguments` 会重新执行完整 preflight，工具不能通过参数改写绕过 schema、rules 或中央安全检查。

## 唯一决策顺序

每次调用严格按以下顺序求值，前一项已经给出终局结果时不会被后项放宽：

1. JSON Schema 验证与参数规范化。
2. managed policy 禁用的 mode 与 managed hard deny。
3. 普通显式 deny rule。
4. 中央不可绕过检查：父子权限 envelope、plan capability gate、workspace/path/symlink escape、秘密读取、
   受保护路径写入、blocked Web domain、SSRF、交互要求以及 unattended Sandbox 要求。
5. 显式 ask rule。
6. `tool.check_permissions()`；参数被改写时回到第 1 步。
7. 工具 `DENY`。
8. 工具 `ASK`。
9. 对确实将进入 Sandbox 且配置明确启用 auto-allow 的 `PASSTHROUGH` 调用执行 Sandbox fast path；
   `excluded_commands` 不适用。
10. `bypass` 只放行已经通过前置检查的剩余 `PASSTHROUGH` 调用。
11. session allow 与显式 allow rule。
12. 工具 `ALLOW`。
13. 对最终 `ASK` 应用 mode 语义。
14. 仍为 `PASSTHROUGH` 时才使用 `ToolRisk` fallback。

等价优先级为：`DENY > central non-bypassable checks > explicit ASK > tool ASK > ALLOW > PASSTHROUGH`。
显式 ask、工具 ask、`bypass_immune` 检查和 `requires_user_interaction` 均不能被 session always、
`acceptedits`、`auto` 或 `bypass` 洗白。

## Mode 状态表

| Mode | 明确安全的调用 | 最终 ASK | 未迁移的副作用工具 | 特殊语义 |
|---|---|---|---|---|
| `default` | 自动执行 | 先调用 `PermissionRequest` hook，再由 UI 确认；无结果且 headless 时拒绝 | ASK | UI 显示 `manual mode on` |
| `acceptedits` | 普通 workspace 文件编辑和工具确认安全的内部状态更新自动执行 | 同 `default` | ASK | 不会按 `ToolRisk.WRITE` 批量放行 Shell、测试、Skill、子 Agent 或外部副作用；显示 `accept edits on` |
| `plan` | 读取、搜索、Todo/Task 与专用 `write_plan` | 仅明确的 planning interaction 可询问 | DENY | 普通项目写入/命令/测试 centrally denied；显示 `plan mode on` |
| `auto` | 普通 workspace edit 与安全工具 fast path | 仅 `classifier_approvable=true` 的调用交给 `AutomatedPermissionEvaluator` | evaluator 评估 | evaluator 故障时顶层交互会话回退人工；headless/子 Agent fail-closed；显示 `auto mode on` |
| `dontask` | 保持 ALLOW | 全部转换为 DENY，不调用 `PermissionRequest` hook、不显示弹窗 | DENY | 已有 DENY 保持不变 |
| `bypass` | 保持 ALLOW | 保持 ASK | 仅通过所有前置检查的 `PASSTHROUGH` 直接 ALLOW | 不是 unconditional allow；managed policy 可禁用 |

`auto` evaluator 只能处理明确标记为 classifier-approvable 的剩余判断，不能推翻 hard deny、显式 deny、
显式 ask、秘密/受保护路径安全网或必须人工交互的工具。项目提供稳定 evaluator protocol 和 deterministic
Fake evaluator，未用 risk matrix 冒充 classifier。

交互式 chat 使用 `/permissions` 或 `/permissions <mode>` 切换六种 mode；Shift+Tab 只循环
`default → acceptedits → plan → auto`。

## Central safety invariants

路径检查使用 resolved path、`normcase` 和 `commonpath`，同时覆盖绝对路径、`..`、符号链接逃逸和
Windows 大小写差异。秘密读取安全网覆盖 `.env*`、`*.pem`、`*.key`、`*.p12`、`*.pfx`、
`id_rsa*`、`id_ed25519*`、`credentials*`、`.ssh`、`.aws`、`.gnupg`、`.kube`。写入安全网另外覆盖
`.git`、`.polaris`、`.claude`、`agent.toml`、`settings.json` 和 `settings.local.json`；写入
`.git/hooks` hard-deny。

`run_command` 在 rule/mode 放行前分解复合命令，并递归分析 `bash -c`、PowerShell 和 `cmd /c`
wrapper；任一危险子命令使整体 DENY。allow rule 必须覆盖全部子命令。`PATH`、`LD_*`、`DYLD_*`、
dynamic evaluation、download-to-shell、persistence、秘密路径和受保护路径不能借 wrapper 或 environment
assignment 获得 fast allow。

Web 工具对初始 URL 和每一跳 redirect 重做 scheme、DNS/IP、SSRF 与 domain policy 检查；
`blocked_domains` 永远拒绝，unattended mode 还要求 allowlist。子 Agent 的 mode 与实际 tool preset
都不能超过父 Agent；headless ASK 只能由 `PermissionRequest` hook 明确处理，否则拒绝。

## Planning workflow

进入 `plan` 时记录之前的 mode，并在每轮 model call 注入明确的 plan-mode system context。模型使用
`write_plan` 写入 agent-owned `~/.polaris/plans/<session>-<agent>.md` artifact（原子写入、256 KiB
上限），不能指定任意路径。`exit_plan` 需要非空 artifact 和人工确认；成功后恢复进入 plan 前的 mode。
工具执行器不会制造 `Dry-run: would execute...` 之类的虚假成功结果。

## Rule provenance 与 managed policy

每条 `PermissionRule` 在解析和 merge 后保留 `source`：`managed`、`user`、`project`、`local`、`cli`
或 `session`。默认配置依次加载用户级 `~/.polaris/agent.toml`、项目 `agent.toml` 和本机
`agent.local.toml`；CLI 与 session grant 使用各自 provenance。相同 specificity 下按来源 authority
确定匹配顺序。

`ManagedPolicyProvider` 是组织策略扩展点，支持 managed allow/deny/ask、禁用 mode、强制 unattended
Sandbox、managed-only allow 以及禁用持久化授权。managed allow 只能参与普通 allow 阶段，不能放宽中央安全不变量。

## 审计与秘密处理

每个权限事件以 JSONL 写入，至少包括 `schema_version`、`tool`、安全的 arguments 摘要、`mode`、
`final_behavior`、`reason`、`decision_source`、`matched_rule`、`rule_source`、`sandboxed`、
`classifier_result`、`parent_agent_id` 和 `tool_source`。命令、patch、content 与 tool result 只保存长度和
SHA-256 摘要；URL 移除 userinfo/query；password、token、Authorization、credential、secret 与 PEM
material 会统一 redaction。transcript 和通用 run logger 使用同一 sanitizer，秘密文件内容不会落盘。

## 工业化兼容与授权持久化

权限模式的规范输出仍为 `default`、`acceptedits`、`plan`、`auto`、`dontask`、`bypass`。CLI、配置文件、
环境变量和 API 同时接受参考项目的 `acceptEdits`、`dontAsk`、`bypassPermissions` 别名，并立即规范化，
避免审计和配置中出现两套状态名。

交互授权默认只允许一次。用户也可以把系统建议的最小作用域规则授权到当前 session，或持久化到：

- `agent.local.toml`（仅本机项目）；
- `agent.toml`（共享项目，写入后同步更新当前用户的 TOFU 指纹）；
- `~/.polaris/agent.toml`（当前用户）。

Shell 建议按规范化后的每条子命令分别授权，复合命令必须全部被覆盖；不会生成无边界的
`run_command` grant。持久化操作需要二次确认，使用进程间 lock、临时文件、`fsync` 和原子替换；
失败时既不执行工具，也不遗留内存 grant。安装依赖可用时由 `tomlkit` 保留原文件注释和格式。

`auto` 分类器明确返回 block 时仍为 hard deny。只有分类器不可用、超时、异常或输出无效时，顶层
交互会话才恢复原始 ASK 并交给 `PermissionRequest` hook/UI；headless、sub-agent 和 team 调用全部
fail closed。取消信号不会被转换成授权拒绝，而是继续向上传播。

`exit_plan` 可携带最多 32 条 `{rule, reason}` 形式的 `requested_permissions`。规则必须指向已注册
工具并具有精确作用域；blanket、wildcard、动态 Shell、破坏性/持久化命令、秘密或受保护路径均被
中央策略拒绝。终端逐条审阅，用户可只批准子集；批准的规则只在恢复后的当前 session 生效。

## 系统托管策略文件

默认只读策略位置为 Windows `%ProgramData%\Polaris\managed-policy.toml`、Linux
`/etc/polaris/managed-policy.toml`、macOS `/Library/Application Support/Polaris/managed-policy.toml`。
可用 `POLARIS_MANAGED_POLICY_PATH` 指定部署路径。格式如下：

```toml
[managed.permissions]
allow = ["run_command(git status)"]
deny = ["run_command(rm *)"]
ask = ["web_fetch(domain:github.com)"]
forbidden_modes = ["bypassPermissions"]
require_sandbox_for_unattended = true
allow_managed_rules_only = false
disable_persistent_grants = false
```

隐式默认文件缺失等同空策略；显式路径缺失或现有文件格式错误会让启动失败。运行中每个工具授权
边界都会热重载；已成功加载后若文件损坏，该调用直接拒绝并写审计，不会继续沿用可能过宽的旧策略。
managed allow 仍不能覆盖 schema、deny/ask、plan gate、敏感路径或其他中央安全不变量。

权限审计 schema v2 额外记录 mode alias 映射、original/final behavior、auto fallback/failure kind、
授权更新与 destination、是否持久化、managed policy digest 以及 plan 授权包计数；v1 的核心字段继续保留，
旧 replay reader 可以忽略新增字段。
