# 沙箱机制 + 权限模式 + 细粒度权限控制：与 Open-ClaudeCode 对齐

本文档说明本次改动为什么做、做了什么、以及各层如何协作，帮助你完整理解新增的
**沙箱（enforcement layer）** 与 **细粒度权限（policy layer）** 体系。

---

## 1. 本次改动解决的问题

改动之前，本项目对危险操作只有两道很弱的约束：

1. `WorkspacePathMixin` 的**词法路径越界检查**——只拦文件工具的 `..`/绝对路径，
   拦不住 `run_command` 里的 `cd /` 或读 `~/.ssh`。
2. 基于 **静态 `ToolRisk`** 的模式化权限闸门——`decide(tool)` 只看工具*类别*
   （READ/WRITE/DANGEROUS），看不到*参数*，所以无法区分 `git status` 和 `rm -rf /`。

危险面集中在 `run_command` / `run_tests` → `_run_subprocess`：直接
`subprocess.run(..., env=os.environ, cwd=workspace)`，**零 OS 级隔离**，`cwd` 只是约定
而非边界，无网络限制、无命令白/黑名单。

参考项目 Open-ClaudeCode 把这件事拆成两层，本次改动照此移植：

| 层 | 作用 | 跨平台性 |
|----|------|----------|
| **策略层（policy）** | `ToolName(content)` 形式的 allow/deny/ask 规则 + 命令拆分/反绕过 + 路径/域匹配 | 全平台一致 |
| **强制层（enforcement）** | 用 `bwrap`(Linux) / `sandbox-exec`(macOS) 真隔离命令执行 | Windows 优雅降级为 no-op |

本项目以 **Windows + PowerShell** 为主。参考项目在 Windows 上**根本没有 OS 沙箱**，
直接禁用、回退到权限规则——本次改动采取同样策略：强制层架构就位、在 Linux/macOS 生效，
在 Windows 惰性透传，靠策略层兜底。

---

## 2. 代码地图（新增 / 修改）

**新增：**

- `agent_core/permission_rules.py` —— 策略层核心。规则解析、shell 命令
  exact/prefix/wildcard 匹配、复合命令拆分、反绕过（env-var / wrapper 剥离）、路径 glob
  与域名匹配。纯函数，失败即降级（drop 坏规则，不抛异常）。
- `agent_core/sandbox/` —— 强制层。
  - `config.py`：`SandboxConfig` + `SandboxNetworkConfig` / `SandboxFilesystemConfig`。
  - `manager.py`：`SandboxManager`（平台探测、依赖检查、`is_enabled`/`should_sandbox`/
    `wrap`/`unavailable_reason`、`fail_if_unavailable` 硬失败）+ 共享的 `NOOP_SANDBOX`。
  - `backends/`：`base.py`（`SandboxBackend` 协议 + `to_argv`/`expand_paths`）、
    `bubblewrap.py`（Linux）、`seatbelt.py`（macOS）、`noop.py`（Windows/降级）。
  - `__init__.py`：导出 + `SandboxAwareMixin`（给命令工具注入 manager 的 seam）。

**修改：**

- `agent_core/permissions.py`：新增 `PermissionMode.BYPASS`；`PermissionPolicy` 接收
  `rules` + `sandbox`；`decide(tool)` → `decide(tool, tool_call)` 参数感知，落地有序流水线。
- `agent_core/tools/executor.py`：`_prepare` 里 `decide(tool)` → `decide(tool, rewritten_call)`。
- `agent_core/tools/builtin.py`：`RunCommandTool` / `RunTestsTool` 加 `SandboxAwareMixin`，
  在 `_run_subprocess` 前 `self.sandbox.wrap(...)`。
- `agent_core/react.py`：`ReActConfig` 增 `sandbox` + `permission_rules`；构造期建
  `SandboxManager` 并 rebind 给命令工具，`PermissionPolicy(rules=..., sandbox=...)`。
- `agent_core/config.py`：`resolve_sandbox_config` + `resolve_permission_rules`。
- `agent_core/cli.py`：`--sandbox/--no-sandbox`、`--allow/--deny/--ask`，`--permission` 增
  `bypass`；`build_agent` 透传。
- `agent.toml.example`：新增 `[permissions]` 与 `[sandbox]`（含子表）。

---

## 3. 决策流水线（`PermissionPolicy.decide`）

对齐参考 `hasPermissionsToUseToolInner`，**首个命中即返回**。前四步*参数感知*，最后一步是
原来的粗粒度矩阵——所以**不配任何规则时行为与改动前完全一致**（`test_no_rules_preserves_legacy_behavior`）。

```
0. session_allow 短路（用户本会话“always”过的工具）
1. deny 规则命中           → 拒绝            （命令拆分：任一子命令命中即拒）
2. 敏感路径安全网          → ask（bypass 免疫）（写 .git/ .polaris/ agent.toml 等）
3. ask 规则命中            → ask            （除非第 4 步的沙箱耦合成立）
4. 沙箱耦合                → 允许            （命令确会被沙箱 + auto_allow… → OS 沙箱即边界）
5. bypass 模式             → 允许
6. allow 规则命中          → 允许            （命令拆分：须*每个*子命令都被覆盖）
7. 回退按模式的 ToolRisk 矩阵（default/acceptedits/plan/auto/dontask 原逻辑）
```

关键不对称（安全关键）：**deny 激进、allow 保守**。复合命令 `a && b | c`——
allow 需要*每个*子命令都被允许规则覆盖；deny 只要*任一*子命令命中即拒，且 deny 同时
匹配**原始**与**归一化**两种形态，wrapper/env-var 花招躲不掉。

---

## 4. 反绕过（policy 层的安全核心）

命令在 allow 匹配前会被归一化（`_normalize_subcommand`）：

- **剥离安全 env 前缀**：`LANG=C npm run test` → 匹配 `npm run test`。安全变量白名单
  `SAFE_ENV_VARS`。
- **绝不剥离劫持变量**：`PATH=` / `LD_*` / `DYLD_*`（`BINARY_HIJACK_VARS`）——留在命令里，
  于是它*不会*匹配朴素 allow，落到 ask/deny。这防止用环境变量偷换二进制/注入加载。
- **剥离安全 wrapper**：`timeout 300`、`nohup`、`nice`…，但遇到不认识的 flag（如
  `timeout -k$(id) 10`）立即停止剥离（fail safe）。

---

## 5. 强制层：`SandboxManager`

- **平台选择**：`darwin`→Seatbelt，`linux`→Bubblewrap，其它（`win32`）→Noop。
- **`is_enabled()`** = `config.enabled` 且 平台受支持 且 依赖就绪（`bwrap` 在 PATH 等）。
- **`should_sandbox(command)`** = `is_enabled()` 且 命令不在 `excluded_commands`。
- **`wrap(spec, shell, command=)`**：把命令前缀成 `bwrap …` / `sandbox-exec -p <profile> …`；
  不该沙箱时**原样返回**。shell 字符串会被规整成 `/bin/sh -c <cmd>`。
- **优雅降级**：不支持平台/缺依赖 → 全程透传；`unavailable_reason()` 仅在用户*显式*开启
  但跑不了时给出可操作提示。
- **`fail_if_unavailable`**：`enabled + fail_if_unavailable` 但当前环境跑不了 → 构造期
  抛 `SandboxUnavailableError`（宁可不启动也不静默裸奔）。

Bubblewrap 策略：`--ro-bind / /` 把整个文件系统挂只读，再把 workspace 与 `allow_write`
重绑为读写，`deny_read` 用空 tmpfs 遮蔽，默认 `--unshare-net` 断网。Seatbelt 生成 SBPL：
`(deny file-write*)` 后再 `allow` workspace，`deny network*` 等。

> 说明：这是参考项目 OS 层的**务实子集**——没有复刻 socat 网络代理与 seccomp 的
> `AF_UNIX` BPF 过滤（需要预编译二进制）；网络目前是「全断」或「共享主机网络」二选一。

---

## 6. 接线：命令工具怎么拿到沙箱

命令工具通过 **`SandboxAwareMixin`** 拿到 manager，模式与既有的 `SessionAwareMixin` 完全
一致：mixin 不定义 `__init__`（避免与 `WorkspacePathMixin` 的 `workspace` 参数冲突），
默认 `sandbox = NOOP_SANDBOX`（类属性），`ReActAgent.__init__` 的绑定循环里
`tool.bind_sandbox(self.sandbox)` 指向真 manager。子代理经 `replace(self.config, …)` 自动
继承 `sandbox` 与 `permission_rules`，故 deny 规则同样约束子代理 fan-out。

---

## 7. 配置界面

权限**模式**仍是顶层 `permission`（新增 `bypass`）。**规则**在 `[permissions]`：

```toml
[permissions]
allow = ["run_command(git *)", "read_text_file(/src/**)"]
deny  = ["run_command(rm *)", "web_fetch(domain:evil.example)"]
ask   = ["run_command"]
```

**沙箱**在 `[sandbox]` / `[sandbox.filesystem]` / `[sandbox.network]`，默认关闭。
env：`AGENT_SANDBOX`、`AGENT_SANDBOX_FAIL_IF_UNAVAILABLE`。CLI：`--sandbox/--no-sandbox`、
`--allow/--deny/--ask`（会 `merge` 到 toml 规则之上）。

---

## 8. 测试

- `tests/test_permission_rules.py`：解析、exact/prefix/wildcard、复合命令 all-vs-any、
  引号拆分、`SAFE_ENV_VARS` 剥离、`BINARY_HIJACK_VARS` 不剥离、wrapper 反绕过、路径/域匹配。
- `tests/test_sandbox.py`：平台选择、Windows/缺依赖降级透传、bwrap/seatbelt 前缀与 profile、
  `excluded_commands`、`fail_if_unavailable` 抛错/放行。
- `tests/test_permissions.py`（扩展）：deny/allow/deny-beats-allow、bypass 与其免疫项、
  敏感路径安全网、沙箱耦合开/关、无规则不回归。
- `tests/test_config.py`（扩展）：`[sandbox]`/`[permissions]` 解析与 env 覆盖、坏规则降级。
