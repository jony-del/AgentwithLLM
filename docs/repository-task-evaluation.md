# 真实仓库任务验收基线

2026-11-01：已提供离线评估与配对比较入口。当前没有用户提供的超过 20,000 文件的真实仓库
及模型配置，因此真实模型任务验收仍待执行。单元测试中的 fixture 只验证统计契约，不计入
百万行／千万行仓库验收。这里的结果与 `evaluation.py` 的运行时 completed 比例分开。

## 数据与职责

使用 `benchmarks/repository_tasks.py` 读取任务清单和独立评估者提供的观测记录。工具不下载仓库、
执行命令、启动模型或运行隐藏测试，也不会把输入中的字符串当作命令。每个 JSON 输入最多
8 MiB，每份清单最多 1,000 个任务；重复实验使用独立观测文件，避免重试覆盖先前失败。

清单格式（以下占位提交不是可验收的真实任务）：

```json
{
  "schema_version": 2,
  "tasks": [{
    "id": "example-1",
    "repository": "local-repository-identity",
    "base_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "problem": "自然语言需求，不包含标准补丁和测试答案",
    "acceptance_id": "independent-hidden-suite-v1",
    "category": "cross_file",
    "relevant_files": ["src/client.py", "src/protocol.py"],
    "relevant_symbols": [{"path": "src/client.py", "qualified": "client.connect", "kind": "function", "range": [40, 58]}],
    "edit_ranges": [{"path": "src/client.py", "start_line": 42, "end_line": 55}],
    "roles": {"src/client.py": "target", "src/protocol.py": "dependency"},
    "gold_sources": {"src/client.py": "maintainer-review"}
  }]
}
```

`schema_version` 1 仍可读取；v2 的新字段全部为可选附加标注。`relevant_symbols` 使用规范化
`(path, qualified, kind, range, content_version)` 身份，缺失的维度不参与匹配；跨工作区从不比较
`symbol_reference.id`。`edit_ranges` 是允许修改的闭区间行范围，`roles` 取值 `target/dependency/test`
且键必须属于 `relevant_files`。`category` 可为 `single_file`、`cross_file`、`cross_module`。
固定真实仓库提交、依赖和验收测试版本，并记录仓库源码行数、文件数及排除规则。`relevant_files`
及其余 gold 标注是评估者标签，必须与标准补丁、隐藏测试一起留在求解工作区之外；给 Agent 的输入
仅包含需求和允许公开的条件。现有 SWE-bench runner 的环境与隐藏测试隔离可复用，但尚未自动适配
这份通用任务清单。

观测格式：

```json
{
  "schema_version": 2,
  "manifest_sha256": "由 fingerprint(manifest) 计算",
  "environment": {
    "model": "精确模型和 provider 标识",
    "configuration_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "hardware": "CPU/RAM/磁盘/OS 标识"
  },
  "runs": [{
    "task_id": "example-1",
    "base_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "acceptance_id": "independent-hidden-suite-v1",
    "status": "finished",
    "retrieved_files": ["src/client.py"],
    "retrieved_candidates": [{"path": "src/client.py", "symbol": {"qualified": "client.connect", "kind": "function", "range": [40, 58]}}],
    "selected_candidates": [{"path": "src/client.py", "symbol": {"qualified": "client.connect", "kind": "function", "range": [40, 58]}}],
    "validated_candidates": [{"path": "src/client.py", "valid": true}],
    "patch_locations": [{"path": "src/client.py", "start_line": 44, "end_line": 50}],
    "patch_applied": true,
    "target_tests_passed": true,
    "hidden_tests_passed": null,
    "seconds": 30,
    "input_tokens": 1000,
    "output_tokens": 100,
    "tool_calls": 8,
    "llm_calls": 4
  }]
}
```

v2 分别记录检索返回候选、Agent 选中候选、候选机械校验、实际补丁位置与隐藏测试结果；未报告的
观测段落产生 null 指标，既不判成功也不判失败，只从该指标的均值分母中排除。编辑范围按一行起始
闭区间重叠匹配。缺失 gold 标注同样产生 null。unknown 成本仍为 null，所有失败状态留在分母。

计算绑定值时使用 `agent_core.benchmarks.repository.fingerprint(manifest)`，而非原始 JSON
文件字节哈希；配置指纹应覆盖 provider、模型、提示词、预算和依赖配置。比较要求相同清单与 K；
环境差异只允许在显式声明的处理变量（`--treatment model,hardware`）上出现，未声明的差异会失败；
配对比较输出逐任务 delta 与 improved/regressed 计数，`--unpaired` 退化为仅汇总费率差。
不要将秘密凭据放入记录或提示词。

任务只有在独立评估者报告 `hidden_tests_passed=true`，且状态为 finished、补丁已应用、
提供 `evaluated_patch_sha256` 时才计入 reported resolved；该 SHA-256 必须对应实际被测试的
补丁文件字节。该绑定只能防止意外混用，不能认证报告来源或检测评估者造假。

## 运行与解释

```powershell
.\.venv\Scripts\python.exe -B benchmarks/repository_tasks.py tasks.json after.json --before before.json --k 20 --output tmp/task-comparison.json
```

报告逐任务及汇总文件/符号/编辑范围 Recall@K、Precision@K、MRR、补丁应用率、目标测试通过率、
隐藏测试报告解决率和成本。Precision@K 分母固定为 K，重复路径只保留首次命中。没有观测的任务
标为 not_run 并留在成功率分母中；failed、timeout、environment_error 同样不从分母消失。
未提供的成本为 null，成本汇总同时标注 observed_tasks，不能将其解读成完整运行成本。

实际执行验收时，应依次记录检索结果、最终补丁身份、公开目标测试、独立隐藏测试和运行环境。
只运行公开测试或 Agent 声称 completed 都不能替代独立判定。首先覆盖三类任务，再固定至少
100 个经人工核对的真实任务进行配对重复实验。符号／修改位置准确率、查询 P50/P95、索引更新
与资源压力仍需专门测量；本工具未把任务耗时冒充检索延迟，也不自动推导这些指标。

当前后续顺序：真实仓库验收待输入。符号身份与解析器版本管理、LSP 位置证据（仅 definition/references/
implementation，覆盖始终标记 server coverage unknown）、最近清单级模块边界和候选机械校验已实现；
仍未实现的是动态检索路由、统一候选池与角色评估、结构化 document symbols 与有界调用层级、完整
workspace/local-package 静态依赖图、per-path 巨型事件消费，以及符号/编辑范围级 gold 指标
（见 implementation-plan-large-repo.md 的 S1–S6）。验证副本复用已实现，但每次租用仍做完整清单、
哈希、模式与权限复核（O(文件数+字节)，非 O(1)），其在真实文件分布和私有存储下的成本尚未测量；
当前没有改变独立可写副本和最终严格哈希认证的约束。

本批验证（2026-11-01）：离线评估、符号身份、检索、分页、证据压缩、探索恢复、规划器、
ReAct、核心契约和任务状态相关回归 174 项通过；离线 CLI 的空观测及配对比较冒烟通过。
Ruff 检查和 Mypy 的 219 个生产模块检查通过。没有执行真实仓库／模型任务。

## 性能对比记录规则（before/after）

任何性能对比（检索、索引、增量、快照、验证副本）必须同时满足：

1. 保留 before/after 两份原始 JSON，不允许只保留汇总结论；样本原始值必须包含在报告内。
2. 报告记录 `provenance`：`source_sha`（运行时代码的 git HEAD，取不到为 null 并标注）、
   `config_sha256`（规范化 JSON 的 SHA-256）、`random_seed`（无随机性的合成语料显式标为
   deterministic）和环境指纹（Python/平台/CPU）。`benchmarks/provenance.py` 提供统一实现。
3. 配对实验随机化并交替 before/after 执行顺序，使用同一清单、K 和环境记录；环境字段不完全
   一致时声明处理变量，不直接比较绝对值。
4. 失败分母规则：timeout、environment error、patch failure、test failure 全部计入分母；
   未知成本记 null 不填零；环境错误单独标注，不得伪装成通过或从统计中剔除。
5. 基准输出只报告实测值；理论可行性、推算指标与实测结果分开陈述。
