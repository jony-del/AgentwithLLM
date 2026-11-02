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
  "schema_version": 1,
  "tasks": [{
    "id": "example-1",
    "repository": "local-repository-identity",
    "base_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "problem": "自然语言需求，不包含标准补丁和测试答案",
    "acceptance_id": "independent-hidden-suite-v1",
    "category": "cross_file",
    "relevant_files": ["src/client.py", "src/protocol.py"]
  }]
}
```

`category` 可为 `single_file`、`cross_file`、`cross_module`。固定真实仓库提交、依赖和验收
测试版本，并记录仓库源码行数、文件数及排除规则。`relevant_files` 是评估者标签，必须与
标准补丁、隐藏测试一起留在求解工作区之外；给 Agent 的输入仅包含需求和允许公开的条件。
现有 SWE-bench runner 的环境与隐藏测试隔离可复用，但尚未自动适配这份通用任务清单。

观测格式：

```json
{
  "schema_version": 1,
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

计算绑定值时使用 `agent_core.benchmarks.repository.fingerprint(manifest)`，而非原始 JSON
文件字节哈希；配置指纹应覆盖 provider、模型、提示词、预算和依赖配置。两组配对实验必须
使用相同清单、K 和环境记录。不要将秘密凭据放入记录或提示词。

任务只有在独立评估者报告 `hidden_tests_passed=true`，且状态为 finished、补丁已应用、
提供 `evaluated_patch_sha256` 时才计入 reported resolved；该 SHA-256 必须对应实际被测试的
补丁文件字节。该绑定只能防止意外混用，不能认证报告来源或检测评估者造假。

## 运行与解释

```powershell
.\.venv\Scripts\python.exe -B benchmarks/repository_tasks.py tasks.json after.json --before before.json --k 20 --output tmp/task-comparison.json
```

报告逐任务及汇总文件 Recall@K、Precision@K、MRR、补丁应用率、目标测试通过率、隐藏测试
报告解决率和成本。Precision@K 分母固定为 K，重复路径只保留首次命中。没有观测的任务
标为 not_run 并留在成功率分母中；failed、timeout、environment_error 同样不从分母消失。
未提供的成本为 null，成本汇总同时标注 observed_tasks，不能将其解读成完整运行成本。

实际执行验收时，应依次记录检索结果、最终补丁身份、公开目标测试、独立隐藏测试和运行环境。
只运行公开测试或 Agent 声称 completed 都不能替代独立判定。首先覆盖三类任务，再固定至少
100 个经人工核对的真实任务进行配对重复实验。符号／修改位置准确率、查询 P50/P95、索引更新
与资源压力仍需专门测量；本工具未把任务耗时冒充检索延迟，也不自动推导这些指标。

当前后续顺序：真实仓库验收待输入；结构定位先完成符号身份和解析器版本管理，再整合 LSP、
真实模块边界及候选验证。验证副本复用仍须在真实文件分布和私有存储下测量后设计，当前没有
改变独立可写副本和最终严格哈希认证的约束。

本批验证（2026-11-01）：离线评估、符号身份、检索、分页、证据压缩、探索恢复、规划器、
ReAct、核心契约和任务状态相关回归 174 项通过；离线 CLI 的空观测及配对比较冒烟通过。
Ruff 检查和 Mypy 的 219 个生产模块检查通过。没有执行真实仓库／模型任务。
