"""Runtime gate benchmarks (SECOND_ROUND_INDEPENDENT_AUDIT §11 phase 5).

Five timed scenarios, each executed in its own timeout-bounded subprocess so a
hang or a semantic regression can never wedge the nightly gate:

- cancellation_unwind  cancel a run parked mid provider read; time the unwind
                       and verify the interrupt result, gate release and task
                       cleanup.
- resume_load          load a large transcript and rebuild the message chain.
- resume_restore       same load plus ``resume_loaded_session()``; verifies the
                       runtime actually swaps onto the loaded session.
- recovery_scan        startup recovery preview scan over terminal journals.
- recovery_apply       real recovery over freshly prepared recoverable
                       journals, verified idempotent on a second pass.

Run from the repository root:
    python benchmarks/runtime_gates.py                       # 20 samples/scenario
    python benchmarks/runtime_gates.py --check benchmarks/thresholds.json
    python benchmarks/runtime_gates.py --json-out gates.json

Every scenario warms up once, then takes ``--runs`` timed samples and reports a
nearest-rank p95. ``--check`` is strict: a missing metric, an invalid or
non-finite threshold, fewer than 20 samples, a scenario timeout, or a semantic
assertion failure all fail the gate (there is no "no threshold, skip" path) and
the partial report is still written. Thresholds are calibrated per OS
(benchmarks/thresholds.json, version 2) from the maximum p95 of three
independent runner executions, doubled and rounded up to 0.1 s.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_core.execution import CancellationToken, ExecutionScope
from agent_core.memory import MemoryConfig
from agent_core.models import LLMResult, Message
from agent_core.providers.base import ProviderGate, provider_attempt
from agent_core.react import ReActAgent, ReActConfig
from agent_core.tools.transaction import (
    JournalStorage,
    RecoveryState,
    TurnExecutionJournal,
)
from agent_core.transcript import SCHEMA_VERSION, build_chain, load_transcript

REPORT_VERSION = 2
DEFAULT_RUNS = 20
MIN_SAMPLES_FOR_CHECK = 20
WARMUP_RUNS = 1
DEFAULT_SCENARIO_TIMEOUT = 900.0
SCENARIO_NAMES = (
    "cancellation_unwind",
    "resume_load",
    "resume_restore",
    "recovery_scan",
    "recovery_apply",
)
# Threshold file keys are the Python platform families the nightly gates run on.
PLATFORM_KEYS = {"linux": "linux", "win32": "win32", "darwin": "darwin"}

# Same opt-out as tests/conftest.py: the benchmark harness has no sandbox backend,
# and the D3 rule (unattended permission modes require one) is enforced separately
# by tests/test_sandbox_required.py.
os.environ.setdefault("AGENT_SANDBOX_ALLOW_UNATTENDED", "1")


def _platform_key() -> str:
    key = PLATFORM_KEYS.get(sys.platform)
    if key is None:
        raise SystemExit(
            f"no thresholds are calibrated for sys.platform={sys.platform!r}; "
            f"calibrated platforms: {sorted(set(PLATFORM_KEYS.values()))}"
        )
    return key


def _nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


# --- scenario implementations (child process) ----------------------------------


def _prepare_roots(tag: str) -> tuple[Path, Path]:
    """Fresh workspace plus a user-state directory kept outside of it."""

    base = Path(tempfile.mkdtemp(prefix=f"polaris-gate-{tag}-"))
    workspace = base / "workspace"
    workspace.mkdir()
    state = base / "user-state"
    # 0o700 keeps the state root owner-only: a default mkdir inherits the temp
    # directory's DACL, which on some hosts grants write to foreign principals
    # and is rightly rejected by the recovery-state security checks.
    state.mkdir(mode=0o700)
    os.environ["POLARIS_HOME"] = str(state)
    os.environ["AGENT_TRUST_STORE"] = str(state / "trusted.json")
    return workspace, state


class _ParkedReadProvider:
    """Provider parked inside a scope-aware in-flight read until cancelled.

    Mirrors tests/test_cancellation_matrix.py's hung-read fixture: the only exit
    is the run scope's cancellation token waking the bounded ``scope.sleep``.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def complete(self, messages, tools, config, stream=None, should_cancel=None, scope=None):
        assert scope is not None
        async with provider_attempt(scope):
            self.entered.set()
            await scope.sleep(60)
        return LLMResult("unreachable", stop_reason="end")


async def _cancellation_sample(workspace: Path) -> tuple[float, list[str]]:
    provider = _ParkedReadProvider()
    agent = ReActAgent(provider, _bench_config(workspace))
    token = CancellationToken()
    gate = ProviderGate(max_concurrency=1)
    scope = ExecutionScope.for_workspace(
        workspace, cancellation=token, provider_gate=gate
    )
    try:
        run_task = asyncio.create_task(agent.run("hang on a read", execution_scope=scope))
        try:
            await provider.entered.wait()
            # Timed region: cancel -> run returned. Startup is excluded.
            started = time.perf_counter()
            token.cancel("benchmark interrupt")
            result = await run_task
            elapsed = time.perf_counter() - started
        finally:
            if not run_task.done():
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
    finally:
        await scope.close()
        # Release the held append handle before TemporaryDirectory cleanup on
        # Windows, where an open file blocks directory removal.
        agent.logger.close()

    checks: list[str] = []
    assert token.cancelled(), "cancellation token was never set"
    assert result.answer.startswith("Stopped after"), (
        f"interrupt produced a non-interrupt result: {result.answer!r}"
    )
    checks.append("interrupt_result")
    semaphore, _bucket = gate._ensure()
    assert semaphore._value == gate.max_concurrency, (
        f"provider gate leaked {gate.max_concurrency - semaphore._value} permit(s)"
    )
    checks.append("gate_released")
    leftovers = [
        task for task in asyncio.all_tasks() if task is not asyncio.current_task()
    ]
    assert not leftovers, f"run leaked tasks: {[task.get_name() for task in leftovers]}"
    checks.append("tasks_cleaned_up")
    return elapsed, checks


def _bench_config(workspace: Path, *, session_dir: str = "") -> ReActConfig:
    return ReActConfig(
        run_dir=str(workspace / "runs"),
        session_dir=session_dir,
        permission="auto",
        memory=MemoryConfig(enabled=False),
        project_instructions=False,
        git_context=False,
        max_api_concurrency=1,
    )


def _build_transcript(path: Path, messages: int, session_id: str, workspace: Path) -> None:
    """Write a schema-faithful session JSONL directly (fsync-per-message append via
    TranscriptStore would dominate prep time; the gate measures loading, not appending)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "session",
                    "v": SCHEMA_VERSION,
                    "session_id": session_id,
                    "cwd": str(workspace),
                    "project_id": "benchmark",
                    "ts": now,
                }
            )
            + "\n"
        )
        parent: str | None = None
        for index in range(messages):
            message = Message(
                "user" if index % 2 == 0 else "assistant",
                f"benchmark message {index} " + "x" * 64,
                parent_uuid=parent,
            )
            record = {
                "type": "message",
                "v": SCHEMA_VERSION,
                **message.to_dict(),
                "session_id": session_id,
                "cwd": str(workspace),
                "git_branch": None,
                "ts": now + index,
            }
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            parent = message.uuid


def _resume_load_sample(transcript: Path, messages: int) -> list[str]:
    loaded = load_transcript(transcript)
    chain = build_chain(loaded)
    assert len(chain) == messages, f"resume chain broke: {len(chain)} != {messages}"
    previous: str | None = None
    for message in chain:
        assert message.parent_uuid == previous, "resume chain parent link broke"
        previous = message.uuid
    return ["chain_length", "parent_links"]


async def _resume_restore_scenario(
    workspace: Path, state: Path, *, messages: int, runs: int
) -> dict:
    from agent_core.providers.fake import FakeProvider

    transcript = workspace / "transcripts" / "bench-project" / "restore.jsonl"
    _build_transcript(transcript, messages, "bench-restore", workspace)
    agent = ReActAgent(
        FakeProvider(),
        _bench_config(workspace, session_dir=str(state / "projects")),
        workspace=workspace,
        session_id="bench-fresh",
    )
    samples: list[float] = []
    checks: list[str] = []
    try:
        for index in range(WARMUP_RUNS + runs):
            started = time.perf_counter()
            loaded = load_transcript(transcript)
            await agent.resume_loaded_session(loaded)
            elapsed = time.perf_counter() - started
            assert agent.session_id == loaded.session_id, "runtime did not swap sessions"
            assert agent.transcript is not None and (
                agent.transcript.session_id == loaded.session_id
            ), "runtime transcript did not follow the loaded session"
            if index >= WARMUP_RUNS:
                samples.append(elapsed)
        checks = ["session_swapped", "transcript_swapped"]
    finally:
        agent.logger.close()
    return {"samples": samples, "checks": checks}


def _seed_terminal_journals(storage: JournalStorage, count: int) -> None:
    for _ in range(count):
        journal = TurnExecutionJournal(storage, uuid.uuid4().hex)
        journal.close()


def _seed_recoverable_journals(storage: JournalStorage, count: int) -> None:
    """Abandon journals mid external tool call, exactly like a crashed process.

    Each records ADMITTED then EXTERNAL_INTENT and releases ownership without a
    terminal state, so real recovery must classify them (indeterminate external
    effect) without ever replaying the tool.
    """

    for index in range(count):
        journal = TurnExecutionJournal(storage, uuid.uuid4().hex)
        journal.record(RecoveryState.ADMITTED, tool="bench_tool", args={"n": index})
        journal.record(
            RecoveryState.EXTERNAL_INTENT,
            tool="bench_tool",
            args={"n": index},
            changed=[f"out-{index}.txt"],
        )
        journal._release_for_later_recovery()


async def _run_child_scenario(
    name: str, *, runs: int, messages: int, journals: int, recoverable: int
) -> dict:
    if os.getenv("RUNTIME_GATES_SELFTEST") == "hang":
        time.sleep(120)  # Self-test hook: force a parent-side scenario timeout.

    if name == "cancellation_unwind":
        workspace, _state = _prepare_roots("cancel")
        try:
            samples: list[float] = []
            checks: list[str] = []
            for index in range(WARMUP_RUNS + runs):
                elapsed, sample_checks = await _cancellation_sample(workspace)
                if index >= WARMUP_RUNS:
                    samples.append(elapsed)
                checks = sample_checks
        finally:
            shutil.rmtree(workspace.parent, ignore_errors=True)
        return {"samples": samples, "checks": checks, "params": {}}

    if name in {"resume_load", "resume_restore"}:
        workspace, state = _prepare_roots(name)
        try:
            if name == "resume_load":
                transcript = workspace / "transcripts" / "bench-project" / "resume.jsonl"
                _build_transcript(transcript, messages, "bench-resume", workspace)
                samples = []
                checks: list[str] = []
                for index in range(WARMUP_RUNS + runs):
                    started = time.perf_counter()
                    sample_checks = _resume_load_sample(transcript, messages)
                    elapsed = time.perf_counter() - started
                    checks = sample_checks
                    if index >= WARMUP_RUNS:
                        samples.append(elapsed)
                return {
                    "samples": samples,
                    "checks": checks,
                    "params": {"messages": messages},
                }
            result = await _resume_restore_scenario(
                workspace, state, messages=messages, runs=runs
            )
            result["params"] = {"messages": messages}
            return result
        finally:
            shutil.rmtree(workspace.parent, ignore_errors=True)

    if name == "recovery_scan":
        workspace, _state = _prepare_roots("scan")
        try:
            (workspace / "ws").mkdir()
            storage = JournalStorage.local(
                workspace / "journals",
                workspace=workspace / "ws",
                session_id="bench-recovery",
                run_id="bench-run",
            )
            _seed_terminal_journals(storage, journals)
            samples = []
            for index in range(WARMUP_RUNS + runs):
                started = time.perf_counter()
                outcomes = TurnExecutionJournal.recover_all(storage)
                if index >= WARMUP_RUNS:
                    samples.append(time.perf_counter() - started)
            assert outcomes == [], "terminal journals must yield no recovery outcomes"
            return {"samples": samples, "checks": ["no_outcomes"], "params": {"journals": journals}}
        finally:
            shutil.rmtree(workspace.parent, ignore_errors=True)

    if name == "recovery_apply":
        workspace, _state = _prepare_roots("apply")
        try:
            samples = []
            outcomes = None
            for index in range(WARMUP_RUNS + runs):
                # Fresh, never-recovered journals every sample: recovery work must
                # not be amortized across samples by an empty second pass.
                root = workspace / f"sample-{index}" / "journals"
                storage = JournalStorage.local(
                    root,
                    workspace=workspace / "ws",
                    session_id="bench-apply",
                    run_id="bench-run",
                )
                _seed_recoverable_journals(storage, recoverable)
                started = time.perf_counter()
                outcomes = TurnExecutionJournal.recover_all(storage, dry_run=False)
                elapsed = time.perf_counter() - started
                again = TurnExecutionJournal.recover_all(storage, dry_run=False)
                assert outcomes == again, "recovery apply is not idempotent"
                assert len(outcomes) == recoverable, (
                    f"expected {recoverable} recovery outcomes, got {len(outcomes)}"
                )
                if index >= WARMUP_RUNS:
                    samples.append(elapsed)
                previous = workspace / f"sample-{index - 1}"
                if index > 0:
                    shutil.rmtree(previous, ignore_errors=True)
            return {
                "samples": samples,
                "checks": ["outcome_count", "idempotent"],
                "params": {"recoverable": recoverable},
            }
        finally:
            shutil.rmtree(workspace.parent, ignore_errors=True)

    raise SystemExit(f"unknown scenario {name!r}")


# --- parent orchestration -------------------------------------------------------


def _dependency_snapshot() -> dict[str, str | None]:
    snapshot: dict[str, str | None] = {}
    for distribution in ("agent-with-llm", "mcp", "httpx", "aiohttp"):
        try:
            snapshot[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            snapshot[distribution] = None
    return snapshot


def _child_command(arguments: argparse.Namespace, name: str) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        name,
        "--runs",
        str(arguments.runs),
        "--messages",
        str(arguments.messages),
        "--journals",
        str(arguments.journals),
        "--recoverable",
        str(arguments.recoverable),
    ]


def _run_scenario(
    arguments: argparse.Namespace, name: str
) -> tuple[dict | None, dict | None]:
    command = _child_command(arguments, name)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=arguments.scenario_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, {
            "scenario": name,
            "reason": (
                f"scenario exceeded the {arguments.scenario_timeout:.0f}s timeout "
                "and was terminated"
            ),
        }
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()[-6:]
        return None, {
            "scenario": name,
            "reason": (
                f"scenario process exited rc={completed.returncode}; "
                f"stderr tail: {' | '.join(detail) or '<empty>'}"
            ),
        }
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        return None, {
            "scenario": name,
            "reason": f"scenario printed unreadable result JSON: {exc}",
        }
    samples = payload.get("samples") or []
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in samples):
        return None, {"scenario": name, "reason": "scenario reported non-finite samples"}
    payload["p50_seconds"] = statistics.median(samples) if samples else None
    payload["p95_seconds"] = _nearest_rank_p95(samples) if samples else None
    return payload, None


def _load_thresholds(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"threshold file {path} is unreadable: {exc}")
    if not isinstance(config, dict) or config.get("version") != 2:
        raise SystemExit(f"threshold file {path} must carry {'version': 2} (per-OS format)")
    platforms = config.get("platforms")
    if not isinstance(platforms, dict) or not isinstance(platforms.get(_platform_key()), dict):
        raise SystemExit(
            f"threshold file {path} has no thresholds calibrated for {_platform_key()!r}"
        )
    thresholds = platforms[_platform_key()]
    for name in SCENARIO_NAMES:
        limit = thresholds.get(f"{name}_p95_seconds")
        if limit is None:
            raise SystemExit(
                f"threshold file {path} is missing {_platform_key()}/{name}_p95_seconds; "
                "a missing metric is a gate failure, not a skip — calibrate and add it"
            )
        if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
            raise SystemExit(
                f"threshold {_platform_key()}/{name}_p95_seconds={limit!r} is invalid "
                "(must be a finite positive number of seconds)"
            )
    return thresholds


async def _main(arguments: argparse.Namespace) -> int:
    if arguments.child:
        result = await _run_child_scenario(
            arguments.child,
            runs=arguments.runs,
            messages=arguments.messages,
            journals=arguments.journals,
            recoverable=arguments.recoverable,
        )
        # Single machine-readable line on stdout; everything else goes to stderr.
        print(json.dumps(result))
        if os.getenv("RUNTIME_GATES_SELFTEST") == "fail-semantics":
            raise AssertionError("injected semantic failure (runtime-gates self-test)")
        return 0

    selected = list(arguments.only or ()) or list(SCENARIO_NAMES)
    unknown = [name for name in selected if name not in SCENARIO_NAMES]
    if unknown:
        raise SystemExit(f"unknown scenario(s) {unknown}; known: {list(SCENARIO_NAMES)}")

    scenarios: dict[str, dict] = {}
    failures: list[dict] = []
    for name in selected:
        payload, failure = _run_scenario(arguments, name)
        if failure is not None:
            failures.append(failure)
            scenarios[name] = {"status": "failed", **failure}
        else:
            scenarios[name] = {"status": "pass", **payload}

    report = {
        "version": REPORT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "platform": sys.platform,
        "platform_key": _platform_key(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "dependencies": _dependency_snapshot(),
        "parameters": {
            "runs": arguments.runs,
            "warmup": WARMUP_RUNS,
            "messages": arguments.messages,
            "journals": arguments.journals,
            "recoverable": arguments.recoverable,
            "scenario_timeout_seconds": arguments.scenario_timeout,
        },
        "scenarios": scenarios,
        "failures": failures,
    }

    print(f"{'metric':<22} {'p50 ms':>10} {'p95 ms':>10}  checks")
    print("-" * 78)
    for name in selected:
        entry = scenarios[name]
        if entry["status"] != "pass":
            print(f"{name:<22} {'-':>10} {'-':>10}  FAILED: {entry['reason']}")
            continue
        print(
            f"{name:<22} "
            f"{(entry['p50_seconds'] or 0) * 1000:>10.1f} "
            f"{entry['p95_seconds'] * 1000:>10.1f}  {', '.join(entry.get('checks', []))}"
        )

    if arguments.json_out:
        Path(arguments.json_out).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"\nreport written to {arguments.json_out}")

    exit_code = 1 if failures else 0
    if arguments.check:
        thresholds = _load_thresholds(Path(arguments.check))
        if arguments.runs < MIN_SAMPLES_FOR_CHECK:
            print(
                f"FAIL sample count: --runs={arguments.runs} is below the required "
                f"{MIN_SAMPLES_FOR_CHECK}; p95 would not be meaningful"
            )
            return 1
        for name in selected:
            entry = scenarios[name]
            limit = thresholds[f"{name}_p95_seconds"]
            if entry["status"] != "pass":
                print(f"FAIL {name}: scenario did not produce a measurement")
                continue
            ok = entry["p95_seconds"] <= limit
            exit_code = exit_code or (0 if ok else 1)
            print(
                f"{'PASS' if ok else 'FAIL'} {name}: p95 {entry['p95_seconds']:.3f}s "
                f"<= {limit:.3f}s ({_platform_key()} threshold)"
            )
    return exit_code


def _parse(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="timed samples per scenario")
    parser.add_argument("--messages", type=int, default=2000, help="resume transcript size")
    parser.add_argument("--journals", type=int, default=200, help="terminal journals to scan")
    parser.add_argument(
        "--recoverable", type=int, default=10, help="recoverable journals per apply sample"
    )
    parser.add_argument(
        "--scenario-timeout",
        type=float,
        default=DEFAULT_SCENARIO_TIMEOUT,
        help="per-scenario subprocess timeout in seconds",
    )
    parser.add_argument("--check", metavar="THRESHOLDS", help="fail nonzero on p95 regression")
    parser.add_argument("--json-out", metavar="PATH", help="write the raw measurement report")
    parser.add_argument(
        "--only",
        action="append",
        choices=SCENARIO_NAMES,
        help="restrict to one scenario (repeatable)",
    )
    parser.add_argument("--_child", dest="child", metavar="SCENARIO", help=argparse.SUPPRESS)
    return parser.parse_args(arguments)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse(sys.argv[1:] if argv is None else argv)
    return asyncio.run(_main(arguments))


if __name__ == "__main__":
    sys.exit(main())
