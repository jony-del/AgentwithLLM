"""End-to-end SWE-bench solve/evaluate orchestration."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import importlib.metadata
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from agent_core.capabilities import CapabilitiesConfig
from agent_core.compression import CompressionConfig
from agent_core.hooks import HooksConfig
from agent_core.memory import MemoryConfig
from agent_core.permissions import PermissionMode
from agent_core.providers import ClaudeProvider, FakeProvider, OpenAICompatProvider, OpenAIResponsesProvider
from agent_core.react import ReActAgent, ReActConfig
from agent_core.sandbox import SandboxConfig
from agent_core.skills import SkillsConfig
from agent_core.storage import JSONLRunLogger
from agent_core.tool_config import ToolSuiteConfig
from agent_core.ui import AgentUI, NullUI

from .dataset import SWEbenchDataset
from .harness import HarnessResult, OfficialHarnessAdapter
from .models import FailureKind, InstanceState, SWEbenchInstance, SWEbenchRunConfig, SWEbenchSelection
from .patch import export_patch, prediction_record, save_patch, upsert_prediction, validate_patch_text
from .prompt import BENCHMARK_SYSTEM_PROMPT, build_swebench_prompt
from .runtime import DockerRuntime, InstanceRuntime, LocalRuntime, prepare_repository, resolve_official_image
from .selection import write_selection
from .tools import build_swebench_registry
from .ui import BenchmarkRecordingUI


@dataclass(slots=True)
class InstanceResult:
    instance_id: str
    state: str
    failure_kind: str | None = None
    failure_reason: str | None = None
    patch_path: str | None = None
    patch_generated: bool = False
    changed_files: tuple[str, ...] = ()
    answer: str = ""
    prepare_seconds: float = 0.0
    solve_seconds: float = 0.0
    patch_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    total_seconds: float = 0.0
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SWEbenchRunner:
    """Run the same Agent loop for selected tasks or the complete Lite split."""

    def __init__(
        self,
        config: SWEbenchRunConfig,
        *,
        provider: Any | None = None,
        runtime_factory: Callable[[SWEbenchInstance, Path, str | None], InstanceRuntime] | None = None,
        ui_factory: Callable[[], AgentUI] | None = None,
        dataset_loader: Callable[..., SWEbenchDataset] = SWEbenchDataset.load,
    ) -> None:
        self.config = config
        self.provider = provider
        self.runtime_factory = runtime_factory
        self.ui_factory = ui_factory or (lambda: NullUI())
        self.dataset_loader = dataset_loader
        self._prediction_lock = asyncio.Lock()

    async def run(self, selection: SWEbenchSelection, *, all_instances: bool = False, limit: int | None = None) -> dict[str, Any]:
        started = time.monotonic()
        if self.config.evaluate and not OfficialHarnessAdapter().available():
            raise RuntimeError(
                "the official SWE-bench Harness is unavailable; install the benchmark extras "
                "with `pip install -e '.[swebench]'`, or pass --no-evaluate"
            )
        load_kwargs: dict[str, Any] = {"include_gold": False}
        cache_dir = self.config.metadata.get("cache_dir")
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        dataset = self.dataset_loader(selection.dataset, selection.split, **load_kwargs)
        if selection.instance_ids:
            instances = dataset.select(selection.instance_ids)
        elif all_instances:
            instances = tuple(dataset)
        else:
            instances = tuple(dataset)[: limit or 0]
        if limit is not None:
            instances = instances[:limit]
        if not instances:
            raise ValueError("selection resolved to zero SWE-bench instances")
        if self.config.run_id:
            run_id = _safe_run_id(self.config.run_id)
        else:
            run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        run_dir = Path(self.config.output_dir).expanduser().resolve() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.config.run_id = run_id
        self._write_run_metadata(run_dir, selection, instances)
        predictions_path = run_dir / "predictions.jsonl"
        instance_rows = run_dir / "instances.jsonl"
        instance_rows.write_text("".join(json.dumps(item.public_dict(), ensure_ascii=False) + "\n" for item in instances), encoding="utf-8")
        write_selection(run_dir / "selection.yaml", SWEbenchSelection(selection.dataset, selection.split, tuple(item.instance_id for item in instances), source=selection.source))

        semaphore = asyncio.Semaphore(max(1, int(self.config.solve_workers)))

        async def solve_one(instance: SWEbenchInstance) -> InstanceResult:
            async with semaphore:
                return await self._solve_instance(instance, run_dir, predictions_path)

        results = list(await asyncio.gather(*(solve_one(item) for item in instances)))
        await self._ensure_predictions(run_dir, predictions_path, results)
        harness_result: HarnessResult | None = None
        if self.config.evaluate:
            patch_ids = [item.instance_id for item in results if item.state == InstanceState.PATCH_READY.value]
            if patch_ids:
                for item in results:
                    if item.instance_id in patch_ids:
                        state_path = run_dir / "instances" / _safe_instance_dir(item.instance_id) / "state.json"
                        self._update_state(state_path, state=InstanceState.EVALUATION_PENDING.value)
                        self._update_state(state_path, state=InstanceState.EVALUATING.value)
                evaluator = OfficialHarnessAdapter()
                eval_started = time.monotonic()
                harness_result = await asyncio.to_thread(
                    evaluator.evaluate,
                    dataset=selection.dataset,
                    split=selection.split,
                    predictions_path=predictions_path,
                    run_id=f"{run_id}-{_prediction_digest(predictions_path)[:12]}",
                    report_dir=run_dir / "harness",
                    instance_ids=patch_ids,
                    max_workers=self.config.evaluation_workers,
                    timeout=(
                        int(max(1.0, self.config.max_wall_seconds))
                        if self.config.max_wall_seconds > 0
                        else 1800
                    ),
                    cwd=run_dir,
                )
                elapsed = time.monotonic() - eval_started
                self._apply_harness_result(results, run_dir, harness_result, elapsed)
        summary = aggregate_results(results, harness_result=harness_result, selected=len(instances), duration=time.monotonic() - started)
        summary["run_id"] = run_id
        summary["run_dir"] = str(run_dir)
        self._write_summary(run_dir, summary)
        return summary

    async def _solve_instance(self, instance: SWEbenchInstance, run_dir: Path, predictions_path: Path) -> InstanceResult:
        started = time.monotonic()
        instance_dir = run_dir / "instances" / _safe_instance_dir(instance.instance_id)
        workspace = instance_dir / "workspace"
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "public_task.json").write_text(json.dumps(instance.public_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        state_path = instance_dir / "state.json"
        old = _read_json(state_path)
        if self.config.resume and old and old.get("state") in {InstanceState.PATCH_READY.value, InstanceState.RESOLVED.value, InstanceState.UNRESOLVED.value}:
            return InstanceResult(**{key: value for key, value in old.items() if key in InstanceResult.__dataclass_fields__})
        if self.config.resume and old and old.get("state") == InstanceState.FAILED.value and not self.config.retry_failed:
            return InstanceResult(**{key: value for key, value in old.items() if key in InstanceResult.__dataclass_fields__})
        result = InstanceResult(instance.instance_id, InstanceState.PENDING.value)
        self._update_state(state_path, instance_id=instance.instance_id, state=InstanceState.PREPARING.value)
        prepare_started = time.monotonic()
        runtime: InstanceRuntime | None = None
        agent: ReActAgent | None = None
        logger: JSONLRunLogger | None = None
        try:
            source_dir = self.config.metadata.get("source_dir")
            prepare_repository(instance, workspace, source_dir=source_dir, force=not self.config.resume)
            result.prepare_seconds = time.monotonic() - prepare_started
            self._update_state(state_path, state=InstanceState.SOLVING.value, prepare_seconds=result.prepare_seconds)
            image = resolve_official_image(instance, self.config.image)
            if self.runtime_factory is not None:
                runtime = self.runtime_factory(instance, workspace, image)
            elif self.config.metadata.get("runtime") == "local":
                runtime = LocalRuntime(workspace)
            else:
                if not image:
                    raise RuntimeError(
                        "could not resolve the official SWE-bench instance image. Install the "
                        "swebench package or pass --image/ SWEBENCH_INSTANCE_IMAGE."
                    )
                runtime = DockerRuntime(workspace, image, name=f"polaris-swebench-{_safe_instance_dir(instance.instance_id)}-{self.config.run_id[:8]}")
            await runtime.start()
            registry = build_swebench_registry(runtime, str(workspace), test_command=self.config.test_command)
            logger = JSONLRunLogger(instance_dir / "agent_logs")
            recording_ui = BenchmarkRecordingUI(self.ui_factory())
            provider = self.provider or _provider_from_name(self.config.provider)
            react_config = _benchmark_react_config(self.config)
            react_config.run_dir = str(instance_dir / "agent_runtime")
            agent = ReActAgent(
                provider=provider,
                config=react_config,
                tools=registry,
                logger=logger,
                ui=recording_ui,
                workspace=workspace,
            )
            solve_started = time.monotonic()
            agent_result = await agent.run(build_swebench_prompt(instance))
            result.solve_seconds = time.monotonic() - solve_started
            result.answer = agent_result.answer
            result.steps = agent_result.steps
            result.tool_counts = dict(recording_ui.stats.get("tool_counts") or {})
            result.input_tokens = int(recording_ui.stats.get("input_tokens") or 0)
            result.output_tokens = int(recording_ui.stats.get("output_tokens") or 0)
            patch_started = time.monotonic()
            export = export_patch(workspace)
            valid, error = validate_patch_text(export.patch)
            result.patch_seconds = time.monotonic() - patch_started
            if not export.valid or not valid:
                result.state = InstanceState.FAILED.value
                result.failure_kind = FailureKind.PATCH.value
                result.failure_reason = export.error or error or "invalid patch"
            else:
                patch_path = save_patch(instance_dir / "patch.diff", export.patch)
                result.patch_path = str(patch_path)
                result.changed_files = export.changed_files
                if not export.patch.strip():
                    # The official Harness records empty patches separately and does
                    # not execute their tests.  Keep the prediction for auditability,
                    # but do not count it as patch-ready or send it for grading.
                    result.state = InstanceState.FAILED.value
                    stop_reason = recording_ui.stop_reason
                    if stop_reason == "deadline":
                        result.failure_kind = FailureKind.TIMEOUT.value
                        result.failure_reason = "agent reached its wall-clock deadline without a patch"
                    elif stop_reason == "interrupted":
                        result.failure_kind = FailureKind.CANCELLED.value
                        result.failure_reason = "agent was interrupted before producing a patch"
                    else:
                        result.failure_kind = FailureKind.PATCH.value
                        result.failure_reason = "agent produced an empty patch"
                else:
                    result.state = InstanceState.PATCH_READY.value
                    result.patch_generated = True
                async with self._prediction_lock:
                    upsert_prediction(
                        predictions_path,
                        prediction_record(
                            instance.instance_id,
                            export.patch,
                            model_name_or_path=_effective_model(self.config),
                        ),
                    )
            self._update_state(state_path, **result.to_dict())
        except asyncio.CancelledError:
            result.state = InstanceState.FAILED.value
            result.failure_kind = FailureKind.CANCELLED.value
            result.failure_reason = "solve cancelled"
            self._update_state(state_path, **result.to_dict())
            raise
        except Exception as exc:  # noqa: BLE001 - isolate one benchmark task
            result.state = InstanceState.FAILED.value
            result.failure_kind = _failure_kind(exc).value
            result.failure_reason = f"{type(exc).__name__}: {exc}"[:2000]
            self._update_state(state_path, **result.to_dict())
        finally:
            result.total_seconds = time.monotonic() - started
            self._update_state(state_path, total_seconds=result.total_seconds)
            if agent is not None:
                try:
                    await agent.fire_session_end("swebench_instance_end")
                except Exception:
                    pass
            if logger is not None:
                logger.close()
            if runtime is not None:
                try:
                    await runtime.close()
                except Exception:
                    pass
            if not self.config.keep_workspaces:
                # Keep patch/state/logs but remove the large mutable checkout for
                # both successes and failures.  ``--keep-workspaces`` is the
                # explicit debugging opt-in; a later retry always prepares a
                # fresh checkout when the old workspace is absent.
                import shutil

                shutil.rmtree(workspace, ignore_errors=True)
        return result

    async def _ensure_predictions(self, run_dir: Path, predictions_path: Path, results: Iterable[InstanceResult]) -> None:
        """Make resume/retry runs produce one prediction record per patch-ready task."""
        async with self._prediction_lock:
            for result in results:
                if result.state not in {
                    InstanceState.PATCH_READY.value,
                    InstanceState.RESOLVED.value,
                    InstanceState.UNRESOLVED.value,
                }:
                    continue
                patch_file = result.patch_path or str(run_dir / "instances" / _safe_instance_dir(result.instance_id) / "patch.diff")
                try:
                    patch = Path(patch_file).read_text(encoding="utf-8")
                except OSError:
                    continue
                upsert_prediction(
                    predictions_path,
                    prediction_record(
                        result.instance_id,
                        patch,
                        model_name_or_path=_effective_model(self.config),
                    ),
                )

    def _apply_harness_result(self, results: list[InstanceResult], run_dir: Path, harness: HarnessResult, eval_elapsed: float) -> None:
        for result in results:
            status = harness.status_by_instance.get(result.instance_id)
            if status == "resolved":
                result.state = InstanceState.RESOLVED.value
            elif status == "unresolved":
                result.state = InstanceState.UNRESOLVED.value
                result.failure_kind = FailureKind.HARNESS.value
                result.failure_reason = harness.failure_by_instance.get(result.instance_id, "tests did not resolve the patch")
            elif status == "failed":
                result.state = InstanceState.FAILED.value
                result.failure_kind = FailureKind.HARNESS.value
                result.failure_reason = harness.failure_by_instance.get(result.instance_id, "Harness error")
            elif result.state == InstanceState.PATCH_READY.value:
                # A missing report is an infrastructure error, not a false resolved.
                result.state = InstanceState.FAILED.value
                result.failure_kind = FailureKind.HARNESS.value
                result.failure_reason = harness.error or "Harness produced no per-instance report"
            result.evaluation_seconds = eval_elapsed
            result.total_seconds += eval_elapsed
            self._update_state(
                run_dir / "instances" / _safe_instance_dir(result.instance_id) / "state.json",
                state=result.state,
                evaluation_seconds=eval_elapsed,
                total_seconds=result.total_seconds,
                failure_kind=result.failure_kind,
                failure_reason=result.failure_reason,
            )

    def _write_run_metadata(self, run_dir: Path, selection: SWEbenchSelection, instances: Iterable[SWEbenchInstance]) -> None:
        data = {
            "run_id": self.config.run_id,
            "created_at": time.time(),
            "selection": {
                "dataset": selection.dataset,
                "split": selection.split,
                "instance_ids": [item.instance_id for item in instances],
                "source": selection.source,
            },
            "dataset_source": selection.dataset,
            "instance_base_digest": hashlib.sha256(
                "\n".join(f"{item.instance_id}\t{item.base_commit}" for item in instances).encode("utf-8")
            ).hexdigest(),
            "config": self.config.to_dict(),
            "swebench_package_version": _package_version("swebench"),
            "oracle_fields_excluded_from_agent": ["patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "pr_url"],
        }
        (run_dir / "run.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _write_summary(self, run_dir: Path, summary: dict[str, Any]) -> None:
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows = summary.get("instances", [])
        if rows:
            with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
                writer.writeheader()
                writer.writerows(rows)

    @staticmethod
    def _update_state(path: Path, **values: Any) -> None:
        current = _read_json(path)
        current.update(values)
        current.setdefault("instance_id", path.parent.name)
        current["updated_at"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def _benchmark_react_config(config: SWEbenchRunConfig) -> ReActConfig:
    tools = ToolSuiteConfig.from_dict({"shell": {"enabled": False}, "scheduler": {"enabled": False}})
    max_wall = config.max_wall_seconds if config.max_wall_seconds > 0 else None
    model = _effective_model(config) if (config.provider or "claude") == "claude" else config.model
    return ReActConfig(
        model=model,
        provider=config.provider or "claude",
        permission=PermissionMode.ACCEPTEDITS,
        system_prompt=BENCHMARK_SYSTEM_PROMPT,
        project_instructions=False,
        git_context=False,
        session_dir="",
        memory=MemoryConfig(enabled=False),
        skills=SkillsConfig(enabled=False),
        capabilities=CapabilitiesConfig(mode="disabled"),
        hooks=HooksConfig(enabled=False),
        sandbox=SandboxConfig(enabled=False),
        tools=tools,
        compression=CompressionConfig(use_llm_summary=False),
        max_wall_seconds=max_wall,
        max_steps=config.max_steps,
        stream=not bool(config.metadata.get("no_stream", False)),
        parallel_tools=False,
        max_tool_workers=1,
        max_api_concurrency=max(1, int(config.metadata.get("max_api_concurrency", 8))),
    )


def _provider_from_name(name: str) -> Any:
    value = (name or "claude").casefold()
    if value == "fake":
        return FakeProvider()
    if value == "openai":
        return OpenAIResponsesProvider()
    if value in {"openai-compat", "openai_compat"}:
        return OpenAICompatProvider()
    return ClaudeProvider()


def _effective_model(config: SWEbenchRunConfig) -> str:
    return config.model or ("claude-opus-4-8" if (config.provider or "claude") == "claude" else config.provider or "polaris")


def _safe_run_id(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return clean[:100] or f"run-{uuid.uuid4().hex[:8]}"


def _safe_instance_dir(value: str) -> str:
    raw = str(value)
    clean = "".join(char if char.isalnum() or char in "._-" else "_" for char in raw)
    if not clean:
        clean = "instance"
    if clean != raw or len(clean) > 180:
        clean = clean[:160] + "-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return clean[:180]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _failure_kind(exc: BaseException) -> FailureKind:
    text = str(exc).casefold()
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timed out" in text:
        return FailureKind.TIMEOUT
    if "patch" in text or "git diff" in text:
        return FailureKind.PATCH
    if "docker" in text or "runtime" in text or "container" in text or "instance image" in text:
        return FailureKind.RUNTIME
    if "clone" in text or "checkout" in text or "repository" in text or "prepare" in text:
        return FailureKind.PREPARE
    return FailureKind.AGENT


def _prediction_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "empty"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def aggregate_results(
    results: Iterable[InstanceResult],
    *,
    harness_result: HarnessResult | None,
    selected: int,
    duration: float,
) -> dict[str, Any]:
    rows = [item.to_dict() for item in results]
    resolved = sum(item["state"] == InstanceState.RESOLVED.value for item in rows)
    patch_ready = sum(
        bool(item.get("patch_generated"))
        or item["state"]
        in {InstanceState.PATCH_READY.value, InstanceState.RESOLVED.value, InstanceState.UNRESOLVED.value}
        for item in rows
    )
    graded = sum(item["state"] in {InstanceState.RESOLVED.value, InstanceState.UNRESOLVED.value} for item in rows)
    failures: dict[str, int] = {}
    failure_reasons: dict[str, int] = {}
    for item in rows:
        kind = item.get("failure_kind") or "none"
        if kind != "none":
            failures[kind] = failures.get(kind, 0) + 1
            reason = item.get("failure_reason") or kind
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    return {
        "selected": selected,
        "patch_ready": patch_ready,
        "graded": graded,
        "resolved": resolved,
        "patch_generation_rate": patch_ready / selected if selected else 0.0,
        "harness_completion_rate": graded / patch_ready if patch_ready else 0.0,
        "resolved_rate": resolved / selected if selected else 0.0,
        "resolved_rate_selected": resolved / selected if selected else 0.0,
        "resolved_rate_graded": resolved / graded if graded else 0.0,
        "duration_seconds": duration,
        "failure_kinds": failures,
        "failure_reasons": failure_reasons,
        "failure_reason_counts": failure_reasons,
        "harness_returncode": harness_result.returncode if harness_result is not None else None,
        "harness_duration_seconds": harness_result.duration if harness_result is not None else 0.0,
        "harness_error": harness_result.error if harness_result is not None else None,
        "instances": rows,
    }
