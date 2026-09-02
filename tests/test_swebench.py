from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agent_core.benchmarks.swebench.dataset import SWEbenchDataset
from agent_core.benchmarks.swebench.harness import parse_harness_reports
from agent_core.benchmarks.swebench.harness import OfficialHarnessAdapter
from agent_core.benchmarks.swebench.models import SWEbenchRunConfig, SWEbenchSelection
from agent_core.benchmarks.swebench.patch import export_patch, validate_patch_text
from agent_core.benchmarks.swebench.prompt import build_swebench_prompt
from agent_core.benchmarks.swebench.runner import SWEbenchRunner
from agent_core.benchmarks.swebench.runtime import LocalRuntime, resolve_official_image
from agent_core.benchmarks.swebench.runtime import prepare_repository
from agent_core.providers import FakeProvider


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "initial")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_dataset_public_projection_and_prompt_oracle_guard() -> None:
    dataset = SWEbenchDataset.from_rows(
        [
            {
                "instance_id": "demo-1",
                "repo": "org/repo",
                "base_commit": "a" * 40,
                "problem_statement": "Fix the parser.",
                "patch": "SECRET GOLD PATCH",
                "test_patch": "SECRET TEST PATCH",
                "FAIL_TO_PASS": '["tests/test_parser.py::test_fix"]',
                "PASS_TO_PASS": ["tests/test_parser.py::test_other"],
            }
        ],
        include_gold=True,
    )
    instance = dataset["demo-1"]
    public = instance.public_dict()
    assert not {"patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS"} & public.keys()
    prompt = build_swebench_prompt(instance)
    assert "SECRET GOLD PATCH" not in prompt
    assert "SECRET TEST PATCH" not in prompt
    assert "Fix the parser." in prompt


def test_official_image_fallback_uses_dockerhub_safe_instance_id(monkeypatch) -> None:
    monkeypatch.setenv("SWEBENCH_IMAGE_NAMESPACE", "swebench")
    monkeypatch.delenv("SWEBENCH_INSTANCE_IMAGE", raising=False)
    instance = SWEbenchDataset.from_rows(
        [{
            "instance_id": "org__repo-123",
            "repo": "org/repo",
            "base_commit": "a" * 40,
            "problem_statement": "Fix it",
        }],
        include_gold=False,
    )["org__repo-123"]
    assert resolve_official_image(instance) == "swebench/sweb.eval.x86_64.org_1776_repo-123:latest"


def test_patch_export_includes_tracked_and_untracked(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    (repo / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "new.py").write_text("print('new')\n", encoding="utf-8")
    exported = export_patch(repo)
    assert exported.valid
    assert "module.py" in exported.patch
    assert "new.py" in exported.patch
    assert set(exported.changed_files) == {"module.py", "new.py"}
    assert validate_patch_text(exported.patch) == (True, None)


def test_patch_export_survives_agent_commit(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    instance = SWEbenchDataset.from_rows(
        [{
            "instance_id": "demo-commit",
            "repo": str(repo),
            "base_commit": base,
            "problem_statement": "change the value",
        }],
        include_gold=False,
    )["demo-commit"]
    workspace = tmp_path / "workspace"
    prepare_repository(instance, workspace, source_dir=repo)
    assert _git(workspace, "rev-parse", "HEAD") == base
    assert _git(workspace, "remote") == ""
    (workspace / "module.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(workspace, "add", ".")
    _git(workspace, "commit", "--quiet", "-m", "agent commit")
    exported = export_patch(workspace)
    assert exported.valid
    assert "VALUE = 3" in exported.patch


def test_harness_report_parser(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text(
        json.dumps({"resolved_ids": ["a"], "unresolved_ids": ["b"], "error_ids": {"c": "timeout"}}),
        encoding="utf-8",
    )
    statuses, failures = parse_harness_reports(tmp_path)
    assert statuses == {"a": "resolved", "b": "unresolved", "c": "failed"}
    assert failures["c"] == "timeout"


def test_harness_report_parser_accepts_per_instance_mapping(tmp_path: Path) -> None:
    (tmp_path / "instance.json").write_text(
        json.dumps({"org__repo-1": {"resolved": True}, "org__repo-2": {"resolved": False}}),
        encoding="utf-8",
    )
    statuses, failures = parse_harness_reports(tmp_path)
    assert statuses == {"org__repo-1": "resolved", "org__repo-2": "unresolved"}
    assert failures == {}


def test_harness_report_parser_does_not_treat_unresolved_as_resolved(tmp_path: Path) -> None:
    (tmp_path / "instance.json").write_text(
        json.dumps({"org__repo-1": {"status": "unresolved"}}),
        encoding="utf-8",
    )
    statuses, _ = parse_harness_reports(tmp_path)
    assert statuses == {"org__repo-1": "unresolved"}


def test_harness_report_parser_handles_nested_result_wrappers(tmp_path: Path) -> None:
    (tmp_path / "nested.json").write_text(
        json.dumps({"results": {"org__repo-1": {"status": "passed"}}}),
        encoding="utf-8",
    )
    statuses, _ = parse_harness_reports(tmp_path)
    assert statuses == {"org__repo-1": "resolved"}


def test_harness_command_uses_absolute_local_dataset_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks.json"
    dataset_path.write_text("[]", encoding="utf-8")
    command = OfficialHarnessAdapter(python="python").build_command(
        dataset=str(dataset_path),
        split="test",
        predictions_path=tmp_path / "predictions.jsonl",
        run_id="run",
        report_dir=tmp_path / "report",
        instance_ids=("a",),
    )
    assert command[command.index("--dataset_name") + 1] == str(dataset_path.resolve())


def test_local_runner_writes_official_prediction_artifact(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    row = {
        "instance_id": "demo-1",
        "repo": str(repo),
        "base_commit": base,
        "problem_statement": "Inspect the repository and describe the requested change.",
        "patch": "do not expose",
    }
    def loader(*args, **kwargs):  # noqa: ARG001
        return SWEbenchDataset.from_rows([row], dataset="memory", split="dev", include_gold=False)
    config = SWEbenchRunConfig(
        dataset="memory",
        split="dev",
        output_dir=str(tmp_path / "runs"),
        model="fake",
        provider="fake",
        evaluate=False,
        keep_workspaces=True,
        max_steps=2,
        metadata={"runtime": "local", "source_dir": str(repo)},
    )
    import asyncio

    summary = asyncio.run(
        SWEbenchRunner(
            config,
            provider=FakeProvider(),
            runtime_factory=lambda instance, workspace, image: LocalRuntime(workspace),
            dataset_loader=loader,
        ).run(SWEbenchSelection("memory", "dev", ("demo-1",)))
    )
    assert summary["selected"] == 1
    assert summary["patch_generation_rate"] == 0.0
    run_dir = next((tmp_path / "runs").iterdir())
    record = json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    assert record["instance_id"] == "demo-1"
    assert set(record) == {"instance_id", "model_name_or_path", "model_patch"}
