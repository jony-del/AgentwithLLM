import json
from pathlib import Path

from agent_core.models import Message, ToolCall
from agent_core.permission_rules import RuleSet
from agent_core.permission_types import PermissionRuleSource
from agent_core.permissions import PermissionPolicy
from agent_core.storage import JSONLRunLogger, read_events
from agent_core.tools.builtin import WriteTextFileTool
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.transcript import TranscriptStore


async def test_permission_event_has_required_schema_and_redacted_arguments(tmp_path: Path) -> None:
    tool = WriteTextFileTool(tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    logger = JSONLRunLogger(tmp_path / "runs", run_id="audit")
    policy = PermissionPolicy(
        "default",
        prompter=lambda name, risk, arguments: "once",
        workspace=tmp_path,
    )
    executor = ToolExecutor(registry, policy, logger=logger)

    await executor.execute_many(
        [ToolCall("write_text_file", {"path": "x.txt", "content": "API_TOKEN=super-secret-value"})]
    )
    event = next(item for item in read_events(logger.path) if item["event"] == "permission")
    serialized = json.dumps(event, ensure_ascii=False)

    required = {
        "tool",
        "arguments_summary",
        "mode",
        "final_behavior",
        "reason",
        "decision_source",
        "matched_rule",
        "rule_source",
        "sandboxed",
        "classifier_result",
        "parent_agent_id",
        "schema_version",
    }
    assert required <= event.keys()
    assert event["final_behavior"] == "allow"
    assert "super-secret-value" not in serialized
    assert event["arguments_summary"]["content"]["chars"] > 0


async def test_permission_audit_records_rule_source(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(WriteTextFileTool(tmp_path))
    logger = JSONLRunLogger(tmp_path / "runs", run_id="rule-source")
    rules = RuleSet.from_lists(
        deny=["write_text_file"], source=PermissionRuleSource.CLI
    )
    executor = ToolExecutor(
        registry,
        PermissionPolicy("bypass", rules=rules, workspace=tmp_path),
        logger=logger,
    )

    await executor.execute_many(
        [ToolCall("write_text_file", {"path": "x.txt", "content": "x"})]
    )
    event = next(item for item in read_events(logger.path) if item["event"] == "permission")

    assert event["final_behavior"] == "deny"
    assert event["rule_source"] == "cli"
    assert event["matched_rule"]["behavior"] == "deny"


async def test_sensitive_transcript_message_does_not_persist_content(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path / "sessions", tmp_path, "session")
    message = Message(
        "tool",
        "read_text_file: TOP_SECRET_WITHOUT_PATTERN",
        metadata={"sensitive": True},
    )

    await store.append_message(message)
    text = store.path.read_text(encoding="utf-8")

    assert "TOP_SECRET_WITHOUT_PATTERN" not in text
    assert "redacted-sensitive-tool-output" in text


async def test_global_log_redaction_strips_url_credentials_and_query(tmp_path: Path) -> None:
    logger = JSONLRunLogger(tmp_path, run_id="redact")
    await logger.write(
        "unit",
        {
            "url": "https://user:password@example.com/path?token=secret#fragment",
            "authorization": "Bearer abc.def",
        },
    )
    event = next(read_events(logger.path))

    assert event["url"] == "https://example.com/path"
    assert event["authorization"] == "<redacted>"
