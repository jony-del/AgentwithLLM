from agent_core.models import ToolRisk
from agent_core.providers import FakeProvider
from agent_core.react import ReActAgent
from agent_core.session import SessionContext
from agent_core.tools.planning import UpdateTodosTool


async def test_update_todos_stores_and_renders() -> None:
    session = SessionContext()
    tool = UpdateTodosTool(session)
    result = await tool.run(
        {
            "todos": [
                {"content": "Read the code", "status": "completed"},
                {"content": "Write the fix", "status": "in_progress"},
                {"content": "Run tests", "status": "pending"},
            ]
        }
    )
    assert result.ok
    assert result.metadata["count"] == 3
    assert "[x] Read the code" in result.content
    assert "[~] Write the fix" in result.content
    assert "[ ] Run tests" in result.content
    # State persisted on the session for the next tool turn.
    assert [t.content for t in session.todos.items()] == ["Read the code", "Write the fix", "Run tests"]


async def test_update_todos_replaces_previous_list() -> None:
    session = SessionContext()
    tool = UpdateTodosTool(session)
    await tool.run({"todos": [{"content": "old"}]})
    await tool.run({"todos": [{"content": "new"}]})
    assert [t.content for t in session.todos.items()] == ["new"]


async def test_update_todos_drops_empty_and_coerces_bad_status() -> None:
    session = SessionContext()
    tool = UpdateTodosTool(session)
    await tool.run({"todos": [{"content": "  "}, {"content": "keep", "status": "bogus"}]})
    items = session.todos.items()
    assert len(items) == 1
    assert items[0].content == "keep"
    assert items[0].status == "pending"  # bad status coerced


async def test_update_todos_notifies_ui() -> None:
    seen: list = []
    session = SessionContext(ui_notify=seen.append)
    await UpdateTodosTool(session).run({"todos": [{"content": "x"}]})
    assert len(seen) == 1
    assert seen[0][0].content == "x"


async def test_update_todos_rejects_non_list() -> None:
    result = await UpdateTodosTool(SessionContext()).run({"todos": "nope"})
    assert not result.ok
    assert result.metadata["error_type"] == "BadArgs"


def test_update_todos_is_read_risk() -> None:
    assert UpdateTodosTool().risk is ToolRisk.READ


def test_agent_binds_session_into_planning_tool() -> None:
    agent = ReActAgent(provider=FakeProvider())
    tool = agent.registry.get("update_todos")
    # The agent rebound the tool to its own live session.
    assert tool.session is agent.session
